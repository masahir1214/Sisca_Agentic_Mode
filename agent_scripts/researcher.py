import sys
import os
import re
import json
import time
import random
import hashlib
import asyncio
from datetime import datetime

from google import genai
from google.genai import types
from playwright.async_api import async_playwright

MODEL = "gemini-flash-latest"
VIEW_W, VIEW_H = 1280, 720
MAX_STEPS = 60
TIME_BUDGET_SEC = 17 * 60          # mainapp.py kills us at 20 min, so finish gracefully before that
MAX_CONSECUTIVE_FAILS = 5          # stop early (with the real reason) instead of looping silently
LOG_FILE = "/app/agent_scripts/agent_run.log"   # visible on the host via the ./agent_scripts volume

# Errors for which switching to another API key can help
ROTATE_HINTS = ("429", "resource_exhausted", "exhausted", "quota", "api_key", "api key",
                "400", "403", "permission_denied", "500", "503", "unavailable", "overloaded", "deadline")


def log(msg):
    """Progress goes to stderr + a log file. stdout is reserved for the FINAL report only
    (mainapp.py shows stdout to the user)."""
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- model helpers
class KeyPool:
    def __init__(self, keys):
        self.keys = keys
        self.i = random.randrange(len(keys))
        self.client = genai.Client(api_key=keys[self.i])

    def rotate(self):
        self.i = (self.i + 1) % len(self.keys)
        self.client = genai.Client(api_key=self.keys[self.i])


def response_text(resp):
    """resp.text can be None (blocked / empty / thinking-only). Fall back to raw parts."""
    try:
        if resp.text:
            return resp.text
    except Exception:
        pass
    parts = []
    for cand in (getattr(resp, "candidates", None) or []):
        for part in (getattr(getattr(cand, "content", None), "parts", None) or []):
            if getattr(part, "text", None):
                parts.append(part.text)
    return "".join(parts)


def extract_command(text):
    """Find the first JSON object in the model output, even with prose / ``` fences around it."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


async def ask_model(pool, screenshot_bytes, prompt):
    """Returns (text, error). Rotates keys on quota/key/overload errors; backs off after a full pass."""
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
    max_attempts = len(pool.keys) * 2
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await asyncio.to_thread(
                pool.client.models.generate_content,
                model=MODEL,
                contents=[types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png"), prompt],
                config=config,
            )
            return response_text(resp), None
        except Exception as e:
            last_error = str(e)
            low = last_error.lower()
            if any(h in low for h in ROTATE_HINTS):
                log(f"  API error (attempt {attempt}/{max_attempts}), rotating key: {last_error[:160]}")
                pool.rotate()
                # after we've tried every key once, wait a bit (per-minute quota)
                await asyncio.sleep(8 if attempt % len(pool.keys) == 0 else 1)
            else:
                log(f"  Non-retryable API error: {last_error[:300]}")
                return None, last_error
    return None, last_error


# ---------------------------------------------------------------- browser helpers
def to_px(x, y):
    """Model is asked for a 0-1000 grid (Gemini's native coordinate space). If it sends >1000 it
    clearly used pixels, so keep those."""
    x, y = float(x), float(y)
    if x > 1000 or y > 1000:
        px, py = x, y
    else:
        px, py = x / 1000 * VIEW_W, y / 1000 * VIEW_H
    return int(min(max(px, 0), VIEW_W - 1)), int(min(max(py, 0), VIEW_H - 1))


async def execute(page, cmd):
    action = cmd.get("action")
    if action == "click":
        px, py = to_px(cmd["x"], cmd["y"])
        await page.mouse.click(px, py)
        await asyncio.sleep(2)
    elif action == "type":
        await page.keyboard.type(str(cmd["text"]), delay=60)
        if cmd.get("submit"):
            await page.keyboard.press("Enter")
            await asyncio.sleep(3)
        else:
            await asyncio.sleep(1)
    elif action == "press_key":
        await page.keyboard.press(str(cmd["key"]))
        await asyncio.sleep(3)
    elif action == "scroll":
        await page.mouse.move(VIEW_W // 2, VIEW_H // 2)
        amount = 600 if str(cmd.get("direction", "down")).lower() != "up" else -600
        await page.mouse.wheel(0, amount)
        await asyncio.sleep(1.5)
    elif action == "goto":
        url = str(cmd["url"]).strip()
        if not re.match(r"^https?://", url):
            url = "https://" + url
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(2)
    elif action == "wait":
        await asyncio.sleep(min(max(float(cmd.get("seconds", 4)), 1), 15))
    else:
        raise ValueError(f"unknown action '{action}'")


def build_prompt(query, url, title, history):
    hist = "\n".join(history[-8:]) if history else "(none yet - this is the first step)"
    return f"""
You are an autonomous web-browsing agent. Your goal: "{query}"

You see a screenshot of the browser page (viewport {VIEW_W}x{VIEW_H}).
Current URL: {url}
Page title: {title}

Return EXACTLY ONE action as a single JSON object (no prose). Allowed actions:
1. {{"action": "click", "x": 500, "y": 420, "reason": "..."}}  -> x,y are on a 0-1000 grid: (0,0)=top-left, (1000,1000)=bottom-right of the screenshot
2. {{"action": "type", "text": "infinix mobile prices priceoye", "submit": true, "reason": "..."}}  -> types into the focused field; submit=true also presses Enter
3. {{"action": "press_key", "key": "Enter", "reason": "..."}}
4. {{"action": "scroll", "direction": "down", "reason": "..."}}  -> scroll to read more of the page
5. {{"action": "goto", "url": "https://www.priceoye.pk", "reason": "..."}}  -> open a URL directly
6. {{"action": "wait", "seconds": 5, "reason": "..."}}
7. {{"action": "finish", "data": "Bulleted report with the actual names/prices/values you read from the pages.", "reason": "..."}}

RULES:
- To type into a field, FIRST click it (one step), THEN type (next step). Check the previous actions below.
- Close cookie banners / pop-ups first if they block the page.
- If Google shows a CAPTCHA / "unusual traffic" page, use goto with https://duckduckgo.com or https://www.bing.com instead.
- If the task names a website, open that website (via search or goto), read the real data from the screen (scroll if needed), then finish.
- Only use "finish" when you have actual data from the pages. Never invent numbers.
- If a previous action says "screen did not change", do something DIFFERENT (other coordinates, other action).

Previous actions:
{hist}
"""


# ---------------------------------------------------------------- main loop
async def agent_loop(context, page, query, pool):
    history = []
    deadline = time.time() + TIME_BUDGET_SEC
    fails = 0
    last_hash = None
    last_error = "unknown"

    for step in range(1, MAX_STEPS + 1):
        if time.time() > deadline:
            return None, "time budget (17 min) used up"

        # a click may open a new tab - follow it
        if len(context.pages) > 1 and context.pages[-1] is not page:
            page = context.pages[-1]
            try:
                await page.bring_to_front()
            except Exception:
                pass

        try:
            shot = await page.screenshot(type="png")
            url = page.url
            title = await page.title()
        except Exception as e:
            fails += 1
            last_error = f"screenshot failed: {e}"
            log(f"[Step {step}] {last_error}")
            if fails >= MAX_CONSECUTIVE_FAILS:
                return None, last_error
            await asyncio.sleep(2)
            continue

        cur_hash = hashlib.md5(shot).hexdigest()
        if history and cur_hash == last_hash and not history[-1].endswith("[screen did not change]"):
            history[-1] += " [screen did not change]"
        last_hash = cur_hash

        log(f"[Step {step}] url={url}")
        text, err = await ask_model(pool, shot, build_prompt(query, url, title, history))
        cmd = extract_command(text)

        if not cmd:
            fails += 1
            last_error = err or f"model returned no usable JSON: {(text or '')[:120]!r}"
            log(f"  no action ({fails}/{MAX_CONSECUTIVE_FAILS}): {last_error}")
            if fails >= MAX_CONSECUTIVE_FAILS:
                return None, last_error
            await asyncio.sleep(2)
            continue

        action = cmd.get("action")
        reason = cmd.get("reason", "")
        log(f"  action={action} {json.dumps({k: v for k, v in cmd.items() if k not in ('reason', 'data')}, ensure_ascii=False)} | {reason}")

        if action == "finish":
            return cmd.get("data") or "Research workflow finished.", None

        try:
            await execute(page, cmd)
            fails = 0
            params = {k: v for k, v in cmd.items() if k not in ("action", "reason")}
            history.append(f"{len(history) + 1}. {action} {json.dumps(params, ensure_ascii=False)} ({reason})")
        except Exception as e:
            fails += 1
            last_error = f"action '{action}' failed: {e}"
            log(f"  {last_error}")
            history.append(f"{len(history) + 1}. {action} FAILED: {str(e)[:80]}")
            if fails >= MAX_CONSECUTIVE_FAILS:
                return None, last_error

    return None, f"step limit ({MAX_STEPS}) reached"


async def main():
    if len(sys.argv) < 3:
        print("Error: Missing execution runtime parameters.")
        return

    try:
        open(LOG_FILE, "w").close()
    except Exception:
        pass

    query = sys.argv[1]
    api_keys = [k.strip() for k in sys.argv[2].split(",") if k.strip()]
    if not api_keys:
        print("Error: No API keys provided to Sandbox.")
        return

    log(f"Task: {query} | keys: {len(api_keys)}")
    pool = KeyPool(api_keys)

    async with async_playwright() as p:
        log("Launching browser on virtual screen :99 ...")
        context = await p.chromium.launch_persistent_context(
            "/app/agent_scripts/google_user_profile",
            headless=False,
            viewport={"width": VIEW_W, "height": VIEW_H},
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-blink-features=AutomationControlled'
            ],
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = context.pages[0] if context.pages else await context.new_page()

        try:
            await page.goto("https://www.google.com")
        except Exception as e:
            log(f"Initial navigation problem (continuing): {e}")
        await asyncio.sleep(3)

        result, reason = await agent_loop(context, page, query, pool)

        if result:
            log("Finished with a report.")
            print(result)
        else:
            log(f"Stopped without a report: {reason}")
            print(f"The visual agent stopped before finishing. Reason: {reason}")

        await asyncio.sleep(2)
        await context.close()


if __name__ == "__main__":
    asyncio.run(main())
