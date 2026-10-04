import sys
import asyncio
import json
import re
from google import genai
from google.genai import types
from playwright.async_api import async_playwright
import random

def clean_json_response(raw_text):
    """Safely strips markdown decoration syntaxes from model output generation."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\n|```$', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()

async def main():
    if len(sys.argv) < 3:
        print("Error: Missing execution runtime parameters.")
        return

    query = sys.argv[1]
    raw_keys_string = sys.argv[2]
    
    # 1. Parse all available keys properly from mainapp.py
    api_keys = [k.strip() for k in raw_keys_string.split(',')] if ',' in raw_keys_string else [raw_keys_string]
    
    if not api_keys:
        print("Error: No API keys provided to Sandbox.")
        return

    # 2. Initialize the modern native client SDK with a random starting key
    current_key = random.choice(api_keys)
    client = genai.Client(api_key=current_key)
    
    action_history = []

    async with async_playwright() as p:
        print("Launching Persistent User Profile Browser on Virtual Screen :99...")
        user_data_dir = "/app/agent_scripts/google_user_profile"
        
        context = await p.chromium.launch_persistent_context(
            user_data_dir,
            headless=False,
            viewport={"width": 1280, "height": 720},
            args=[
                '--no-sandbox', 
                '--disable-dev-shm-usage', 
                '--disable-gpu',
                '--disable-blink-features=AutomationControlled'
            ],
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        
        page = context.pages[0] if context.pages else await context.new_page()
        
        print("Navigating to baseline web search ecosystem...")
        await page.goto("https://www.google.com")
        await asyncio.sleep(4)

        max_steps = 60  
        final_data_summary = None

        for step in range(max_steps):
            print(f"\n[Step {step + 1}] Processing browser interaction frame...")
            
            screenshot_bytes = await page.screenshot(type="png")
            
            prompt = f"""
            You are an advanced autonomous Vision-Action Web Engine operating exactly like Manus AI.
            Your exact target goal objective is: "{query}"
            
            Current Viewport Canvas Frame Resolution: 1280x720 pixels.
            
            Analyze the provided browser image interface carefully. Determine where to click or type based purely on visual elements.
            
            Return exactly one action command object structured inside a single clean JSON block. Do not append regular prose text.
            
            Valid Target JSON JSON Output Schemas:
            1. {{"action": "click", "x": 640, "y": 300, "reason": "Clicking the main web search box"}}
            2. {{"action": "type", "text": "latest honda car prices on pakwheels", "reason": "Typing target search terms"}}
            3. {{"action": "press_key", "key": "Enter", "reason": "Submitting search query form input"}}
            4. {{"action": "wait", "seconds": 5, "reason": "Waiting for elements to complete dynamic client render"}}
            5. {{"action": "finish", "data": "Comprehensive bulleted report outlining variants, estimated pricing listings, and market data found.", "reason": "Research task solved completely"}}
            
            CRITICAL DESIGN GUIDELINES:
            - If a cookie banner, location prompt, or modal pop-up is blocking the screen, immediately target the coordinate matching its close/decline option to throw it out.
            - Coordinates must be strict integers fitting within X: 0-1280 and Y: 0-720.
            
            Recent Actions History:
            {action_history[-6:]}
            """
            
            # --- NAYA FIX: Robust Inner Loop for Native Key Rotation ---
            response = None
            attempts = 0
            max_attempts = len(api_keys) * 2
            
            while attempts < max_attempts:
                try:
                    response = client.models.generate_content(
                        model='gemini-flash-latest',
                        contents=[
                            types.Part.from_bytes(data=screenshot_bytes, mime_type='image/png'),
                            prompt
                        ]
                    )
                    break # Success! Break out of the retry loop.
                    
                except Exception as e:
                    error_str = str(e).lower()
                    # Catch API Key issues, 429 Quota Exhausted, or 503 Overloaded
                    if "429" in error_str or "exhausted" in error_str or "api_key" in error_str or "400" in error_str or "503" in error_str:
                        print(f"System Notification: Key Exhausted or Invalid. Rotating API Key in Sandbox...")
                        current_key = random.choice(api_keys)
                        client = genai.Client(api_key=current_key) # Destroy and rebuild the client natively
                        attempts += 1
                        await asyncio.sleep(2)
                    else:
                        print(f"Visual Loop Parser Exception on Step {step + 1}: {str(e)}")
                        break # Break retry loop, move to JSON extraction (which will fail gracefully)
            
            if not response:
                print("Critical Engine Failure: Unable to generate response after rotating all keys.")
                continue
            # -----------------------------------------------------------

            try:
                raw_json = clean_json_response(response.text)
                command = json.loads(raw_json)
                
                reasoning = command.get("reason", "No reason provided")
                action_type = command.get("action")
                print(f"Selected Action Path: {reasoning}")
                
                action_history.append(f"Action: {action_type} | Reason: {reasoning}")

                if action_type == "click":
                    await page.mouse.click(command["x"], command["y"])
                    await asyncio.sleep(2)
                elif action_type == "type":
                    await page.keyboard.type(command["text"], delay=60)
                    await asyncio.sleep(1)
                elif action_type == "press_key":
                    await page.keyboard.press(command["key"])
                    await asyncio.sleep(3)
                elif action_type == "wait":
                    await asyncio.sleep(command.get("seconds", 4))
                elif action_type == "finish":
                    final_data_summary = command.get("data", "Research workflow successfully finished.")
                    print("\n[Engine Notification]: Final task extraction completed.")
                    break

            except Exception as e:
                print(f"JSON Extraction Exception on Step {step + 1}: {str(e)}")
                await asyncio.sleep(2)

        # Print data directly to standard output stream for mainapp.py collection
        if final_data_summary:
            print(final_data_summary)
        else:
            print("The visual agent reached its step execution limit without gathering text output data.")
            
        await asyncio.sleep(2)
        await context.close()

if __name__ == "__main__":
    asyncio.run(main())