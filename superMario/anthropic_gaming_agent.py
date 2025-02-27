import time
import os
import pyautogui
import base64
import anthropic
import numpy as np
import concurrent.futures
import re

def encode_image(image_path):
    """
    Read a file from disk and return its contents as a base64-encoded string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def log_output(thread_id, log_text):
    """
    Logs output to `cache/thread_{thread_id}/output.log`
    """
    thread_folder = f"cache/thread_{thread_id}"
    os.makedirs(thread_folder, exist_ok=True)
    
    log_path = os.path.join(thread_folder, "output.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(log_text + "\n\n")

def extract_python_code(content):
    """
    Extracts Python code from the assistant's response.
    - Detects code enclosed in triple backticks (```python ... ```)
    - If no triple backticks are found, returns the raw content.
    """
    match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return content.strip()

def worker_short(thread_id, offset):
    """
    Worker function for short-term (1 second) motion control.
    1) Sleeps 'offset' seconds before starting (to stagger starts).
    2) Continuously takes screenshots, calls Anthropic with streaming output, logs latency, executes returned code, etc.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    all_response_time = []

    time.sleep(offset)
    print(f"[Thread {thread_id} - SHORT] Starting after {offset}s delay...")

    short_prompt = (
        f"Analyze the current game state and generate PyAutoGUI code to control Mario "
        "for up to the next 1 second.\n"
        "For every game state shown, 2~5 seconds might have elapsed when the generated code gets to execute.\n"
        "Your objective is to avoid obstacles, enemies, and hazards.\n"

        "### General Controls:\n"
        "- Press 'Enter' to start the game ONLY IF the game hasn't started.\n"
        "  Otherwise the game will be paused.\n"
        "- Press the right arrow to move forward.\n"
        "- Press 'X' along with right/left arrow to jump over obstacles or gaps. Be very careful with gaps, do lopped jumps if necessary.\n\n"

        "### Strategies and Caveats:\n"
        "- Whenever a gap is detected, AVOID jumping over the gap. Only do small position adjustments to prepare for big jump.\n"
        "- If an obstacle or enemy is near, move/jump left to dodge.\n"
        "- If an enemy is detected, do one big jump ONLY IF very confident, ortherwise do consecutive short jumps.\n"
        "- Prioritize survival over speed: if in doubt, take a more defensive approaches like moving to the left (move back).\n"
        "- Sleep and do nothing if no obvious danger."

        "### Output Format:\n"
        "- Output ONLY the Python code for PyAutoGUI commands.\n"
        "- Include brief comments for each action.\n"
    )

    try:
        while True:
            screen_width, screen_height = pyautogui.size()
            region = (0, 0, screen_width, screen_height)
            screenshot = pyautogui.screenshot(region=region)

            thread_folder = f"cache/thread_{thread_id}"
            os.makedirs(thread_folder, exist_ok=True)

            screenshot_path = os.path.join(thread_folder, "screenshot.png")
            screenshot.save(screenshot_path)

            base64_image = encode_image(screenshot_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": short_prompt
                        },
                    ],
                }
            ]

            start_time = time.time()

            with client.messages.stream(
                max_tokens=1024,
                messages=messages,
                temperature=0,
                model="claude-3-7-sonnet-20250219", # claude-3-5-sonnet-20241022 # claude-3-7-sonnet-20250219
            ) as stream:
                partial_chunks = []
                for chunk in stream.text_stream:
                    partial_chunks.append(chunk)

            end_time = time.time()
            latency = end_time - start_time
            all_response_time.append(latency)

            print(f"[Thread {thread_id} - SHORT] Request latency: {latency:.2f}s")
            avg_latency = np.mean(all_response_time)
            print(f"[Thread {thread_id} - SHORT] Latencies: {all_response_time}")
            print(f"[Thread {thread_id} - SHORT] Average latency: {avg_latency:.2f}s")

            generated_code_str = "".join(partial_chunks)
            print(f"\n[Thread {thread_id} - SHORT] --- Generation (Streaming) ---\n{generated_code_str}\n")

            clean_code = extract_python_code(generated_code_str)
            log_output(thread_id, f"[Thread {thread_id} - SHORT] Python code to be executed:\n{clean_code}\n")
            print(f"[Thread {thread_id} - SHORT] Python code to be executed:\n{clean_code}\n")

            try:
                exec(clean_code)
            except Exception as e:
                print(f"[Thread {thread_id} - SHORT] Error executing code: {e}")

    except KeyboardInterrupt:
        print(f"[Thread {thread_id} - SHORT] Interrupted by user. Exiting...")

def worker_long(thread_id, offset):
    """
    Worker function for long-term (2 seconds) motion control.
    1) Sleeps 'offset' seconds before starting (to stagger starts).
    2) Continuously takes screenshots, calls Anthropic with streaming output, logs latency, executes returned code, etc.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    all_response_time = []

    time.sleep(offset)
    print(f"[Thread {thread_id} - LONG] Starting after {offset}s delay...")

    long_prompt = (
        f"Analyze the current game state and generate PyAutoGUI code to control Mario "
        "for the next 2~3 seconds.\n"
        "For every game state shown, 2~5 seconds might have elapsed when the generated code gets to execute.\n"
        "Your objective is to make progress while avoiding obstacles, enemies, and hazards.\n"

        "### General Controls:\n"
        "- Press 'Enter' to start the game ONLY IF the game hasn't started.\n"
        "  Otherwise the game will be paused.\n"
        "- Press the right arrow to move forward.\n"
        "- Press 'X' along with right/left arrow to jump over obstacles or gaps. Be very careful with gaps, do lopped jumps if necessary.\n\n"

        "### Strategies and Caveats:\n"
        "- Don't move too fast, as unseen enemies may appear from off-screen.\n"
        "- If an obstacle or enemy is near, move forward in small increments and be ready to jump.\n"
        "- Avoid walking forward without jumping as Mario can run into off-screen enemies.\n"
        "- If a gap is detected, do big jump over the gap ONLY IF very confident. Otherwise move left and adjust.\n"
        "- Prioritize survival over speed: if in doubt, take a more defensive approaches like moving to the left (move back).\n"
        "- Secondary goal: only if very safe, collect as many question mark blocks as possible.\n\n"

        "### Output Format:\n"
        "- Output ONLY the Python code for PyAutoGUI commands.\n"
        "- Include brief comments for each action.\n"
    )

    try:
        while True:
            screen_width, screen_height = pyautogui.size()
            region = (0, 0, screen_width, screen_height)
            screenshot = pyautogui.screenshot(region=region)

            thread_folder = f"cache/thread_{thread_id}"
            os.makedirs(thread_folder, exist_ok=True)

            screenshot_path = os.path.join(thread_folder, "screenshot.png")
            screenshot.save(screenshot_path)

            base64_image = encode_image(screenshot_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": long_prompt
                        },
                    ],
                }
            ]

            start_time = time.time()

            with client.messages.stream(
                max_tokens=1024,
                messages=messages,
                temperature=0,
                model="claude-3-7-sonnet-20250219", # claude-3-5-sonnet-20241022 # claude-3-7-sonnet-20250219
            ) as stream:
                partial_chunks = []
                for chunk in stream.text_stream:
                    partial_chunks.append(chunk)

            end_time = time.time()
            latency = end_time - start_time
            all_response_time.append(latency)

            print(f"[Thread {thread_id} - LONG] Request latency: {latency:.2f}s")
            avg_latency = np.mean(all_response_time)
            print(f"[Thread {thread_id} - LONG] Latencies: {all_response_time}")
            print(f"[Thread {thread_id} - LONG] Average latency: {avg_latency:.2f}s")

            generated_code_str = "".join(partial_chunks)
            print(f"\n[Thread {thread_id} - LONG] --- Generation (Streaming) ---\n{generated_code_str}\n")

            clean_code = extract_python_code(generated_code_str)
            log_output(thread_id, f"[Thread {thread_id} - LONG] Python code to be executed:\n{clean_code}\n")
            print(f"[Thread {thread_id} - LONG] Python code to be executed:\n{clean_code}\n")

            try:
                exec(clean_code)
            except Exception as e:
                print(f"[Thread {thread_id} - LONG] Error executing code: {e}")

    except KeyboardInterrupt:
        print(f"[Thread {thread_id} - LONG] Interrupted by user. Exiting...")

def main():
    """
    Spawns a number of short-term and long-term workers.
    """
    num_threads = 14
    offsets = [i * 0.5 for i in range(num_threads)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            if i % 2 == 0:
                executor.submit(worker_long, i, offsets[i])
            else:
                executor.submit(worker_short, i, offsets[i])

        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nMain thread interrupted. Exiting all threads...")

if __name__ == "__main__":
    main()
