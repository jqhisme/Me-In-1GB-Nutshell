import json
import os
import re
import random
import subprocess
import textwrap
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import ImageFont
from luma.core.render import canvas
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from radxa_hid_keyboard import OptionalHidKeyboardWriter, setup as setup_hid_keyboard

I2C_BUS = 3       # OLED detected at 0x3C on this bus
I2C_ADDRESS = 0x3C
DISPLAY_CHARS_PER_SECOND = 20  # Lower this for a slower text reveal
PROMPT_WORDS_PER_SECOND = 3
BOARD_HOME = Path("/home/radxa")
MODEL = BOARD_HOME / "Desktop/MeInANutshell/models/checkpoint-270_Q8_0.gguf"
LLAMA_SERVER = BOARD_HOME / "llama.cpp/build/bin/llama-server"
SERVER_URL = "http://127.0.0.1:8080"

if os.geteuid() != 0:
    raise SystemExit("USB HID setup requires root; run this script with sudo")

# setup the device as a HID keyboard
setup_hid_keyboard()

# Read prompts relative to this script, including when started by systemd.
with Path(__file__).with_name("rag_corpus_deduped.jsonl").open(encoding="utf-8") as f:
    rag_docs = [json.loads(line)["text"] for line in f if line.strip()]
if not rag_docs:
    raise SystemExit("No prompts found in rag_corpus_deduped.jsonl")


device = sh1106(i2c(port=I2C_BUS, address=I2C_ADDRESS), width=128, height=64)
font = ImageFont.load_default()
char_width = max(1, int(font.getlength("M")))
line_height = 11
columns = int((device.width // char_width) * 1.5)
rows = device.height // line_height


def show(message):
    lines = []
    for paragraph in message[-3000:].split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=columns) or [""])

    with canvas(device) as draw:
        for row, line in enumerate(lines[-rows:]):
            draw.text((0, row * line_height), line, font=font, fill="white")

command = [
    str(LLAMA_SERVER),
    "-m", str(MODEL),
    "--host", "127.0.0.1", "--port", "8080",
    "--no-webui", "-np", "1",
    "-c", "512", "-b", "64", "-ub", "32", "-t", "2",
]

def wait_for_model(process, first_prompt, keyboard):
    """Reveal only the first prompt while the server loads the model."""
    words = re.findall(r"\S+\s*", "My name is Qihan(Christoff) Jiang. " + first_prompt)
    visible_prompt = ""
    next_word_at = time.monotonic()
    word_index = 0

    while True:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup: {process.returncode}")
        if word_index < len(words) and time.monotonic() >= next_word_at:
            visible_prompt += words[word_index]
            keyboard.send_character(words[word_index])
            word_index += 1
            show(visible_prompt)
            next_word_at = time.monotonic() + 1 / PROMPT_WORDS_PER_SECOND
        try:
            with urlopen(SERVER_URL + "/health", timeout=0.2) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            pass  # The server is still starting or loading the model.
        time.sleep(0.05)


def generate(prompt, keyboard):
    """Stream one completion while keeping llama-server alive."""
    payload = {
        "prompt": prompt,
        "n_predict": 512,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "min_p": 0.05,
        "cache_prompt": False,
        "stream": True,
    }
    request = Request(
        SERVER_URL + "/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    generated = ""
    last_refresh = 0.0
    with urlopen(request) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            for character in event.get("content", ""):
                generated += character
                keyboard.send_character(character)
                print(character, end="", flush=True)
                if time.monotonic() - last_refresh >= 0.25:
                    show(generated)
                    last_refresh = time.monotonic()
                if DISPLAY_CHARS_PER_SECOND > 0:
                    time.sleep(1 / DISPLAY_CHARS_PER_SECOND)
            if event.get("stop"):
                break

    show(generated if generated.strip() else "No output. Check SSH terminal.")
    print("\n")


prompt_index = random.randrange(len(rag_docs))
first_prompt = rag_docs[prompt_index]
process = subprocess.Popen(command, stdout=subprocess.DEVNULL)
try:
    with OptionalHidKeyboardWriter() as keyboard:
        wait_for_model(process, first_prompt, keyboard)
        while True:
            generate(rag_docs[prompt_index], keyboard)
            if len(rag_docs) > 1:
                next_index = random.randrange(len(rag_docs) - 1)
                prompt_index = next_index + (next_index >= prompt_index)
            # Later prompts are never shown, and consecutive prompts differ.
finally:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
