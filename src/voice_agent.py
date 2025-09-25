import os, sys, queue, json, subprocess, threading
from pathlib import Path
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# === Agent bits from your working calculator setup ===
from dotenv import load_dotenv
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

load_dotenv()
model = OpenAIModel(
    client_args={"api_key": os.getenv("OPENAI_API_KEY")},
    model_id=os.getenv("MODEL_ID", "gpt-4o-mini"),
    params={"temperature": 0, "max_tokens": 96}  # keep cost tiny
)
agent = Agent(model=model, tools=[calculator], callback_handler=None)

# === Vosk model path ===
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "vosk-model-small-en-us-0.15"
if not MODEL_DIR.exists():
    print(f"[ERROR] Vosk model not found at {MODEL_DIR}")
    print("Download 'vosk-model-small-en-us-0.15' and unzip there, then rerun.")
    sys.exit(1)

vosk_model = Model(str(MODEL_DIR))

# === Audio capture params ===
SAMPLE_RATE = 16000  # Vosk prefers 16k mono
BLOCK_SIZE = 8000    # ~0.5s per block
CHANNELS = 1

def transcribe_once(timeout_s: float = 8.0) -> str:
    """
    Records from mic until silence or timeout, returns a single utterance.
    Simple heuristic: stop when we get 1.2s of 'no new words' or we hit timeout.
    """
    q = queue.Queue()
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.SetWords(False)

    def audio_cb(indata, frames, time, status):
        if status:  # underrun/overrun messages
            pass
        q.put(bytes(indata))

    stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                               channels=CHANNELS, dtype='int16', callback=audio_cb)
    text, got_words = "", False
    idle_blocks, max_idle = 0, 3  # ~1.5s (3 * 0.5s)
    timer = [0.0]

    def tick():
        # coarse timeout timer
        import time
        start = time.time()
        while timer[0] == 0.0:
            time.sleep(0.1)
        while time.time() - timer[0] < timeout_s:
            time.sleep(0.1)
        # force stop on timeout
        q.put(None)

    try:
        stream.start()
        import time
        timer[0] = time.time()
        threading.Thread(target=tick, daemon=True).start()

        while True:
            data = q.get()
            if data is None:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                chunk = (res.get("text") or "").strip()
                if chunk:
                    text = (text + " " + chunk).strip() if text else chunk
                    got_words = True
                    idle_blocks = 0
            else:
                # partials indicate speech activity; we won't print them
                idle_blocks += 1
                if got_words and idle_blocks >= max_idle:
                    break
    finally:
        stream.stop()
        stream.close()

    return (text or "").strip()

def speak_mac(text: str):
    """Use macOS 'say' for quick TTS."""
    if not text:
        return
    # Trim to something short for demo
    text = text.replace("\n", " ")
    subprocess.run(["say", text])

def ask_agent(text: str) -> str:
    if not text.strip():
        return ""
    result = agent(text)
    # Prefer short, clear answers for voice
    return str(result).strip()

if __name__ == "__main__":
    print("🎤 Voice agent ready.")
    print("Press ENTER to start talking, ENTER again to ask, or type 'q' to quit.")
    while True:
        cmd = input("\n[ENTER=record / q=quit] ")
        if cmd.strip().lower() in {"q", "quit", "exit"}:
            break
        print("Listening… speak now.")
        utterance = transcribe_once(timeout_s=8.0)
        print(f"You said: {utterance!r}")
        if not utterance:
            print("…heard nothing. Try again.")
            continue
        reply = ask_agent(utterance)
        print("Agent:", reply or "[no reply]")
        speak_mac(reply)