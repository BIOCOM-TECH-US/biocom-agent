import os
import time
import tempfile
import subprocess
import threading
import shutil

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from tools_web import web_search

# ---------------- Env & keys ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "gpt-4o-mini")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Put it in your .env at repo root.")

# ---------------- Strands agent ----------------
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator
from tools_kb_langchain import kb_search  # FAISS-backed RAG tool

system = (
    "You are Biocom’s voice agent.\n"
    "• If a question is about Biocom (company/people/products/markets), FIRST call kb_search "
    "  with the user's question. Then answer ONLY using the returned snippets. "
    "  Always keep the [file.md] citation.\n"
    "• Keep answers ≤ 2 sentences unless asked for more.\n"
    "• If the name 'Jan Springer' appears, say it as 'Yaan Springer' and include "
    "  (pronounced 'Yaan') the first time per session.\n"
    "• For math, prefer the calculator tool.\n"
    "• If neither applies, give a brief 1–2 sentence general answer."
)

agent = Agent(
    model=OpenAIModel(
        client_args={"api_key": OPENAI_API_KEY},
        model_id=MODEL_ID,
        params={"temperature": 0, "max_tokens": 512},  # roomy answers
    ),
    tools=[kb_search, calculator],
    system_prompt=system,
    callback_handler=None,
)

# ---------------- OpenAI STT/TTS ----------------
from openai import OpenAI
oai = OpenAI(api_key=OPENAI_API_KEY)

def stt_transcribe(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        res = oai.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            language="en",
            prompt="User speaks English; keep it concise."
        )
    return (res.text or "").strip()

def tts_speak(text: str, voice="alloy"):
    if not text:
        return
    # keep one-line for quick TTS
    text = text.replace("\n", " ")
    resp = oai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(resp.read())   # stream -> bytes
        tmp.flush()
        subprocess.run(["afplay", tmp.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ---------------- Earcons (macOS) ----------------

SYSTEM_SOUNDS = {
    "start": ["Glass.aiff", "Submarine.aiff", "Ping.aiff"],
    "processing": ["Pop.aiff", "Blow.aiff", "Funk.aiff"],
    "ready": ["Hero.aiff", "Tink.aiff", "Bottle.aiff"],
}

def _resolve_sound(name: str) -> str | None:
    base = "/System/Library/Sounds"
    for fname in SYSTEM_SOUNDS.get(name, []):
        path = os.path.join(base, fname)
        if os.path.exists(path):
            return path
    return None

def _fallback_beep(freq=880, dur=0.08, vol=0.2):
    try:
        sr = 22050
        t = np.linspace(0, dur, int(sr * dur), False)
        tone = (vol * np.sin(2 * np.pi * freq * t)).astype("float32")
        sd.play(tone, sr); sd.wait()
    except Exception:
        if shutil.which("osascript"):
            subprocess.run(["osascript", "-e", "beep"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def play_sound(name: str):
    """Play a short earcon without blocking the main thread."""
    def _play():
        path = _resolve_sound(name)
        if path and shutil.which("afplay"):
            try:
                subprocess.run(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass
        # Fallback if afplay or file not available
        _fallback_beep(freq=880 if name == "ready" else (700 if name == "start" else 600))
    threading.Thread(target=_play, daemon=True).start()

# ---------- Continuous "thinking" sound loop (runs until stopped) ----------

_thinking_thread: threading.Thread | None = None
_thinking_stop: threading.Event | None = None

def _thinking_loop():
    """Background loop: quietly tick while thinking."""
    pop_path = _resolve_sound("processing")
    use_afplay = pop_path is not None and shutil.which("afplay")

    # Gentle tempo: ~2 ticks per second
    interval = 0.45

    while _thinking_stop and not _thinking_stop.is_set():
        start = time.time()
        try:
            if use_afplay:
                # -v sets volume (0.0–1.0). Not all macOS versions support -v for afplay;
                # if it fails, we'll just run without it and rely on system volume.
                cmd = ["afplay", pop_path]
                # Try quiet mode; if it errors once, next loop just runs default.
                try:
                    subprocess.run(cmd + ["-v", "0.2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Soft synthesized click: 400 Hz, 25 ms
                sr = 22050
                dur = 0.025
                t = np.linspace(0, dur, int(sr * dur), False)
                # fast attack-decay envelope to make it percussive
                env = np.exp(-40 * t)
                click = (0.18 * np.sin(2 * np.pi * 400 * t) * env).astype("float32")
                sd.play(click, sr); sd.wait()
        except Exception:
            # Ignore and keep looping
            pass

        # keep the cadence stable
        slept = time.time() - start
        if slept < interval:
            time.sleep(interval - slept)

def start_thinking_sound():
    """Start background thinking sound loop."""
    global _thinking_thread, _thinking_stop
    # If already running, don't start another
    if _thinking_thread and _thinking_thread.is_alive():
        return
    _thinking_stop = threading.Event()
    _thinking_thread = threading.Thread(target=_thinking_loop, daemon=True)
    _thinking_thread.start()

def stop_thinking_sound():
    """Stop background thinking sound loop."""
    global _thinking_thread, _thinking_stop
    if _thinking_stop:
        _thinking_stop.set()
    if _thinking_thread and _thinking_thread.is_alive():
        _thinking_thread.join(timeout=0.2)
    _thinking_thread = None
    _thinking_stop = None

# ---------------- Audio capture ----------------
SAMPLE_RATE = 16000
CHANNELS = 1

def record_until_enter(max_seconds=10):
    play_sound("start")  # 🔔 start listening
    print("🎙️  Recording… press ENTER to stop.")
    stop_flag = {"stop": False}

    def stopper():
        input()  # ENTER
        stop_flag["stop"] = True

    t = threading.Thread(target=stopper, daemon=True); t.start()
    buf = []

    def cb(indata, frames, time_info, status):
        if status: pass
        buf.append(indata.copy())
        if stop_flag["stop"]:
            raise sd.CallbackAbort

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", callback=cb):
        try:
            sd.sleep(int(max_seconds * 1000))
        except sd.CallbackAbort:
            pass

    audio = np.concatenate(buf, axis=0) if buf else np.zeros((1, CHANNELS), dtype="float32")
    return audio

def save_wav(data, path):
    sf.write(path, data, SAMPLE_RATE, subtype="PCM_16")

# ---------------- Normalizers ----------------
if not hasattr(__builtins__, "_said_yaan"):
    __builtins__._said_yaan = False

def normalize_names_for_tts(text: str) -> str:
    """Ensure first Jan mention has pronunciation note; later mentions say 'Yaan Springer'."""
    if "Jan Springer" in text:
        if not __builtins__._said_yaan:
            __builtins__._said_yaan = True
            text = text.replace("Jan Springer", "Jan Springer (pronounced 'Yaan')")
        else:
            text = text.replace("Jan Springer", "Yaan Springer")
    # common misspells
    text = text.replace("Atit Karel", "Atit Kharel")
    return text

def normalize_transcript(txt: str) -> str:
    """Fix common STT mis-hears (Biocom/Viacom, names, etc.)."""
    s = txt.strip()
    fixes = {
        "viacom": "Biocom",
        "bio com": "Biocom",
        "biocam": "Biocom",
        "bicom": "Biocom",
        "biocomm": "Biocom",
        "martin mccourtal": "Martin McCorkle",
        "atit karel": "Atit Kharel",
        "yaan": "Jan Springer",
        "jan springer": "Jan Springer",
    }
    low = s.lower()
    for wrong, right in fixes.items():
        if wrong in low:
            # simple case-insensitive token replace
            s = " ".join([right if w.lower() == wrong else w for w in s.split()])
            low = s.lower()
    return s

# ---------------- Ask agent ----------------
def ask_agent(text: str) -> str:
    res = agent(text)
    return str(res).strip()

# ---------------- Main loop ----------------
if __name__ == "__main__":
    print("🔊 OpenAI Voice PTT ready. Press ENTER to record; type q to quit.")
    while True:
        cmd = input("\n[ENTER=record / q=quit] ")
        if cmd.strip().lower() in {"q", "quit", "exit"}:
            break

        audio = record_until_enter(max_seconds=10)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpwav:
            save_wav(audio, tmpwav.name)
            wav_path = tmpwav.name

        # One-shot "processing" chime + start continuous thinking loop
        play_sound("processing")   # short cue
        start_thinking_sound()     # continuous tick while thinking

        try:
            t0 = time.time()
            text = stt_transcribe(wav_path)
            text = normalize_transcript(text)
            print(f"You said: {text}   [STT {time.time()-t0:.2f}s]")

            if not text:
                stop_thinking_sound()
                print("…heard nothing.")
                continue

            t1 = time.time()
            reply = ask_agent(text)
            print(f"[Agent {time.time()-t1:.2f}s]")

            # Stop the thinking loop right before we speak the reply
            stop_thinking_sound()

            reply = normalize_names_for_tts(reply)
            print("Agent:", reply)

            play_sound("ready")  # short cue just before speaking
            t2 = time.time()
            tts_speak(reply, voice="alloy")
            print(f"[TTS {time.time()-t2:.2f}s]")

        except KeyboardInterrupt:
            stop_thinking_sound()
            break
        except Exception as e:
            stop_thinking_sound()
            print("Error:", e)