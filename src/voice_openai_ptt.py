import os, tempfile, subprocess, threading, numpy as np
import sounddevice as sd, soundfile as sf
from dotenv import load_dotenv

# ----- Load env -----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "gpt-4o-mini")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Put it in .env")

# ----- Strands: model + tools -----
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
    "• If the name 'Jan Springer' appears, say it as 'Yaan Springer'.\n"
    "• For math, prefer the calculator tool.\n"
    "• If neither applies, give a brief 1–2 sentence general answer."
)

agent = Agent(
    model=OpenAIModel(
        client_args={"api_key": OPENAI_API_KEY},
        model_id=MODEL_ID,
        params={"temperature": 0, "max_tokens": 120}
    ),
    tools=[kb_search, calculator],  # order hints priority
    system_prompt=system,
    callback_handler=None
)

# ----- OpenAI STT/TTS -----
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
    text = text.replace("\n", " ")
    resp = oai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(resp.read())   # resp is a stream of bytes
        tmp.flush()
        subprocess.run(["afplay", tmp.name])


# ----- Audio (press ENTER to stop = lowest latency) -----
SAMPLE_RATE = 16000
CHANNELS = 1

def record_until_enter(max_seconds=10):
    print("🎙️  Recording… press ENTER to stop.")
    stop_flag = {"stop": False}

    def stopper():
        input()  # ENTER
        stop_flag["stop"] = True

    t = threading.Thread(target=stopper, daemon=True); t.start()
    buf = []

    def cb(indata, frames, time, status):
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

# ---- Agent ask + pronunciation normalization ----
if not hasattr(__builtins__, "_said_yaan"):
    __builtins__._said_yaan = False

def normalize_names_for_tts(text: str) -> str:
    if "Jan Springer" in text:
        if not __builtins__._said_yaan:
            __builtins__._said_yaan = True
            text = text.replace("Jan Springer", "Jan Springer (pronounced 'Yaan')")
        else:
            text = text.replace("Jan Springer", "Yaan Springer")
    # common mis-spell
    text = text.replace("Atit Karel", "Atit Kharel")
    return text

def ask_agent(text: str) -> str:
    res = agent(text)
    return str(res).strip()

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

        try:
            text = stt_transcribe(wav_path)
            print("You said:", text)
            if not text:
                print("…heard nothing.")
                continue

            reply = ask_agent(text)
            reply = normalize_names_for_tts(reply)
            print("Agent:", reply)
            tts_speak(reply, voice="alloy")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)