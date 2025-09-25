import io, os, time, sounddevice as sd, soundfile as sf, tempfile, subprocess
from dotenv import load_dotenv

from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "gpt-4o-mini")  # budget-friendly

agent = Agent(
    model=OpenAIModel(
        client_args={"api_key": OPENAI_API_KEY},
        model_id=MODEL_ID,
        params={"temperature": 0, "max_tokens": 64}
    ),
    tools=[calculator],
    callback_handler=None
)

# --- Audio capture (push-to-talk) ---
SAMPLE_RATE = 16000
CHANNELS = 1

def record_once(seconds=6):
    print(f"🎙️  Recording for up to {seconds}s… (Ctrl+C to stop early)")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    # trim trailing silence a bit (quick heuristic)
    return audio

def save_wav(data, path):
    sf.write(path, data, SAMPLE_RATE, subtype="PCM_16")

# --- OpenAI STT (Whisper) + TTS (Speech) ---
from openai import OpenAI
oai = OpenAI(api_key=OPENAI_API_KEY)

def stt_transcribe(wav_path):
    with open(wav_path, "rb") as f:
        res = oai.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",  # faster/cheaper than whisper-1
            file=f
        )
    return (res.text or "").strip()

def tts_speak(text, voice="alloy"):
    if not text:
        return
    text = text.replace("\n", " ")
    audio = oai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )
    import tempfile, subprocess
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        # New SDK exposes raw bytes on `.content`
        tmp.write(audio.content if hasattr(audio, "content") else audio)
        tmp.flush()
        subprocess.run(["afplay", tmp.name])

def ask_agent(text: str) -> str:
    res = agent(text)
    return str(res).strip()

if __name__ == "__main__":
    print("🔊 OpenAI Voice PTT ready. Press ENTER to record; type q to quit.")
    while True:
        cmd = input("\n[ENTER=record / q=quit] ")
        if cmd.strip().lower() in {"q", "quit", "exit"}: break

        audio = record_once(seconds=6)
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
            print("Agent:", reply)
            tts_speak(reply, voice="alloy")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)
