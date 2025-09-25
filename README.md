# Biocom Voice Agent (Prototype)

A minimal voice-first agent built with [Strands Agents](https://strandsagents.com/) and OpenAI.  
You can speak to it (push-to-talk) and it answers back using OpenAI STT + TTS.  
Right now it can solve math via the **calculator tool**. Later, we’ll add headset controls.

---

## Setup (Mac)

1. Clone repo (or open with GitHub Desktop).
2. Create and activate venv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

3. Install deps:
pip install "strands-agents[openai]" strands-agents-tools python-dotenv sounddevice soundfile

4. Create .env with your key
