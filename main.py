import os
from dotenv import load_dotenv

from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

load_dotenv()

model = OpenAIModel(
    client_args={
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    model_id=os.getenv("MODEL_ID", "gpt-4o-mini"),
    params={"temperature": 0, "max_tokens": 64}
)


agent = Agent(model=model, tools=[calculator], callback_handler=None)

if __name__ == "__main__":
    print("Calculator agent ready. Type a math question (q to quit).")
    while True:
        try:
            q = input("\nYou: ")
            if q.strip().lower() in {"q", "quit", "exit"}:
                break
            result = agent(q)
            print("Agent:", str(result))
        except (KeyboardInterrupt, EOFError):
            break
