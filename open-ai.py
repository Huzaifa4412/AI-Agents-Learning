from agents import Agent, Runner
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")
    result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)


if __name__ == "__main__":
    main()
