import os  # To work with os such as files etc.
import asyncio  # To perform Asynchronous tasks in python
from dotenv import load_dotenv  # To Access Secrete Keys
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)  # To Integrate Gemini with open ai Sdk and to create a proper ai agent

from agents.run import RunConfig

# Task 1 Load the env file
load_dotenv()

# Get gemini api key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini Api key not founded")

# Create an External Client -> Here I use Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)

config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)


async def main():
    agent = Agent(
        name="Huzaifa",
        model=model,
        instructions="Your are a poet and your name is Huzaifa",
    )

    results = await Runner.run(
        agent,
        "Tell me a 4 sentence amazing poet in urdu note: The poem use rhyming at the end also mention your name as a takhulus of the poet",
        run_config=config,
    )

    print(results.final_output)


if __name__ == "__main__":
    asyncio.run(main())
