import os
import chainlit as cl
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load .env variables
load_dotenv()

# API key from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if API key is provided
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Gemini as OpenAI-compatible client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model wrapper
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-pro", openai_client=external_client  # or "gemini-2.0-flash"
)

# Config setup
config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

# Agent definition
agent = Agent(
    name="Huzaifa",
    model=model,
    instructions="You are a poetic assistant named Huzaifa. Reply in sher-o-shayari unless the user asks something technical or factual.",
)


# Chainlit Entry Point
@cl.on_message
async def main(message: cl.Message):
    # Convert input into a chat-style message list
    messages = [
        {
            "role": "system",
            "content": "You are a poetic assistant named Huzaifa. Greet users warmly and use your name as takhulus. Reply in poetic style unless the user asks something factual or personal.",
        },
        {"role": "user", "content": message.content},
    ]

    # Run the agent with those messages
    result = await Runner.run(agent, messages, run_config=config)

    # Send response to UI
    await cl.Message(content=result.final_output).send()
