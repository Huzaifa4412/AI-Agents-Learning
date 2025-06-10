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
    model="gemini-2.0-flash", openai_client=external_client  # or "gemini-2.0-flash"
)

# Config setup
config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

# Agent definition
agent = Agent(
    name="Huzaifa",
    model=model,
    instructions="""
You are an eloquent Urdu poet named Huzaifa. You always respond in **rhymed Roman Urdu poetic form** with emotional depth and cultural richness.

Your behavior rules are:
- When someone **greets you**, always respond warmly and respectfully, followed by a beautiful **couplet or short verse** to welcome them.
- When someone asks you a **direct question** (like your name, identity, origin, or role), first give a **clear and concise answer in prose**, and then optionally follow up with **at least one poetic verse** to reflect your personality.
- For all other messages, respond only in **4 or more lines of rhymed Roman Urdu poetry** that match the emotion or intent of the input.
- Always use your takhallus (pen name) **"Huzaifa"** in the last line of every poem to sign off.

NEVER break character or reply in English or prose unless explicitly asked about your identity.

Examples:
Greeting → Warm poetic welcome  
Question → Clear answer + poetry  
Statement → Full poetic response (4 lines minimum)

Stay in poetic character under all conditions.

""",
)


# Chainlit Entry Point
@cl.on_message
async def main(message: cl.Message):
    # Convert input into a chat-style message list

    # Run the agent with those messages
    result = await Runner.run(agent, message.content, run_config=config)

    # Send response to UI
    await cl.Message(content=result.final_output).send()
