import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, set_tracing_disabled

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(True)

# Check if API key exists
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Setup model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
)

# Agents
Billing_Agent = Agent(
    name="Billing Agent",
    instructions="You handle billing-related issues. Help with payments, subscriptions, invoices and refunds.",
    model=model
)

Technical_Agent = Agent(
    name="Technical Agent", 
    instructions="You handle technical issues. Help with troubleshooting, bugs, setup problems and technical errors.",
    model=model
)

Account_Agent = Agent(
    name="Account Agent",
    instructions="You handle account-related issues. Help with login problems, account settings, password resets and security.",
    model=model
)

Triage_Agent = Agent(
    name="Triage Agent",
    instructions="""You determine whether the user needs billing, technical, or account support.
    
    Based on user's issue:
    - For billing, payments, subscriptions → transfer to Billing Agent
    - For technical problems, bugs, setup → transfer to Technical Agent  
    - For account, login, security issues → transfer to Account Agent
    
    Always explain why you're transferring them to that specific agent.""",
    handoffs=[Billing_Agent, Technical_Agent, Account_Agent],
    model=model
)

async def main():
    print("🤖 Multi-Agent Customer Support System")
    print("=" * 30)

    while True:
        user_input = input("Enter your message (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting support agent.")
            break

        print("⏳ Processing...")

        response = await Runner.run(Triage_Agent, input=user_input)

        try:
            print(f"\nAgent Response:\n{response.final_output.strip()}\n")
        except AttributeError:
            print("\nAgent Response:\nSorry, I couldn't understand your request.\n")

if __name__ == "__main__":
    asyncio.run(main())