from agents import ModelSettings , RunConfig, OpenAIChatCompletionsModel ,  set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
set_tracing_disabled(True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_API_KEY,base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
)

def model_config():
    config = RunConfig(
        model=model,
        model_provider=client,
        tracing_disabled=True,
        model_settings=ModelSettings(
            temperature=0.7,
            top_p=0.5,
            tool_choice="auto",
            max_tokens=1024
        ),
    )
    return config
