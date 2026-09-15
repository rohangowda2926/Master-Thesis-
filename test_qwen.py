import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model="qwen/qwen3-8b",
    messages=[
        {
            "role": "user",
            "content": "Calculate the percentage increase from 120 to 150. Show the calculation and final answer."
        }
    ],
    temperature=0
)

print(response.choices[0].message.content)