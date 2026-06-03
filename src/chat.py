import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def ask(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()
