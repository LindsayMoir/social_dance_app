import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that returns business addresses in a precise format."},
        {"role": "user", "content": "What is the full address (including postal code and city) of North Point Brewing Co in British Columbia, Canada?"}
    ]
)

print("\n🔍 GPT-4o result:\n")
print(response.choices[0].message.content)
