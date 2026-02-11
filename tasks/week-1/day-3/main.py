import fitz  
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

text = ""
for page in fitz.open("D:\\Work\\gen-ai\\tasks\\week-1\\day-3\\input\\constitution_pak.pdf"):
    text += page.get_text()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {"role": "system", "content": f"Answer based on this document:\n{text}"},
        {"role": "user", "content": "What is the minimum age to become President of Pakistan?"}
    ]
)
print(response)