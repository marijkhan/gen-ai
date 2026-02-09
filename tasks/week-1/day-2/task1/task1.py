import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Configuration
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(BASE_DIR, "prompt.txt")
SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on the provided API documentation."


def read_document(file_path: str) -> str:
    """Read the entire document from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def run_completion(client: Groq, model: str, document: str, question: str) -> dict:
    """Send document + question to chat completion."""
    user_content = f"""Here is the API documentation:

{document}

---

Based on the documentation above, please answer the following question:
{question}"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            model=model,
            temperature=0.0
        )
        return {
            "success": True,
            "message": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "error": ""
        }
    except Exception as e:
        return {
            "success": False,
            "message": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": str(e)
        }


def main():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Read the entire document
    document = read_document(PROMPT_FILE)
    print(f"Loaded document from {PROMPT_FILE}")
    print(f"Document length: {len(document)} characters")
    print("-" * 50)

    # Test question - modify as needed
    question = input("Enter your question about the API: ")

    print(f"\nSending to {MODEL}...")
    response = run_completion(client, MODEL, document, question)

    if response["success"]:
        print(f"\n{'=' * 50}")
        print("RESPONSE:")
        print(response["message"])
        print(f"\n{'=' * 50}")
        print(f"Prompt tokens: {response['prompt_tokens']}")
        print(f"Completion tokens: {response['completion_tokens']}")
        print(f"Total tokens: {response['total_tokens']}")
    else:
        print(f"ERROR: {response['error']}")


if __name__ == "__main__":
    main()
