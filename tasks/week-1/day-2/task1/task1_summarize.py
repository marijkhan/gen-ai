import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Configuration
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(BASE_DIR, "prompt.txt")
SUMMARY_FILE = os.path.join(BASE_DIR, "summary.txt")
SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on the provided API documentation."


def read_document(file_path: str) -> str:
    """Read the entire document from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_summary(file_path: str, summary: str) -> None:
    """Save the summary to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(summary)


def load_summary(file_path: str) -> str:
    """Load the summary from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def summarize_document(client: Groq, model: str, document: str) -> dict:
    """First pass: create a condensed summary of the API docs."""
    summary_prompt = """Please create a comprehensive but condensed summary of this API documentation.
Focus on:
1. List of all event types and their purpose
2. Common fields across all events
3. Key structure patterns (e.g., headerValue fields for each event type)
4. Important field types and their possible values

Keep the summary structured and reference-friendly for answering future questions.

Documentation:
"""
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a technical documentation summarizer. Create concise but complete summaries that preserve all important details."},
                {"role": "user", "content": summary_prompt + document}
            ],
            model=model,
            temperature=0.0
        )
        return {
            "success": True,
            "summary": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "error": ""
        }
    except Exception as e:
        return {
            "success": False,
            "summary": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": str(e)
        }


def run_completion(client: Groq, model: str, summary: str, question: str) -> dict:
    """Send summary + question to chat completion."""
    user_content = f"""Here is a summary of the API documentation:

{summary}

---

Based on the documentation summary above, please answer the following question:
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

    # Token tracking
    summarization_tokens = {"prompt": 0, "completion": 0, "total": 0}
    qa_tokens = {"prompt": 0, "completion": 0, "total": 0}

    # Check if summary exists
    if os.path.exists(SUMMARY_FILE):
        print(f"Loading cached summary from {SUMMARY_FILE}")
        summary = load_summary(SUMMARY_FILE)
        print(f"Summary length: {len(summary)} characters")
        print("(No summarization tokens used - loaded from cache)")
    else:
        # Read the full document and generate summary
        print(f"Reading document from {PROMPT_FILE}")
        document = read_document(PROMPT_FILE)
        print(f"Document length: {len(document)} characters")
        print("-" * 50)

        print(f"\nGenerating summary using {MODEL}...")
        summary_result = summarize_document(client, MODEL, document)

        if not summary_result["success"]:
            print(f"ERROR generating summary: {summary_result['error']}")
            return

        summary = summary_result["summary"]
        summarization_tokens = {
            "prompt": summary_result["prompt_tokens"],
            "completion": summary_result["completion_tokens"],
            "total": summary_result["total_tokens"]
        }

        # Save summary to file
        save_summary(SUMMARY_FILE, summary)
        print(f"Summary saved to {SUMMARY_FILE}")
        print(f"Summary length: {len(summary)} characters")
        print(f"\nSummarization tokens:")
        print(f"  Prompt tokens: {summarization_tokens['prompt']}")
        print(f"  Completion tokens: {summarization_tokens['completion']}")
        print(f"  Total tokens: {summarization_tokens['total']}")

    print("-" * 50)

    # Get user's question
    question = input("\nEnter your question about the API: ")

    print(f"\nAnswering using {MODEL}...")
    response = run_completion(client, MODEL, summary, question)

    if response["success"]:
        qa_tokens = {
            "prompt": response["prompt_tokens"],
            "completion": response["completion_tokens"],
            "total": response["total_tokens"]
        }

        print(f"\n{'=' * 50}")
        print("RESPONSE:")
        print(response["message"])
        print(f"\n{'=' * 50}")
        print("\nToken Usage Summary:")
        print(f"  Summarization tokens: {summarization_tokens['total']}")
        print(f"  Q&A tokens: {qa_tokens['total']}")
        print(f"  Total tokens (this session): {summarization_tokens['total'] + qa_tokens['total']}")
    else:
        print(f"ERROR: {response['error']}")


if __name__ == "__main__":
    main()
