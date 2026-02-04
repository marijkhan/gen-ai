import os
import csv
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Configuration
MODELS = [
    # "compound-beta-mini",
    "meta-llama/llama-4-scout-17b-16e-instruct"
    # "openai/gpt-oss-120b"
]
TEMPERATURES = [0, 0.7, 1]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(BASE_DIR, "input", "prompts.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
SYSTEM_PROMPT = "You are a helpful assistant that gives accurate answers in less than 25 words."


def load_prompts(filepath: str) -> list[str]:
    """Load prompts from a text file (one prompt per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def run_prompt(client: Groq, model: str, prompt: str, temperature: float) -> dict:
    """Run a single prompt and return response with metadata."""
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model=model,
            temperature=temperature
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


def write_results_to_csv(results: list[dict], filepath: str):
    """Write results to a CSV file."""
    fieldnames = [
        "prompt", "model", "temperature", "message",
        "prompt_tokens", "completion_tokens", "total_tokens", "error"
    ]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    # Initialize client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Load prompts from file
    prompts = load_prompts(PROMPTS_FILE)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")
    print(f"Models: {MODELS}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Total API calls: {len(prompts) * len(MODELS) * len(TEMPERATURES)}")
    print("-" * 50)

    results = []

    # For each prompt -> for each model -> for each temperature
    for prompt_idx, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {prompt_idx}/{len(prompts)}: {prompt[:50]}...")

        for model in MODELS:
            print(f"  Model: {model}")

            for temp in TEMPERATURES:
                print(f"    Temperature: {temp}", end=" -> ")
                for _run in range(3):
                    print(f"    Run: {_run + 1}", end=" -> ")
                    response = run_prompt(client, model, prompt, temp)

                    results.append({
                        "prompt": prompt,
                        "model": model,
                        "temperature": temp,
                        "message": response["message"],
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                        "total_tokens": response["total_tokens"],
                        "error": response["error"]
                    })

                    if response["success"]:
                        print(f"OK ({response['total_tokens']} tokens)")
                    else:
                        print(f"ERROR: {response['error'][:50]}")

    # Write results to CSV
    write_results_to_csv(results, OUTPUT_FILE)
    print(f"\n{'=' * 50}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Total records: {len(results)}")


if __name__ == "__main__":
    main()
