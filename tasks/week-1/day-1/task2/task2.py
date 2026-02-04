import os
import csv

import tiktoken

# Configuration
TOKENIZERS = {
    "cl100k_base (GPT-4)": "cl100k_base",
    "o200k_base (GPT-4o)": "o200k_base",
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRINGS_FILE = os.path.join(BASE_DIR, "input", "strings2.txt")


def get_output_file(input_file: str) -> str:
    """Derive output filename from input filename. e.g. strings1.txt -> results1.csv"""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    suffix = base_name.replace("strings", "")
    return os.path.join(BASE_DIR, "output", f"results{suffix}.csv")


def load_strings(filepath: str) -> list[str]:
    """Load strings from a text file (one string per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        strings = [line.strip() for line in f if line.strip()]
    return strings


def categorize_string(s: str) -> str:
    """Assign a category to a string based on its content."""
    if s.startswith("http"):
        return "URL"
    if s.startswith("{") or s.startswith("["):
        return "JSON"
    if s.startswith("<") or "def " in s or "import " in s or "SELECT " in s or "const " in s or "for " in s:
        return "Code"
    if any("\u0600" <= c <= "\u06FF" for c in s):
        return "Urdu/Deutsch Mix"
    if any(ord(c) > 0x1F000 for c in s):
        return "Emoji"
    return "Other"


def tokenize_string(encoder: tiktoken.Encoding, text: str) -> dict:
    """Tokenize a string and return token count and tokens."""
    tokens = encoder.encode(text)
    return {
        "token_count": len(tokens),
        "tokens": tokens,
    }


def write_results_to_csv(results: list[dict], filepath: str):
    """Write results to a CSV file."""
    fieldnames = [
        "string", "category", "char_count",
        "tokenizer", "token_count", "tokens",
    ]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    # Resolve output file from input file name
    output_file = get_output_file(STRINGS_FILE)

    # Load strings from file
    strings = load_strings(STRINGS_FILE)
    print(f"Loaded {len(strings)} strings from {STRINGS_FILE}")
    print(f"Tokenizers: {list(TOKENIZERS.keys())}")
    print(f"Total tokenizations: {len(strings) * len(TOKENIZERS)}")
    print("-" * 50)

    # Initialize tokenizers
    encoders = {}
    for name, encoding_name in TOKENIZERS.items():
        encoders[name] = tiktoken.get_encoding(encoding_name)
        print(f"Loaded tokenizer: {name}")

    results = []

    # For each string -> for each tokenizer
    for str_idx, text in enumerate(strings, 1):
        category = categorize_string(text)
        char_count = len(text)
        print(f"\nString {str_idx}/{len(strings)} [{category}]: {text[:50]}...")

        for tokenizer_name, encoder in encoders.items():
            result = tokenize_string(encoder, text)

            results.append({
                "string": text,
                "category": category,
                "char_count": char_count,
                "tokenizer": tokenizer_name,
                "token_count": result["token_count"],
                "tokens": result["tokens"],
            })

            print(f"  {tokenizer_name}: {result['token_count']} tokens")

    # Write results to CSV
    write_results_to_csv(results, output_file)
    print(f"\n{'=' * 50}")
    print(f"Results saved to: {output_file}")
    print(f"Total records: {len(results)}")


if __name__ == "__main__":
    main()
