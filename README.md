# Gen-AI: LLM Experimentation Framework

A hands-on exploration of LLM behavior and tokenization economics.

## Project Structure

```
gen-ai/
├── tasks/
│   ├── .env                          # API keys (GROQ_API_KEY)
│   └── week-1/
│       └── day-1/
│           ├── task1/                # LLM Temperature & Model Comparison
│           │   ├── task1.py
│           │   ├── Observations
│           │   ├── input/
│           │   │   └── prompts.txt
│           │   └── output/
│           │       └── results_*.csv
│           │
│           └── task2/                # Tokenizer Comparison
│               ├── task2.py
│               ├── Observations
│               ├── input/
│               │   ├── strings.txt
│               │   └── strings2.txt
│               └── output/
│                   └── results*.csv
├── .gitignore
└── README.md
```

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate virtual environment

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install groq python-dotenv tiktoken
```

### 4. Configure API key

Create `tasks/.env` with your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

Get a key from https://console.groq.com/keys

## Tasks

### Task 1: LLM Temperature & Model Comparison

Runs prompts against LLM models at different temperature settings and records responses with token usage.

**Configuration** (top of `task1.py`):
- `MODELS` - list of Groq model IDs to test
- `TEMPERATURES` - list of temperature values (default: 0, 0.7, 1)
- `SYSTEM_PROMPT` - system message sent with every request

**Input:** One prompt per line in `input/prompts.txt`

**Run:**
```bash
python tasks/week-1/day-1/task1/task1.py
```

**Output:** `output/results_YYYYMMDD_HHMMSS.csv` with columns:
| Column | Description |
|--------|-------------|
| prompt | Input prompt |
| model | Model used |
| temperature | Temperature setting |
| message | Model response |
| prompt_tokens | Input tokens consumed |
| completion_tokens | Output tokens generated |
| total_tokens | Total tokens used |
| error | Error message (if any) |

**Execution order:** For each prompt -> for each model -> for each temperature

### Task 2: Tokenizer Comparison

Tokenizes strings using different tokenizers and compares token counts across categories.

**Configuration** (top of `task2.py`):
- `TOKENIZERS` - dictionary of tokenizer names and encoding IDs
- `STRINGS_FILE` - path to input file (change to test different string sets)

**Input:** One string per line in `input/strings.txt` (or `strings2.txt`, etc.)

**Run:**
```bash
python tasks/week-1/day-1/task2/task2.py
```

**Output:** Filename derived from input (e.g. `strings.txt` -> `results.csv`, `strings2.txt` -> `results2.csv`) with columns:
| Column | Description |
|--------|-------------|
| string | Input string |
| category | Auto-detected category (Code, URL, JSON, Emoji, Urdu/Deutsch Mix) |
| char_count | Character count |
| tokenizer | Tokenizer used |
| token_count | Number of tokens |
| tokens | Raw token IDs |

**String categories tested:**
- Code (Python, SQL, JS, HTML)
- Urdu/Deutsch mixed-script text
- Emojis (simple, flags, compound ZWJ sequences)
- Long URLs with query parameters
- JSON (nested objects, arrays, unicode values)

## Observations

Each task has an `Observations` file documenting findings:

- **Task 1:** LLM behavior cheatsheet covering temperature, sampling, determinism, and why identical inputs can produce different outputs
- **Task 2:** Tokenization report with 5 key findings:
  1. Mixed-script text costs 17-33% more tokens than sum of parts
  2. Flag emojis cost 4-6x more tokens than simple emojis
  3. JSON costs 1.8-2.7x more tokens than equivalent plain text
  4. o200k only outperforms cl100k for non-Latin scripts
  5. A single URL can cost more tokens than a full English paragraph

## Adding New Experiments

To add new input sets:
1. Create a new file in the task's `input/` folder (e.g. `prompts2.txt`, `strings3.txt`)
2. Update the file path in the configuration section of the script
3. Run the script — output files are generated automatically
