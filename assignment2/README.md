# LinkedIn Content Curation Agentic System

An AI-powered agentic system that generates LinkedIn posts on any given topic. It discovers recent content via web search, synthesizes it into a professional post, generates a banner image, and exposes the full flow through a planner-driven agent architecture.

## Architecture

### Agent Roles

| Agent | Role |
|-------|------|
| **Planner Agent** | Analyzes the topic and produces a dynamic execution plan (a dependency graph of steps). Decides which tools to use, how many searches to run, and whether summarization is needed. |
| **Generator Agent** | Takes the topic and aggregated research from upstream steps and writes a LinkedIn post with a hook, body, CTA, and hashtags. |
| **Editor Agent** | Polishes the draft post for clarity, tone, structure, and LinkedIn best practices. Returns the final version. |

### Tools

| Tool | Description |
|------|-------------|
| `search_web` | Searches the web via Tavily API for recent articles and insights |
| `summarizer` | Uses Gemini to condense raw search results into bullet points |
| `image_generator` | Generates a professional LinkedIn banner image via Gemini Imagen |

### Why the Planner?

This system uses a **dynamic planner** rather than a hardcoded pipeline. The planner LLM decides at runtime:
- How many searches to perform (varies by topic breadth)
- Whether to include a summarization step
- The dependency structure between steps

This means the execution graph is **generated, not predefined**. For a narrow topic, the planner might use a single search. For a broad topic, it might issue multiple searches with a summarizer. The executor then runs independent steps in parallel waves based on the dependency graph.

This is fundamentally different from a pipeline/DAG where the flow is fixed at development time.

### Execution Flow

```
Topic → Planner Agent → Execution Plan (dependency graph)
                              ↓
                    Parallel Executor
                    ├── Wave 1: search_web (×N) + image_generator (parallel)
                    ├── Wave 2: summarizer (optional)
                    ├── Wave 3: content_generator
                    └── Wave 4: content_editor
                              ↓
                    Final Post + Image
```

## Setup

1. **Install dependencies**:
   ```bash
   cd assignment2
   pip install -r requirements.txt
   ```

2. **Configure API keys** — edit `.env`:
   ```
   GOOGLE_API_KEY=your-google-api-key
   TAVILY_API_KEY=your-tavily-api-key
   ```

3. **Start the FastAPI backend**:
   ```bash
   cd assignment2
   uvicorn api:app --reload --port 8000
   ```

4. **Start the Streamlit frontend** (in a separate terminal):
   ```bash
   cd assignment2
   streamlit run app.py
   ```

## API Endpoints

### `POST /plan`
Returns the execution plan without executing it.
```bash
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{"topic": "GenAI agents for backend engineers"}'
```

### `POST /execute`
Generates the plan, executes it, and returns the full result including the post and image.
```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"topic": "GenAI agents for backend engineers"}'
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Search API fails | Step marked as error; generator proceeds with partial results |
| Image generation fails | `image_base64` is null; UI shows placeholder message |
| Invalid plan from LLM | Structured output validation error → 500 response |
| Dependency deadlock | Executor breaks loop, returns partial results |
| Missing API keys | `config.py` raises `EnvironmentError` on import |

## Known Failure Case

**Image generation quota/rate limits**: The Gemini Imagen API has strict rate limits on the free tier. When generating images for multiple topics in quick succession, the API may return a 429 (rate limit) error. The system handles this gracefully — the post is still generated and displayed, but the image section shows a "No image was generated" message. The error details are visible in the debug panel's step results.
