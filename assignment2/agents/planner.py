from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, GEMINI_MODEL
from schemas import ExecutionPlan

_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

_structured_llm = _llm.with_structured_output(ExecutionPlan)

PLANNER_SYSTEM_PROMPT = """\
You are a planning agent for a LinkedIn content creation system.

Given a topic, produce an execution plan — a list of steps with dependencies.

Available tools:
- search_web: Search the internet for recent articles, news, or insights on a query. Input: a search query string.
- summarizer: Condense raw text into concise bullet points. Input: raw text from search results.
- content_generator: Write a LinkedIn post from research material. Input: topic + aggregated research.
- content_editor: Polish and refine a draft LinkedIn post. Input: a draft post.
- image_generator: Generate a professional banner image. Input: topic string.

Rules:
1. Always include at least one search_web step to gather recent information.
2. content_generator MUST depend on all search/summarizer steps so it has research to work with.
3. content_editor MUST depend on content_generator.
4. image_generator has NO dependencies — it can run in parallel with everything.
5. You may include a summarizer step between search and generator if the topic is broad.
6. Each step needs a unique step number starting from 1.
7. depends_on lists the step numbers that must complete before this step can start.
8. Keep plans between 4-7 steps.

Produce a plan that maximizes parallelism where possible.
"""


def create_plan(topic: str) -> ExecutionPlan:
    """Use the planner LLM to generate a dynamic execution plan for the given topic."""
    response = _structured_llm.invoke(
        [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "human", "content": f"Create a plan for a LinkedIn post about: {topic}"},
        ]
    )
    # Ensure the topic is set correctly
    response.topic = topic
    return response
