from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import GOOGLE_API_KEY, GEMINI_MODEL

_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert LinkedIn content creator. "
        "Write an engaging LinkedIn post based on the research provided.\n\n"
        "Format:\n"
        "1. Start with a compelling headline/hook (1 line, attention-grabbing)\n"
        "2. Body: 150-250 words, conversational yet professional tone\n"
        "3. Use short paragraphs (2-3 sentences max)\n"
        "4. Include a call-to-action or thought-provoking question at the end\n"
        "5. End with 3-5 relevant hashtags on a new line\n\n"
        "Do NOT use markdown formatting. Write plain text suitable for LinkedIn."
    )),
    ("human", "Topic: {topic}\n\nResearch:\n{research}"),
])

_chain = _prompt | _llm


def generate_post(topic: str, research: str) -> str:
    """Generate a LinkedIn post from the topic and aggregated research."""
    try:
        response = _chain.invoke({"topic": topic, "research": research})
        return response.content
    except Exception as e:
        return f"Generation error: {e}"
