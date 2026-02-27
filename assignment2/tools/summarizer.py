from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, GEMINI_MODEL

_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)


def summarize(text: str) -> str:
    """Condense raw search results into concise bullet points."""
    try:
        response = _llm.invoke(
            f"Summarize the following content into concise bullet points "
            f"capturing the key insights, trends, and facts. "
            f"Keep only the most relevant information.\n\n{text}"
        )
        return response.content
    except Exception as e:
        return f"Summarization error: {e}"
