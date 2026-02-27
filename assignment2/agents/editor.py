from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import GOOGLE_API_KEY, GEMINI_MODEL

_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior LinkedIn content editor. "
        "Polish the following draft post for maximum engagement.\n\n"
        "Check and improve:\n"
        "- Clarity: every sentence should be immediately understandable\n"
        "- Tone: professional but approachable, not salesy\n"
        "- Structure: short paragraphs, good flow, strong hook\n"
        "- LinkedIn best practices: appropriate length, clear CTA, relevant hashtags\n"
        "- Remove any jargon that doesn't add value\n\n"
        "Return ONLY the final polished post, no explanations or meta-commentary."
    )),
    ("human", "{draft}"),
])

_chain = _prompt | _llm


def edit_post(draft: str) -> str:
    """Polish and refine a draft LinkedIn post."""
    try:
        response = _chain.invoke({"draft": draft})
        return response.content
    except Exception as e:
        return f"Editing error: {e}"
