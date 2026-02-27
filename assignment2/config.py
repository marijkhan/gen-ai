import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-google-api-key-here":
    raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to assignment2/.env")
if not TAVILY_API_KEY or TAVILY_API_KEY == "your-tavily-api-key-here":
    raise EnvironmentError("TAVILY_API_KEY is not set. Add it to assignment2/.env")

GEMINI_MODEL = "gemini-2.5-flash"
