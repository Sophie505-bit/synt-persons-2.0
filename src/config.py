import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(temperature: float = 0.7, model: str = None) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    model_name = model or os.environ.get("MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def has_api_key() -> bool:
    load_dotenv()
    return bool(os.environ.get("OPENAI_API_KEY", ""))
