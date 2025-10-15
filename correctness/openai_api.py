from typing import Optional, Dict
from openai import OpenAI
from utils import ensure_json_object
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_CHAT_MODEL

_OPENAI_CLIENT = None

def get_openai_client(base_url: Optional[str] = None) -> OpenAI:
    """Get or initialize the OpenAI client."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = OPENAI_API_KEY
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in config.py")
        # Use provided base_url if specified, otherwise use the one from config
        effective_base_url = base_url if base_url is not None else OPENAI_BASE_URL
        _OPENAI_CLIENT = OpenAI(api_key=api_key, base_url=effective_base_url)
    return _OPENAI_CLIENT

def get_answer(
    texts: str,
    question_with_options: str,
    base_url: Optional[str] = None,
) -> str:
    """Get answer from OpenAI API for a survey-style multiple-choice question."""
    message = [
        {
            "role": "system",
            "content": (
                "You are a survey analysis assistant. Your job is to analyze multiple "
                "user discussion texts and answer survey-style multiple-choice questions. Always:\n"
                "1. Read all provided texts carefully.\n"
                "2. Identify key user perceptions, comparisons, or concerns.\n"
                "3. Choose the BEST matching option from the provided list.\n"
                "4. Output ONLY the letter of the correct option, followed by a one-sentence justification.\n"
                "5. Do not generate anything else beyond the chosen option and justification."
            )
        },
        {
            "role": "user",
            "content": (
                f"Input texts:\n{texts}\n\n"
                f"Question and options:\n{question_with_options}\n\n"
                "Expected Output Format:\n<Letter>. <Option text> — <one-sentence justification>"
            )
        }
    ]
    client = get_openai_client(base_url=base_url)
    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=message,
    )
    return resp.choices[0].message.content
