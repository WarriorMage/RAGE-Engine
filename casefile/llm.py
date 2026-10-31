"""Small wrapper around the OpenAI API that returns JSON matching a schema."""

import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError

load_dotenv()

DEFAULT_MODEL = "gpt-4.1"


def model_name() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your environment or .env file.")
    return OpenAI(api_key=key, max_retries=3)


def chat_json(system: str, user: str, schema: dict, schema_name: str) -> dict:
    """Ask the model and get back a dict guaranteed to match `schema` (strict structured output)."""
    try:
        response = _client().chat.completions.create(
            model=model_name(),
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        )
    except AuthenticationError:
        raise RuntimeError("OpenAI rejected OPENAI_API_KEY. Check the key in your .env file.") from None
    except RateLimitError as e:
        if "credit" in str(e).lower() or "quota" in str(e).lower():
            raise RuntimeError(
                "OpenAI account has no credits. Add credit at "
                "https://platform.openai.com/settings/organization/billing/ and try again."
            ) from None
        raise
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"The model refused: {message.refusal}")
    return json.loads(message.content)
