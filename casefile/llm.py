"""Small LLM wrapper that returns JSON matching a schema.

Works with two providers through the OpenAI Python SDK:
  gemini (default) : Google Gemini via its OpenAI-compatible endpoint (free tier available)
  openai           : OpenAI

Settings (.env):
  LLM_PROVIDER=gemini            # or "openai"
  GEMINI_API_KEY=...             # from https://aistudio.google.com/apikey
  GEMINI_MODEL=gemini-2.5-flash  # optional
  OPENAI_API_KEY=...             # only if LLM_PROVIDER=openai
  OPENAI_MODEL=gpt-4.1           # optional
"""

import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AuthenticationError, BadRequestError, NotFoundError, OpenAI, RateLimitError

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
PROVIDERS = {
    #           key env var        model env var    default model         base url
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash", GEMINI_BASE_URL),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4.1", None),
}


def _clean(value: str | None) -> str:
    return (value or "").strip().strip('"').strip("'")


def provider() -> str:
    name = _clean(os.environ.get("LLM_PROVIDER")).lower() or "gemini"
    if name not in PROVIDERS:
        raise RuntimeError(f"LLM_PROVIDER must be one of: {', '.join(PROVIDERS)} (got '{name}')")
    return name


def model_name() -> str:
    _, model_var, default, _ = PROVIDERS[provider()]
    return _clean(os.environ.get(model_var)) or default


@lru_cache(maxsize=2)
def _client(name: str) -> OpenAI:
    key_var, _, _, base_url = PROVIDERS[name]
    key = _clean(os.environ.get(key_var))
    if not key:
        hint = " Get a free key at https://aistudio.google.com/apikey" if name == "gemini" else ""
        raise RuntimeError(f"{key_var} is not set (LLM_PROVIDER={name}). Add it to your .env file.{hint}")
    # max_retries makes the SDK wait and retry automatically on rate limits (429).
    return OpenAI(api_key=key, base_url=base_url, max_retries=5)


def chat_json(system: str, user: str, schema: dict, schema_name: str) -> dict:
    """Ask the model and get back a dict matching `schema` (structured output)."""
    name = provider()
    try:
        response = _client(name).chat.completions.create(
            model=model_name(),
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        )
    except AuthenticationError:
        raise RuntimeError(f"{name} rejected the API key. Check {PROVIDERS[name][0]} in your .env file.") from None
    except BadRequestError as e:
        details = str(e.body or e)
        if "api key" in details.lower():   # Gemini reports a bad key as 400 INVALID_ARGUMENT
            raise RuntimeError(f"{name} rejected the API key. Check {PROVIDERS[name][0]} in your .env file.") from None
        raise RuntimeError(f"{name} rejected the request: {details[:300]}") from None
    except NotFoundError:
        raise RuntimeError(
            f"Model '{model_name()}' was not found on {name}. Run: python scripts/check_llm.py "
            f"to list available models, then set {PROVIDERS[name][1]} in .env."
        ) from None
    except RateLimitError as e:
        text = str(e).lower()
        if "credit" in text or "billing" in text:
            raise RuntimeError(f"{name} account has no credits. Add billing, or switch LLM_PROVIDER.") from None
        raise RuntimeError(
            f"{name} rate limit reached even after retries. Wait a minute and try again "
            "(free tiers allow only a few requests per minute)."
        ) from None

    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"The model refused: {message.refusal}")
    content = (message.content or "").strip()
    if content.startswith("```"):  # some models wrap JSON in a code fence
        content = content.strip("`").removeprefix("json").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError(f"The model did not return valid JSON. Got: {content[:200]}") from None
