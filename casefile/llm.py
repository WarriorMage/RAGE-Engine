"""Small LLM wrapper that returns JSON matching a schema.

Works with two providers through the OpenAI Python SDK:
  gemini (default) : Google Gemini via its OpenAI-compatible endpoint (free tier available)
  openai           : OpenAI

Settings (.env):
  LLM_PROVIDER=gemini            # or "openai"
  GEMINI_API_KEY=...             # from https://aistudio.google.com/apikey
  GEMINI_MODEL=gemini-flash-latest  # optional
  OPENAI_API_KEY=...             # only if LLM_PROVIDER=openai
  OPENAI_MODEL=gpt-4.1           # optional
"""

import json
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
PROVIDERS = {
    #           key env var        model env var    default model         base url
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-flash-latest", GEMINI_BASE_URL),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4.1", None),
}

# Backup models, tried in order when the main one is busy, rate-limited or unavailable.
FALLBACK_MODELS = {
    "gemini": ["gemini-flash-latest", "gemini-3.5-flash", "gemini-flash-lite-latest", "gemini-3.5-flash-lite"],
    "openai": ["gpt-4.1-mini"],
}
BUSY_WAIT = 30  # seconds to wait when every model is busy


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
    # Only 1 quick SDK retry: on failure we switch to a backup model instead (see chat_json).
    return OpenAI(api_key=key, base_url=base_url, max_retries=1)


def _call(name: str, model: str, system: str, user: str, schema: dict, schema_name: str):
    return _client(name).chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
    )


def models_to_try() -> list[str]:
    """The configured model first, then backups (free tiers limit each model separately)."""
    models: list[str] = []
    for m in [model_name(), *FALLBACK_MODELS[provider()]]:
        if m not in models:
            models.append(m)
    return models


def chat_json(system: str, user: str, schema: dict, schema_name: str) -> dict:
    """Ask the model and get back a dict matching `schema` (structured output).

    If a model is busy (503), rate-limited (429) or unavailable (404), the next backup model
    is tried. If all are busy, wait BUSY_WAIT seconds and try the whole list once more.
    """
    name = provider()
    models = models_to_try()
    problems: list[str] = []
    response = None
    for attempt in range(2):
        for model in models:
            try:
                response = _call(name, model, system, user, schema, schema_name)
                break
            except AuthenticationError:
                raise RuntimeError(f"{name} rejected the API key. Check {PROVIDERS[name][0]} in your .env file.") from None
            except BadRequestError as e:
                details = str(e.body or e)
                if "api key" in details.lower():   # Gemini reports a bad key as 400 INVALID_ARGUMENT
                    raise RuntimeError(f"{name} rejected the API key. Check {PROVIDERS[name][0]} in your .env file.") from None
                raise RuntimeError(f"{name} rejected the request: {details[:300]}") from None
            except RateLimitError as e:
                text = str(e).lower()
                if "credit" in text or "billing" in text:
                    raise RuntimeError(f"{name} account has no credits. Add billing, or switch LLM_PROVIDER.") from None
                problems.append(f"{model}: rate limited")
            except NotFoundError:
                problems.append(f"{model}: not available")
            except (InternalServerError, APIConnectionError, APITimeoutError):
                problems.append(f"{model}: busy")
            print(f"  ({problems[-1]}, trying another model...)")
        if response is not None:
            break
        if attempt == 0:
            print(f"  (all models busy, waiting {BUSY_WAIT}s and trying again...)")
            time.sleep(BUSY_WAIT)
    if response is None:
        raise RuntimeError(f"{name} is busy or rate-limiting every model ({'; '.join(problems[-len(models):])}). "
                           "Wait a minute and try again.")

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
