"""Turn text into embeddings (lists of numbers) with Voyage AI.

Documents and questions are embedded with different input_type values, which Voyage
recommends for retrieval. Each chunk is embedded together with a short context line
(title, type, date) so a chunk like "01:43:12 | DOOR-E2 | HB-9042" still carries
the meaning "door access log" in its vector.
"""

import os
import time
from functools import lru_cache

import voyageai
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "voyage-4"
EMBEDDING_DIMS = 1024
BATCH_SIZE = 32                  # max chunks per API call
MAX_TOKENS_PER_REQUEST = 3000    # keeps each call well inside the free-tier per-minute limit


def model_name() -> str:
    return os.environ.get("VOYAGE_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def _client() -> voyageai.Client:
    key = os.environ.get("VOYAGE_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("VOYAGE_API_KEY is not set. Put it in your environment or .env file.")
    return voyageai.Client(api_key=key, max_retries=0)  # we handle retries ourselves


def chunk_embedding_text(chunk: dict) -> str:
    """The text we actually embed for a chunk: a context line plus the chunk text."""
    date = chunk["date"].strftime("%Y-%m-%d") if hasattr(chunk.get("date"), "strftime") else chunk.get("date", "")
    context = f"{chunk['title']} ({chunk['doc_type'].replace('_', ' ')}, {chunk['source_id']}, {date})"
    return f"{context}\n{chunk['text']}"


RATE_LIMIT_WAIT = 60   # seconds; Voyage limits are per minute
RATE_LIMIT_TRIES = 6


def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    for attempt in range(1, RATE_LIMIT_TRIES + 1):
        try:
            return _client().embed(texts, model=model_name(), input_type=input_type).embeddings
        except voyageai.error.RateLimitError:
            if attempt == RATE_LIMIT_TRIES:
                break
            print(f"  Voyage rate limit hit (try {attempt}/{RATE_LIMIT_TRIES}), waiting {RATE_LIMIT_WAIT}s... "
                  "(tip: adding a payment method in the Voyage dashboard raises the limits)")
            time.sleep(RATE_LIMIT_WAIT)
    raise RuntimeError("Voyage rate limit: gave up. Add a payment method at dash.voyageai.com to raise limits.")


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.5) + 8   # rough, errs on the high side


def _batches(texts: list[str]) -> list[list[str]]:
    """Group texts so each request stays under MAX_TOKENS_PER_REQUEST.

    Voyage's free tier allows about 10K tokens per minute, so a request bigger than that
    can never succeed; small requests also recover faster after a rate limit.
    """
    batches, current, size = [], [], 0
    for t in texts:
        n = _estimate_tokens(t)
        if current and (size + n > MAX_TOKENS_PER_REQUEST or len(current) >= BATCH_SIZE):
            batches.append(current)
            current, size = [], 0
        current.append(t)
        size += n
    if current:
        batches.append(current)
    return batches


def embed_documents(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for batch in _batches(texts):
        out.extend(_embed(batch, "document"))
    return out


def embed_query(text: str) -> list[float]:
    return _embed([text], "query")[0]
