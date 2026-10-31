"""MongoDB connection helpers.

Settings come from environment variables, or from a .env file in the repo root.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

load_dotenv()

DEFAULT_DB_NAME = "case_file"


def _uri() -> str:
    uri = os.environ.get("MONGODB_URI", "").strip().strip('"').strip("'")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not set. Put it in your environment or in a .env file "
            "(see .env.example)."
        )
    return uri


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    """Return one shared MongoClient for the whole app."""
    return MongoClient(_uri(), appname="case-file-rag", serverSelectionTimeoutMS=10_000)


def get_db() -> Database:
    """Return the app's database (name from MONGODB_DB, default 'case_file')."""
    return get_client()[os.environ.get("MONGODB_DB", DEFAULT_DB_NAME)]
