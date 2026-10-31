"""Step 2 test: check that we can reach MongoDB Atlas.

Run from the repo root:
    python scripts/check_connection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo.errors import ConfigurationError, OperationFailure, ServerSelectionTimeoutError

from casefile.db import get_client, get_db


def main() -> int:
    try:
        client = get_client()
        client.admin.command("ping")
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1
    except OperationFailure:
        print("❌ MongoDB rejected the username or password in MONGODB_URI.")
        print("   Check the password (no < > brackets) in Atlas > Database Access.")
        return 1
    except ServerSelectionTimeoutError:
        print("❌ Could not reach the cluster (timed out).")
        print("   Check Atlas > Network Access has 0.0.0.0/0, and the cluster is running.")
        return 1
    except ConfigurationError as e:
        print(f"❌ MONGODB_URI looks malformed: {e}")
        return 1

    db = get_db()
    info = client.server_info()

    # Prove we can write and read, then clean up.
    probe = db["_connection_check"]
    probe.insert_one({"hello": "case file"})
    found = probe.find_one({"hello": "case file"})
    probe.drop()

    print("Connected ✅")
    print(f"  MongoDB version : {info.get('version')}")
    print(f"  Database        : {db.name}")
    print(f"  Write/read test : {'passed' if found else 'FAILED'}")
    print(f"  Collections     : {db.list_collection_names() or '(none yet)'}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
