"""Check that the AI provider (Gemini or OpenAI) works.

Run from the repo root:
    python scripts/check_llm.py            # test call with the configured model
    python scripts/check_llm.py --models   # also list models your key can use
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casefile.llm import _client, chat_json, model_name, provider  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "reply": {"type": "string"}},
    "required": ["ok", "reply"],
    "additionalProperties": False,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", action="store_true", help="list available models")
    args = parser.parse_args()

    try:
        name = provider()
        print(f"Provider: {name}   Model: {model_name()}")
        if args.models:
            ids = sorted(m.id.removeprefix("models/") for m in _client(name).models.list())
            if name == "gemini":
                ids = [i for i in ids if i.startswith("gemini")]
            print("Available models:\n  " + "\n  ".join(ids))
        result = chat_json("You are a test. Reply in JSON.", 'Say "hello detective" in reply and set ok to true.',
                           SCHEMA, "check")
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1

    print(f"Reply: {result}")
    print("AI connected ✅" if result.get("ok") else "⚠️ Got a reply, but ok was not true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
