"""Step 12: generate a cited case report and save it as Markdown.

Run from the repo root:
    python scripts/report.py                                   # whole case
    python scripts/report.py --focus "movements of vehicle 231-D-55120"
    python scripts/report.py --type cctv_log door_access_log   # only some document types

Reports are saved to reports/ (ignored by git).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from casefile.db import get_db  # noqa: E402
from casefile.ingest import SOURCES  # noqa: E402
from casefile.report import generate_report, to_markdown  # noqa: E402
from casefile.search import Filters  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--focus", default="", help="optional focus for the report")
    parser.add_argument("--type", nargs="+", dest="doc_types", default=[], metavar="TYPE")
    parser.add_argument("--from", dest="date_from", metavar="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", metavar="YYYY-MM-DD")
    args = parser.parse_args()

    try:
        filters = Filters(args.doc_types, args.date_from, args.date_to)
        print("Gathering evidence and writing the report (about 30-60 seconds)...")
        db = get_db()
        report = generate_report(db, args.focus, filters)
    except (ValueError, RuntimeError) as e:
        print(f"❌ {e}")
        return 1

    titles = {d["_id"]: d["title"] for d in db[SOURCES].find({}, {"title": 1})}
    md = to_markdown(report, titles)
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"case-report-{datetime.now():%Y%m%d-%H%M%S}.md"
    out.write_text(md, encoding="utf-8")

    print(md)
    print(f"Saved to {out.relative_to(ROOT)}")
    print(f"{len(report.timeline)} timeline entries, {len(report.persons)} persons, {len(report.findings)} findings, "
          f"{len(report.contradictions)} contradictions, {len(report.cited_sources)} sources cited, "
          f"{report.removed_uncited} uncited line(s) removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
