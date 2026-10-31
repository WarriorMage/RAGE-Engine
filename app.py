"""Case File: web app (Step 11).

Run from the repo root:
    streamlit run app.py

Tabs:
  Ask           - question -> verdict, cited answer, contradictions, sources with highlighted passages
  Library       - browse and read every document in the collection
  Add documents - upload or paste documents; they are chunked, embedded and searchable at once
  Search lab    - compare vector, keyword and hybrid retrieval side by side
"""

import html
import re
from datetime import date as Date
from datetime import time as Time

import streamlit as st

from casefile.answer import answer_question
from casefile.db import get_db
from casefile.embeddings import embed_query
from casefile.ingest import SOURCES
from casefile.llm import model_name, provider
from casefile.parsing import parse_document
from casefile.pipeline import ID_PREFIXES, add_document, build_raw, collection_stats, next_source_id
from casefile.search import DOC_TYPES, Filters, search

st.set_page_config(page_title="Case File", page_icon="🕵️", layout="wide")

DEMO_QUESTIONS = [
    "Was the security desk staffed at around 01:30?",
    "Did Marcus Whelan return to the hub after leaving at 23:31?",
    "Who used badge HB-9042?",
    "What did the person who got out of the car at 01:40 look like?",
    "Where is the stolen laptop right now?",
    "Why was Team 7's prototype stolen, and who was the thief working for?",
]
STATUS_STYLE = {
    "answered": ("success", "✅ Answered: the case files answer this question."),
    "conflicting": ("warning", "⚠️ Conflicting evidence: sources disagree. Both sides are shown below."),
    "partial": ("info", "🟡 Partly answered: some of this is not in the case files."),
    "insufficient": ("error", "❌ Not in the case files: the app will not guess."),
}
TYPE_LABELS = {t: t.replace("_", " ") for t in DOC_TYPES}

st.markdown("""
<style>
.doc-body { white-space: pre-wrap; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.85rem;
            background: rgba(127,127,127,0.08); padding: 0.8rem 1rem; border-radius: 0.5rem;
            max-height: 420px; overflow-y: auto; line-height: 1.45; }
.doc-body mark { background: #ffd54a; color: #000; padding: 0 2px; border-radius: 3px; }
.cite { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.85em; background: rgba(66,135,245,0.15);
        padding: 1px 5px; border-radius: 4px; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers

@st.cache_resource(show_spinner=False)
def db():
    return get_db()


@st.cache_data(ttl=30, show_spinner=False)
def stats() -> dict:
    return collection_stats(db())


@st.cache_data(ttl=60, show_spinner=False)
def load_source(source_id: str) -> dict | None:
    return db()[SOURCES].find_one({"_id": source_id}, {"raw": 0})


def highlight(body: str, spans: list[tuple[int, int]]) -> str:
    """HTML for a document body with the given character spans highlighted."""
    merged: list[list[int]] = []
    for s, e in sorted((s, e) for s, e in spans if s is not None and e is not None and 0 <= s < e <= len(body)):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, pos = [], 0
    for s, e in merged:
        out.append(html.escape(body[pos:s]))
        out.append(f'<mark id="q{s}">{html.escape(body[s:e])}</mark>')
        pos = e
    out.append(html.escape(body[pos:]))
    return f'<div class="doc-body">{"".join(out)}</div>'


def render_citations_inline(text: str) -> str:
    """Escape the answer text and style [SOURCE-ID] markers as small chips."""
    safe = html.escape(text)
    return re.sub(r"\[([A-Z]+-\d+)\]", r'<span class="cite">\1</span>', safe).replace("\n", "<br>")


def fmt_date(d) -> str:
    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d or "")


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🕵️ Case File")
    st.caption("Evidence-backed answers from your own case files: MongoDB Atlas Vector Search + Atlas Search.")
    try:
        s = stats()
        c1, c2 = st.columns(2)
        c1.metric("Documents", s["sources"])
        c2.metric("Chunks", s["chunks"])
        if s["embedded"] < s["chunks"]:
            st.warning(f"{s['chunks'] - s['embedded']} chunks have no embedding yet. Run scripts/embed.py.")
        with st.expander("Documents by type"):
            for t, n in s["by_type"].items():
                st.write(f"{TYPE_LABELS.get(t, t)}: **{n}**")
    except Exception as e:  # show connection problems instead of crashing
        st.error(f"Cannot reach MongoDB: {e}")
        st.stop()
    st.divider()
    st.caption(f"AI: {provider()} / {model_name()}")


ask_tab, library_tab, add_tab, lab_tab = st.tabs(["🔎 Ask", "📂 Library", "➕ Add documents", "🧪 Search lab"])


# ---------------------------------------------------------------- Ask

def filters_ui(key: str) -> Filters | None:
    with st.expander("Filters (optional)"):
        c1, c2, c3 = st.columns(3)
        types = c1.multiselect("Document types", DOC_TYPES, format_func=TYPE_LABELS.get, key=f"{key}_types")
        d_from = c2.date_input("From date", value=None, key=f"{key}_from")
        d_to = c3.date_input("To date", value=None, key=f"{key}_to")
        people = st.text_input("Mentions person (exact name, comma-separated)", key=f"{key}_people")
    try:
        return Filters(types, d_from.isoformat() if d_from else None, d_to.isoformat() if d_to else None,
                       [p.strip() for p in people.split(",") if p.strip()])
    except ValueError as e:
        st.error(str(e))
        return None


def show_answer(a) -> None:
    kind, text = STATUS_STYLE.get(a.status, ("info", a.status))
    getattr(st, kind)(text)
    st.markdown(f"<div style='font-size:1.05rem; line-height:1.6'>{render_citations_inline(a.answer)}</div>",
                unsafe_allow_html=True)

    if a.question_parts:
        marks = {"yes": "✅", "partly": "🟡", "no": "❌"}
        st.caption("Question parts: " + " · ".join(f"{marks.get(p['covered'], '?')} {p['part']}" for p in a.question_parts))

    if a.conflicts:
        st.subheader("⚠️ Contradictions")
        for c in a.conflicts:
            with st.container(border=True):
                st.markdown(f"**{c.topic}**")
                st.markdown(f"🗣️ **Claim:** {c.claim} " + " ".join(f"`{s}`" for s in c.claim_sources))
                st.markdown(f"📼 **But:** {c.contradiction} " + " ".join(f"`{s}`" for s in c.contradiction_sources))
                st.markdown(f"⚖️ **Assessment:** {c.assessment}")

    if a.missing:
        st.subheader("❓ Not in the case files")
        for m in a.missing:
            st.markdown(f"- {m}")

    for w in a.warnings:
        st.warning(f"Check: {w}")

    st.subheader(f"📄 Sources ({len(a.cited_sources)})")
    st.caption("Open a source to read the original document with the quoted passages highlighted.")
    for sid in a.cited_sources:
        cites = [c for c in a.citations if c.source_id == sid]
        verified = all(c.quote_verified for c in cites)
        with st.expander(f"{'✅' if verified else '⚠️'} [{sid}] {cites[0].title}  ({TYPE_LABELS.get(cites[0].doc_type, '')})"):
            for c in cites:
                st.markdown(f"> {c.quote}" + ("" if c.quote_verified else "  \n⚠️ *quote not found word-for-word*"))
            src = load_source(sid)
            if src:
                st.caption(f"{src['title']} · {fmt_date(src.get('date'))} {src.get('time', '')} · by {src.get('author', '')}")
                st.markdown(highlight(src["body"], [(c.quote_start, c.quote_end) for c in cites]), unsafe_allow_html=True)

    with st.expander(f"🔬 How this answer was found ({len(a.evidence)} evidence chunks, model {a.model})"):
        if a.follow_up_searches:
            st.write("Follow-up keyword searches: " + ", ".join(f"`{q}`" for q in a.follow_up_searches))
        for e in a.evidence:
            st.markdown(f"**{e['label']}** `{e['source_id']}` {e['title']}")
            st.caption(" ".join(e["text"].split())[:220] + "...")


with ask_tab:
    st.header("Ask the case files")
    st.write("Try a demo question:")
    cols = st.columns(3)
    for i, q in enumerate(DEMO_QUESTIONS):
        if cols[i % 3].button(q, key=f"demo{i}", width="stretch"):
            st.session_state["question"] = q
            st.session_state["run"] = True

    with st.form("ask_form"):
        question = st.text_input("Your question", key="question", placeholder="e.g. Who came in through DOOR-E2 at night?")
        submitted = st.form_submit_button("Ask", type="primary")
    filters = filters_ui("ask")

    if (submitted or st.session_state.pop("run", False)) and question.strip() and filters is not None:
        with st.spinner("Searching the case files and checking the evidence... (free AI tiers can take ~30s)"):
            try:
                st.session_state["answer"] = answer_question(db(), question.strip(), filters)
            except RuntimeError as e:
                st.session_state.pop("answer", None)
                st.error(str(e))
    if "answer" in st.session_state:
        st.divider()
        st.markdown(f"#### Q: {st.session_state['answer'].question}")
        show_answer(st.session_state["answer"])


# ---------------------------------------------------------------- Library

with library_tab:
    st.header("Library")
    pick_types = st.multiselect("Show types", DOC_TYPES, format_func=TYPE_LABELS.get, key="lib_types")
    query = {"doc_type": {"$in": pick_types}} if pick_types else {}
    docs = list(db()[SOURCES].find(query, {"title": 1, "doc_type": 1, "date": 1, "time": 1, "author": 1,
                                         "chunk_count": 1}).sort("_id", 1))
    st.dataframe(
        [{"id": d["_id"], "title": d["title"], "type": TYPE_LABELS.get(d["doc_type"], d["doc_type"]),
          "date": fmt_date(d.get("date")), "author": d.get("author", ""), "chunks": d.get("chunk_count", 0)}
         for d in docs],
        hide_index=True, width="stretch",
    )
    chosen = st.selectbox("Read a document", [d["_id"] for d in docs], index=None, placeholder="Choose a document id")
    if chosen:
        src = load_source(chosen)
        st.subheader(f"[{chosen}] {src['title']}")
        st.caption(f"{TYPE_LABELS.get(src['doc_type'])} · {fmt_date(src.get('date'))} {src.get('time', '')} · "
                   f"by {src.get('author', '')} · people: {', '.join(src.get('people_mentioned') or []) or 'none'}")
        st.markdown(highlight(src["body"], []), unsafe_allow_html=True)


# ---------------------------------------------------------------- Add documents

def ingest_and_report(raw: str) -> None:
    try:
        with st.spinner("Storing, chunking and embedding..."):
            result, embedded = add_document(db(), raw)
    except (ValueError, RuntimeError) as e:
        st.error(f"Could not add the document: {e}")
        return
    st.success(f"[{result.source_id}] {result.status}: {result.chunk_count} chunk(s), {embedded} embedded. "
               "It is searchable within a few seconds.")
    stats.clear()
    load_source.clear()


with add_tab:
    st.header("Add documents")
    st.write("New documents are stored in MongoDB with their metadata, split into chunks, embedded with Voyage AI "
             "and picked up automatically by the Atlas Search and Vector Search indexes.")

    st.subheader("Option 1: upload files in case-file format")
    st.caption("Markdown/text files that start with the `---` metadata header (like data/case_files/*.md).")
    files = st.file_uploader("Upload .md or .txt", type=["md", "txt"], accept_multiple_files=True)
    if files and st.button("Import uploaded files", type="primary"):
        for f in files:
            raw = f.getvalue().decode("utf-8", errors="replace")
            try:
                parse_document(raw)
            except ValueError as e:
                st.error(f"{f.name}: {e}. Use option 2 for plain text.")
                continue
            st.write(f"**{f.name}**")
            ingest_and_report(raw)

    st.subheader("Option 2: paste plain text")
    with st.form("paste_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        doc_type = c1.selectbox("Document type", list(ID_PREFIXES), format_func=TYPE_LABELS.get)
        title = c2.text_input("Title", placeholder="e.g. Witness statement of Jane Doe")
        c3, c4, c5 = st.columns(3)
        author = c3.text_input("Author / owner")
        doc_date = c4.date_input("Date", value=Date(2026, 10, 4))
        doc_time = c5.time_input("Time", value=Time(12, 0))
        people = st.text_input("People mentioned (comma-separated, exact names)")
        body = st.text_area("Document text", height=220)
        add = st.form_submit_button("Add document", type="primary")
    if add:
        if not (title.strip() and author.strip() and body.strip()):
            st.error("Title, author and text are required.")
        else:
            sid = next_source_id(db(), doc_type)
            ingest_and_report(build_raw(sid, title, doc_type, author, doc_date.isoformat(),
                                        doc_time.strftime("%H:%M"), people.split(","), body))


# ---------------------------------------------------------------- Search lab

with lab_tab:
    st.header("Search lab: vector vs keyword vs hybrid")
    st.write("The same question through each retrieval method. Hybrid search merges the other two "
             "with Reciprocal Rank Fusion.")
    lab_q = st.text_input("Search for", value="231-D-55120", key="lab_q")
    lab_filters = filters_ui("lab")
    if st.button("Compare", type="primary") and lab_q.strip() and lab_filters is not None:
        with st.spinner("Searching..."):
            try:
                qvec = embed_query(lab_q)   # one Voyage call, shared by vector and hybrid
                results = {m: search(db(), lab_q, k=5, filters=lab_filters, mode=m, query_vector=qvec)
                           for m in ("vector", "text", "hybrid")}
            except RuntimeError as e:
                st.error(str(e))
                results = {}
        for col, (mode, rows) in zip(st.columns(3), results.items()):
            col.subheader({"vector": "🧠 Vector (meaning)", "text": "🔤 Keyword (BM25)", "hybrid": "🔀 Hybrid (RRF)"}[mode])
            for i, r in enumerate(rows, 1):
                ranks = ", ".join(f"{k} #{v}" for k, v in r.get("ranks", {}).items())
                col.markdown(f"**{i}. `{r['source_id']}`** {r['title']}")
                col.caption(f"score {r['score']:.4f}" + (f" · {ranks}" if ranks else ""))
