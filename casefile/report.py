"""Generate a cited investigation report (Step 12): the "generate an artefact" part of the brief.

Pipeline:
  1. Gather   : hybrid search for several report angles (timeline, access, vehicles, ...),
                all query embeddings in ONE Voyage request; merge into one evidence set
  2. Write    : one LLM call returns a structured report that cites [E1], [E2], ...
  3. Verify   : quotes checked against the sources; timeline entries and findings with no
                valid citation are removed (and counted), so every factual line is backed
  4. Render   : Markdown with a Sources appendix, ready to download
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

from pymongo.database import Database

from casefile.answer import (
    LABEL_PATTERN,
    normalize_labels,
    Citation,
    format_evidence,
    label_evidence,
    replace_labels,
    select_evidence,
    verify_citations,
)
from casefile.embeddings import embed_queries
from casefile.llm import chat_json, last_model_used
from casefile.search import Filters, hybrid_search

# Angles searched to gather evidence for a whole-case report.
REPORT_ANGLES = [
    "timeline of events during the night with exact times",
    "who entered or left the building: door access logs and badge numbers",
    "vehicles in the car park, number plates and who they belong to",
    "witness statements that contradict CCTV, phone or door records",
    "forensic evidence found at the scene",
    "missing property, current status of the investigation and outstanding enquiries",
]
SOURCES_PER_ANGLE = 6
MAX_EVIDENCE = 40


# ---------------------------------------------------------------- data classes

@dataclass
class Report:
    focus: str
    title: str
    summary: str
    timeline: list[dict]              # {"time", "event"}            event has [SOURCE-ID] citations
    persons: list[dict]               # {"name", "role", "what_records_show", "status"}
    findings: list[dict]              # {"finding", "strength"}
    contradictions: list[dict]        # {"claim", "contradiction", "assessment"}
    open_questions: list[str]
    citations: list[Citation]
    evidence: list[dict]
    model: str = ""
    removed_uncited: int = 0          # lines dropped because they had no valid citation
    warnings: list[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    @property
    def cited_sources(self) -> list[str]:
        seen: list[str] = []
        for c in self.citations:
            if c.source_id not in seen:
                seen.append(c.source_id)
        return seen


# ---------------------------------------------------------------- 1. gather

def gather_evidence(db: Database, focus: str, filters: Filters | None) -> list[dict]:
    angles = ([focus] if focus.strip() else []) + REPORT_ANGLES
    vectors = embed_queries(angles)                      # one Voyage request for all angles
    picked: list[dict] = []
    seen: set[str] = set()
    for angle, vec in zip(angles, vectors):
        results = hybrid_search(db, angle, k=30, filters=filters, depth=30, query_vector=vec)
        for r in select_evidence(results, max_sources=SOURCES_PER_ANGLE):
            if r["_id"] not in seen and len(picked) < MAX_EVIDENCE:
                seen.add(r["_id"])
                picked.append(r)
    return label_evidence(picked)


# ---------------------------------------------------------------- 2. write

REPORT_SYSTEM = """You are Case File, writing an investigation brief for detectives.
Write it using ONLY the evidence excerpts provided.

RULES
- Every factual statement must cite its evidence like [E3]; cite all pieces when combining.
- Prefer ORIGINAL RECORDS over SUMMARY documents (police notes) when both show the same fact.
- Never state that someone is guilty. Describe what the records show and what remains unproven.
- People can be mistaken or lie: report claims as claims and compare them with records.
- Motives, current locations of property and the content of unrecorded conversations are
  unknown unless a document states them explicitly. Put such gaps in open_questions.
- Use exact times, codes and names from the evidence.
- For each citation give a short quote copied EXACTLY from that evidence excerpt.

SECTIONS
- title: short report title.
- summary: 4-6 cited sentences.
- timeline: key events in time order; "time" like "01:43" (or a date), "event" cited.
- persons: people of interest; "what_records_show" cited; "status" is one of
  "cleared by records", "unresolved", "subject of open enquiries", "witness".
- findings: key established facts; "strength" is "confirmed by records" (several independent
  records agree), "single source", or "claim only" (only someone's statement).
- contradictions: claim vs what other evidence shows, with an assessment, all cited.
- open_questions: important things the case files do NOT answer (no citations needed)."""

_STR = {"type": "string"}
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": _STR,
        "summary": _STR,
        "timeline": {"type": "array", "items": {
            "type": "object", "properties": {"time": _STR, "event": _STR},
            "required": ["time", "event"], "additionalProperties": False}},
        "persons": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": _STR, "role": _STR, "what_records_show": _STR,
                           "status": {"type": "string", "enum": ["cleared by records", "unresolved",
                                                                  "subject of open enquiries", "witness"]}},
            "required": ["name", "role", "what_records_show", "status"], "additionalProperties": False}},
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {"finding": _STR,
                           "strength": {"type": "string", "enum": ["confirmed by records", "single source", "claim only"]}},
            "required": ["finding", "strength"], "additionalProperties": False}},
        "contradictions": {"type": "array", "items": {
            "type": "object", "properties": {"claim": _STR, "contradiction": _STR, "assessment": _STR},
            "required": ["claim", "contradiction", "assessment"], "additionalProperties": False}},
        "open_questions": {"type": "array", "items": _STR},
        "citations": {"type": "array", "items": {
            "type": "object", "properties": {"evidence_id": _STR, "quote": _STR},
            "required": ["evidence_id", "quote"], "additionalProperties": False}},
    },
    "required": ["title", "summary", "timeline", "persons", "findings", "contradictions",
                 "open_questions", "citations"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- 3. verify

def _has_valid_label(text: str, evidence: list[dict]) -> bool:
    labels = {e["label"] for e in evidence}
    return any(m.group(1) in labels for m in LABEL_PATTERN.finditer(normalize_labels(text, labels)))


def _clean_items(items: list[dict], cited_field: str, evidence: list[dict]) -> tuple[list[dict], int]:
    """Keep items whose cited_field has a valid [E#]; convert labels to source ids in all fields."""
    kept, removed = [], 0
    for item in items:
        if not _has_valid_label(item.get(cited_field, ""), evidence):
            removed += 1
            continue
        kept.append({k: replace_labels(v, evidence) if isinstance(v, str) else v for k, v in item.items()})
    return kept, removed


def generate_report(db: Database, focus: str = "", filters: Filters | None = None) -> Report:
    evidence = gather_evidence(db, focus, filters)
    if not evidence:
        raise RuntimeError("No evidence found for this report. Check the filters or the collection.")

    request = f"REPORT FOCUS: {focus.strip() or 'the whole case'}\n\nEVIDENCE:\n{format_evidence(evidence)}"
    data = chat_json(REPORT_SYSTEM, request, REPORT_SCHEMA, "case_report")

    timeline, r1 = _clean_items(data.get("timeline", []), "event", evidence)
    persons, r2 = _clean_items(data.get("persons", []), "what_records_show", evidence)
    findings, r3 = _clean_items(data.get("findings", []), "finding", evidence)
    contradictions, r4 = _clean_items(data.get("contradictions", []), "contradiction", evidence)
    citations = verify_citations(data.get("citations", []), evidence)

    warnings = []
    removed = r1 + r2 + r3 + r4
    if removed:
        warnings.append(f"{removed} line(s) were removed because they had no valid citation.")
    bad = [c for c in citations if not c.quote_verified]
    if bad:
        warnings.append(f"{len(bad)} of {len(citations)} quotes could not be found word-for-word "
                        f"({', '.join(sorted({c.source_id for c in bad}))}).")
    summary = data.get("summary", "")
    if not _has_valid_label(summary, evidence):
        warnings.append("The summary has no citations; rely on the cited sections below.")

    return Report(
        focus=focus.strip() or "Whole case",
        title=data.get("title", "Case report"),
        summary=replace_labels(summary, evidence),
        timeline=timeline,
        persons=persons,
        findings=findings,
        contradictions=contradictions,
        open_questions=[q for q in data.get("open_questions", []) if q.strip()],
        citations=citations,
        evidence=evidence,
        model=last_model_used(),
        removed_uncited=removed,
        warnings=warnings,
    )


# ---------------------------------------------------------------- 4. render

def to_markdown(r: Report, source_titles: dict[str, str] | None = None) -> str:
    """The report as a Markdown document (what the Download button saves)."""
    titles = source_titles or {}
    for c in r.citations:
        titles.setdefault(c.source_id, c.title)
    lines = [f"# {r.title}", "",
             f"*Generated by Case File on {r.created} · focus: {r.focus} · model: {r.model}*", "",
             "> Every statement cites its source document in [brackets]. "
             "Lines without a valid citation were removed automatically.", ""]
    for w in r.warnings:
        lines.append(f"> ⚠️ {w}")
    lines += ["", "## Summary", "", r.summary, ""]
    if r.timeline:
        lines += ["## Timeline", "", "| Time | Event |", "|---|---|"]
        lines += [f"| {t['time']} | {t['event'].replace('|', '/')} |" for t in r.timeline]
        lines.append("")
    if r.persons:
        lines += ["## Persons of interest", "", "| Person | Role | What the records show | Status |", "|---|---|---|---|"]
        lines += [f"| {p['name']} | {p['role']} | {p['what_records_show'].replace('|', '/')} | {p['status']} |" for p in r.persons]
        lines.append("")
    if r.findings:
        lines += ["## Key findings", ""]
        lines += [f"- **{f['strength']}**: {f['finding']}" for f in r.findings]
        lines.append("")
    if r.contradictions:
        lines += ["## Contradictions", ""]
        for i, c in enumerate(r.contradictions, 1):
            lines += [f"{i}. **Claim:** {c['claim']}", f"   **But:** {c['contradiction']}",
                      f"   **Assessment:** {c['assessment']}", ""]
    if r.open_questions:
        lines += ["## Open questions (not answered by the case files)", ""]
        lines += [f"- {q}" for q in r.open_questions]
        lines.append("")
    used = sorted(set(re.findall(r"\[([A-Z]+-\d+)\]", "\n".join(lines))) | set(r.cited_sources))
    lines += ["## Sources", ""]
    for sid in used:
        lines.append(f"- **[{sid}]** {titles.get(sid, '')}")
        for c in r.citations:
            if c.source_id == sid:
                lines.append(f"  - \"{c.quote}\"{'' if c.quote_verified else ' ⚠️ not found word-for-word'}")
    return "\n".join(lines) + "\n"
