# 🕵️ Case File: evidence-backed answers from your own case files

**Case File is a RAG investigation assistant built on MongoDB Atlas.** You load a collection of
investigation documents (witness statements, interview transcripts, CCTV logs, door access logs,
phone records, emails and more), then ask questions in plain English. Every answer:

- cites the **exact passage** behind each claim, and you can open the original document with the quote highlighted;
- **catches contradictions**, e.g. *"the guard says he never left the desk, but CAM-01 shows him walking out at 01:24"*;
- **refuses to guess** when the files don't contain the answer, and lists what is unknown;
- gets a **verdict badge**: ✅ answered · ⚠️ conflicting evidence · 🟡 partly answered · ❌ not in the case files.

The app can also generate a full **cited case report** (timeline, persons of interest, findings,
contradictions, open questions) as a downloadable document.

> **Why investigations?** Most RAG demos treat conflicting or unanswerable questions as edge
> cases. In an investigation, they're the whole job: witnesses misremember, suspects lie, and
> some facts are simply unknown. So the hardest part of the brief is the core feature here.

---

## The dataset: "The Midnight Merge"

A fictional theft at a 24-hour student hackathon in Dublin: a prototype and the laptop holding its
only source code vanish from Team 7's table overnight. The collection is
[**40 documents**](data/case_files/) across 10 types, written to include:

| Built into the data | Why it matters for the demo |
|---|---|
| 3 contradictions between witnesses | Conflict detection |
| A lie that only CCTV + door logs **together** disprove | Multi-source reasoning |
| 2 red herrings (suspicious but innocent) | The app must clear people using records |
| The key identity link spread over **3+ documents** | No single document gives it away |
| 3 facts recorded **nowhere** (motive, laptop's current location, call content) | Honest refusals |
| 15+ exact codes (badges `HB-9042`, plates `231-D-55120`, receipts) | Hybrid / keyword search |

The ground truth is in [`answer_key/case_bible.md`](answer_key/case_bible.md). It is **never**
imported into the app; it exists only to write the test questions and check answers.

---

## How it works

```
                 ┌──────────────────────── MongoDB Atlas ────────────────────────┐
 case files ──►  │ sources  : original document + metadata (type, date, author,  │
 (.md / paste)   │            people mentioned)                                   │
   │ parse,      │ chunks   : ~140-word pieces + same metadata + exact character  │
   │ chunk,      │            offsets + Voyage embedding (1024 numbers)           │
   │ embed       │ indexes  : Vector Search (meaning)  +  Atlas Search (BM25)     │
   ▼             └────────────────────────────────────────────────────────────────┘
 question ──► hybrid search (vector + keyword, Reciprocal Rank Fusion, metadata filters)
          ──► top 10 sources ──► AI suggests keyword follow-ups for codes it spotted (e.g. HB-9042)
          ──► AI answers ONLY from the evidence, citing [E1], [E2] ... with exact quotes
          ──► code checks: every quote must exist word-for-word in its source;
              the verdict is decided by fixed rules from the question parts + listed conflicts
          ──► answer + verdict + contradictions + "not in the case files" + clickable sources
```

| Piece | Choice |
|---|---|
| Database & search | **MongoDB Atlas**: `$vectorSearch` with pre-filters, `$search` (BM25), stored together with sources, chunks, embeddings and metadata |
| Embeddings | **Voyage AI** `voyage-4` (documents and queries embedded with different `input_type`) |
| Answers | **Google Gemini** (`gemini-flash-latest`, free tier) with automatic backup models; OpenAI supported via `LLM_PROVIDER=openai` |
| Web app | **Streamlit** |

### Design decisions, backed by measurements
- **Hybrid search by default.** Vector search alone missed exact codes (e.g. a plate number ranked an unrelated phone record #1), while keyword search missed paraphrases. Hybrid ranks the first correct document highest (best MRR, below). Queries containing codes like `HB-9042` give the keyword side double weight.
- **10 sources per answer, not 5.** Recall rose from 80–83% at 5 to 87–90% at 10.
- **No similarity-score threshold for "I don't know".** Our unanswerable questions scored 0.74–0.82, *higher* than many answerable ones, so a threshold can't work. Instead the AI judges the evidence and the code enforces the verdict.
- **The code decides the verdict, not the AI.** When the AI chose its own label, only 1/7 honesty cases were right. Splitting the question into parts and applying fixed rules raised that to 6/7, even on the weakest backup model.
- **Police notes are labelled as summaries**, so answers cite original records first.

---

## Results

Measured with [`eval/test_questions.json`](eval/test_questions.json): 20 questions (6 multi-source,
4 conflict, 3 unanswerable, 4 exact-code, 3 filtered), each with the source ids that contain the answer.

**Retrieval** (`python scripts/evaluate.py`), unique source documents in the top *k*:

| Mode | Recall@5 | Recall@10 | MRR | Conflict recall@5 |
|---|---|---|---|---|
| Vector only | 83% | **90%** | 0.80 | 74% |
| Keyword only (BM25) | 73% | 87% | 0.79 | 75% |
| **Hybrid (RRF)** | 80% | 87% | **0.85** | **80%** |

Hit@5 is 100% for every mode: every question finds at least one right source in the top 5.

**Answers** (`python scripts/evaluate_answers.py --types conflict unanswerable`):

| | AI picks its own verdict | Verdict decided by code |
|---|---|---|
| Conflicts flagged ⚠️ | 1/4 | **4/4** |
| Unanswerable refused ❌ | 0/3 | **2/3** (the miss gave the true last-known location and invented nothing) |
| Quotes verified word-for-word | 98% | **100%** |

*Honest note:* these answer runs used Gemini's free tier, whose rate limits often pushed the work onto the
smallest backup model (`gemini-flash-lite-latest`). Expect better results from the main model.

---

## Setup

You need Python 3.10+, a free [MongoDB Atlas](https://www.mongodb.com/atlas) cluster, a
[Voyage AI](https://dash.voyageai.com) key and a free [Gemini](https://aistudio.google.com/apikey) key.

```bash
git clone git@github.com:WarriorMage/RAGE-Engine.git
cd RAGE-Engine
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env, then fill in your keys
```

In Atlas, allow your IP under **Network Access** (or `0.0.0.0/0` for a hackathon).

**Load the data (once per database):**
```bash
python scripts/check_connection.py   # MongoDB reachable?
python scripts/check_dataset.py      # 40 case files + 20 test questions consistent?
python scripts/ingest.py             # sources + chunks into MongoDB
python scripts/embed.py              # Voyage embeddings (slow mode on free tier, a few minutes)
python scripts/create_indexes.py     # Vector Search + Atlas Search indexes
python scripts/check_llm.py          # AI reachable?
```

**Run the app:**
```bash
streamlit run app.py                 # opens http://localhost:8501
```

### Command-line tools
| Command | What it does |
|---|---|
| `python scripts/search.py "HB-9042" --mode hybrid\|vector\|text --type cctv_log --from 2026-10-10` | Search with filters |
| `python scripts/ask.py "Was the security desk staffed at around 01:30?"` | Cited answer with verdict |
| `python scripts/report.py --focus "vehicle 231-D-55120"` | Cited case report → `reports/` |
| `python scripts/evaluate.py -k 10 --details` | Retrieval scoreboard |
| `python scripts/evaluate_answers.py --types conflict unanswerable` | Answer scoreboard |
| `python scripts/remove_document.py STMT-011` | Remove a document added during a demo |

### Free-tier notes
- Voyage without billing allows ~3 requests/minute; the code switches to "slow mode" automatically.
- Gemini's free tier limits each model separately; busy or rate-limited models fall back to backups automatically. Each question uses one Voyage call and two AI calls.

---

## Project layout
```
app.py                    Streamlit web app (Ask, Case report, Library, Add documents, Search lab)
casefile/
  parsing.py chunking.py  read documents, split into chunks with exact character offsets
  ingest.py pipeline.py   store sources + chunks; add a document end to end
  embeddings.py           Voyage embeddings with free-tier pacing
  search.py               filters, vector / keyword / hybrid (RRF) search, Atlas index definitions
  answer.py               retrieve → follow-ups → cited answer → quote checks → verdict
  report.py               cited case report
  llm.py                  Gemini/OpenAI structured JSON output with model fallback
scripts/                  command-line tools (see above)
data/case_files/          the 40 case documents
data/demo/                a document for the live "add a document" demo
eval/test_questions.json  20 test questions with expected source ids
answer_key/case_bible.md  ground truth (never imported into the app)
docs/DEMO_SCRIPT.md       what to click and say in the demo
```
