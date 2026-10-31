# Demo script (about 5 minutes)

It follows the brief's demo flow: **import → multi-source request → open citations → conflicting /
unanswerable request**, then the case report and our measurements.

## Before the demo (do this 30+ minutes early)

- [ ] `git pull`, `.env` has all keys, `python scripts/check_connection.py` and `python scripts/check_llm.py` both ✅
- [ ] Check the main AI model is not rate-limited: `check_llm.py` should not print "trying another model"
- [ ] Remove any test documents: `python scripts/remove_document.py STMT-011 --yes` (sidebar should show **40** documents)
- [ ] `streamlit run app.py`, browser zoom so text is readable from the back (**Ctrl +**)
- [ ] Generate one **case report** and download it, as a backup in case the AI is slow on stage
- [ ] Open `data/demo/hotel_night_porter.txt` in a text editor, ready to copy
- [ ] Close other apps using the same Voyage / Gemini keys (free tiers are shared per key)
- [ ] Optional: screenshots of each step, in case the venue Wi-Fi fails

## The script

### 1. The hook (20 s)
> "Investigators drown in statements, CCTV logs and phone records, and the hard part isn't finding
> text. It's knowing **who's lying, what's proven, and what's still unknown**. Case File answers
> questions from your own case files, shows the exact passage behind every claim, flags
> contradictions, and refuses to guess."

Point at the sidebar: **40 documents, 88 chunks**, all in MongoDB Atlas with metadata and embeddings.

### 2. Import a document and it's searchable (60 s)
1. **Ask** tab, type: `Did Marcus Whelan leave his hotel during the night?`
   - Expect ❌ or 🟡: only his own claim and a gap in his phone records. Say:
   > "Right now the files can't answer this. Only Marcus himself says where he was."
2. **Add documents** tab → *Option 2: paste plain text*:
   - Type: `witness statement` · Title: `Witness statement of Declan Shaw` · Author: `Declan Shaw`
   - Date: `2026-10-06` · Time: `10:00` · People: `Marcus Whelan, Declan Shaw`
   - Paste the text of `data/demo/hotel_night_porter.txt` → **Add document**
   - Expect: `[STMT-011] added: … chunk(s) … embedded`. Sidebar shows **41**. Say:
   > "It was just stored in MongoDB with its metadata, chunked, embedded with Voyage, and picked up by
   > both Atlas indexes automatically."
3. Back to **Ask**, same question again → now it cites **[STMT-011]**: he left about 01:20 and came
   back about 02:20 carrying a grey case.

### 3. A question that needs several documents (45 s)
Click the demo question **"Did Marcus Whelan return to the hub after leaving at 23:31?"**
> "No single document says this. The app combined his statement, the car-park camera that read his
> rental car's plate 231-D-55120 at 01:39, and the door log where his badge HB-9042 opened the back
> door at 01:43. It even ran follow-up searches for those codes by itself."

Open **🔬 How this answer was found** to show the follow-up searches.

### 4. Open the citations (30 s)
Under **📄 Sources**, open `[CCTV-004]` or `[ACCESS-001]`:
> "Every source opens the original document with the exact quoted passage highlighted. And the code
> checks every quote word-for-word. If the AI misquotes, you get a ⚠️, not a silent error."

### 5. Conflicting and unanswerable requests (60 s)
1. Demo question **"Was the security desk staffed at around 01:30?"** → ⚠️ **Conflicting evidence**
   > "The guard swore he never left. CCTV, the door log and his own phone record show a 27-minute call
   > outside, right in the theft window. The app shows both sides and which one the records support."
2. Demo question **"Why was Team 7's prototype stolen, and who was the thief working for?"** → ❌
   > "No document states a motive, so it says so, instead of inventing one."
3. Optional: **"Where is the stolen laptop right now?"** → it gives only the *last known* ping (02:21) and says the current location is unknown.

### 6. Hybrid search (30 s, optional if short on time)
**Search lab** → `231-D-55120` → **Compare**:
> "Search by meaning alone is bad at codes like number plates. Keyword search finds them exactly.
> Hybrid search combines both with Reciprocal Rank Fusion."

### 7. The case report (30 s)
**Case report** tab → **Generate case report** (or open the downloaded backup):
> "One click: a timeline, persons of interest, findings rated by strength, contradictions and open
> questions. Every line is cited, and lines without a valid citation are removed automatically.
> It never calls anyone guilty: Marcus is 'subject of open enquiries', and the two innocent suspects
> are 'cleared by records'."

### 8. The numbers (30 s)
> "We kept 20 test questions with expected source ids and measured every change:
> - 10 sources instead of 5 raised recall from about 80% to about 90%;
> - hybrid search ranks the right document highest (MRR 0.85);
> - a similarity threshold can't detect unanswerable questions, because they scored *higher* than real ones;
> - so the verdict is decided by code: honesty cases went from **1/7 to 6/7**, with **100%** of quotes verified."

## If something goes wrong
| Problem | What to do |
|---|---|
| Spinner takes > 30 s | Say "free-tier AI, it's queueing", and narrate the pipeline while it runs |
| "busy or rate-limiting every model" | Wait 30 s and retry, or switch to the pre-generated report and screenshots |
| Wi-Fi dies | Use a phone hotspot; worst case, walk through the screenshots |
| A wrong answer on stage | Open its sources: "this is why every claim is linked, so you can check it in one click" |

## Likely judge questions
- **Why not paste all the documents into ChatGPT?** It doesn't scale beyond a context window, can't
  filter by date/type/person, can't verify quotes against stored originals, and can't show
  measured retrieval quality. MongoDB keeps sources, chunks, embeddings and metadata together.
- **How do you stop hallucinations?** The AI sees only retrieved evidence; every quote is checked
  word-for-word in code; uncited report lines are deleted; the verdict is decided by fixed rules.
- **Why hybrid search?** Measured: vector search alone ranked an unrelated phone record #1 for a
  plate number; keyword search alone misses paraphrases (multi-source recall 47%).
- **Is the data real?** No, it's fictional and generated, with a hidden answer key, so we can
  measure whether the app finds the truth.
- **What would you do next?** Images and audio (CCTV stills, interview recordings); a timeline view;
  `$rankFusion` inside Atlas; per-case access control.
