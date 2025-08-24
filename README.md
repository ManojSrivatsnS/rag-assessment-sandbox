Option B — RAG Q&A (Money Classics)

Builds a small, working RAG app over four public‑domain “money mindset” books. It ingests a corpus, builds a FAISS vector index, retrieves top‑k chunks, and optionally uses an LLM to synthesize grounded answers with citations. Includes an eval harness, abstention (“I don’t know”), and a simple Streamlit UI.

Highlights
Embeddings   : MiniLM (sentence-transformers/all-MiniLM-L6-v2)

Index        : FAISS (cosine via normalized inner product)

Chunking     : 750 chars, 150 overlap (tunable)

Answering    : LLM on/off; abstains when confidence is low

Extras       : author boost, MMR diversity, citation de‑dup, prompt logging

Evaluation   : mini harness + threshold sweep

UI           : Streamlit demo (toggle LLM on/off)

Repository Layout:

option_b_rag_qa/

    app.py                   # CLI (ingest/ask/eval), core pipeline
    ui.py                    # Streamlit UI (optional)
    corpus/                  # 4 books (txt)
    rag_index/               # FAISS index artifacts (generated or checked in)
    questions.json           # mini eval set (in/out-of-domain)
    requirements.txt         # minimal deps
PROMPTLOG.md               # auto-logged prompts (created at runtime)

.gitignore

# 1. Clone -> Install -> Run
  ## Windows/PowerShell
    
    # clone
       git clone https://github.com/ManojSrivatsnS/rag-assessment-sandbox.git
       cd rag-assessment-sandbox
    
    # create & activate venv
        python -m venv .venv
        .\.venv\Scripts\Activate.ps1
    
    # install deps
        python -m pip install --upgrade pip
        python -m pip install -r option_b_rag_qa\requirements.txt
    
    # (optional) for the UI
        python -m pip install streamlit

 ## macOS / Linux
    # clone
      git clone https://github.com/ManojSrivatsnS/rag-assessment-sandbox.git
      cd rag-assessment-sandbox

    # create & activate venv
      python3 -m venv .venv
      source .venv/bin/activate
    
    # install deps
      python -m pip install --upgrade pip
      python -m pip install -r option_b_rag_qa/requirements.txt
    
    # (optional) UI
      python -m pip install streamlit

# 2. Environment (LLM optional)
  ## Create option_b_rag_qa/.env (not committed):
     OPENAI_API_KEY=sk-...
     OPENAI_MODEL=gpt-4o-mini

   LLM optional    : If no key is present or --llm off, the app returns a grounded extract (no generation).
    With a key and --llm on, it synthesizes a concise answer with citations.
    All prompts are auto‑logged to PROMPTLOG.md (for evidence of AI use).
    
# 3. Build the index (ingestion)
    python option_b_rag_qa\app.py ingest --corpus option_b_rag_qa\corpus --index option_b_rag_qa\rag_index

  #### Optional tunables:
    # bigger chunks, smaller overlap
    python option_b_rag_qa\app.py ingest --corpus option_b_rag_qa\corpus --index option_b_rag_qa\rag_index --chunk_size 1000 --overlap 100

  #### Check if the Artifacts written to option_b_rag_qa/rag_index/:
    index.faiss, meta.jsonl, embedder.json

# 4. CLI execution
  #### LLM OFF (fast, extractive)
    python option_b_rag_qa\app.py ask "What does Barnum advise about frugality?" --index option_b_rag_qa\rag_index --llm off --verbose

  #### LLM ON (synthesized, grounded)
    python option_b_rag_qa\app.py ask "What does Barnum advise about frugality?" --index option_b_rag_qa\rag_index --llm on --model gpt-4o-mini --temp 0 --verbose

  #### Handling “I don’t know” (Out of Domain)
    # LLM-driven abstention (retrieval passes but model abstains)
    python option_b_rag_qa\app.py ask "Who won the Battle of Hastings?" --index option_b_rag_qa\rag_index --llm on --threshold 0.25    
    # Retrieval-driven abstention (force low-confidence)
    python option_b_rag_qa\app.py ask "Who won the Battle of Hastings?" --index option_b_rag_qa\rag_index --llm on --threshold 0.5

  #### Useful knobs 
    --threshold 0.25         -> abstain if top score < threshold (we chose 0.25 via sweep)
    --author_boost on|off    -> nudge results if query mentions Barnum/Smiles/Conwell/Wattles
    --mmr on|off             -> diversify near‑duplicate chunks
    --dedupe on|off          -> de‑dupe citations by file
    --local_rewriter on|off  -> optional local summary fallback (big download once; default off)

# 5. Streamlit UI execution (optional)
    python -m streamlit run option_b_rag_qa\ui.py

## In the app:
  Set Index directory 'dir' to option_b_rag_qa/rag_index
  Toggle Use LLM on/off to compare outputs
  Ask an in‑domain Question to expect grounded answer and an Out of Domain (OOD) Question  to expect “I don’t know.” response.
  Expand Citations and Retrieved chunks to inspect grounding
  
# 6. Mini Evaluation Harness
  option_b_rag_qa/questions.json (JSON array) mixes in‑domain and Out of Domain, e.g.:
  [
    {"question": "What does Barnum say about advertising?", "expect_doc":"barnum_art_of_money_getting.txt", "ood": false},
    {"question": "What is Wattles' core method to get rich?", "expect_doc":"wattles_science_of_getting_rich.txt", "ood": false},
    {"question": "Which habits does Samuel Smiles emphasise for self-improvement?", "expect_doc":"smiles_self_help.txt", "ood": false},
    {"question": "What is the central idea of Acres of Diamonds?", "expect_doc":"conwell_acres_of_diamonds.txt", "ood": false},
    {"question": "Who won the Battle of Hastings?", "expect_doc": null, "ood": true},
    {"question": "What is the boiling point of mercury?", "expect_doc": null, "ood": true}
  ]

### Single run:
  python option_b_rag_qa\app.py eval --index option_b_rag_qa\rag_index --qas option_b_rag_qa\questions.json --k 5 --threshold 0.25 --llm on
Outputs console metrics and writes eval_report.md.

### Threshold sweep (pick a threshold):
  python option_b_rag_qa\app.py eval --index option_b_rag_qa\rag_index --qas option_b_rag_qa\questions.json --k 5 --sweep 0.2,0.25,0.3 --llm on

### Writes:
  eval_threshold_sweep.json
  eval_threshold_sweep.md (table of hit@k, citation@ref, abstain P/R)
Recommended: use 0.25 for strong abstain precision while keeping recall reasonable.

# 7. Prompt logging (evidence of AI use)
When --llm on, the request prompt & top retrieval scores are auto‑appended to PROMPTLOG.md, including:
  - ISO timestamp
  - model & temperature
  - system + user messages (with the serialized sources)
  - top retrieval scores
Include a few entries in your process report / Loom.

# 8. Design & defaults

  Chunking            : 750/150 works well for long prose; tune if needed.
  
  Similarity          : L2‑normalized embeddings + FAISS IP ≈ cosine similarity.
  
  Abstention  :
  
    - Retrieval threshold: top_score < 0.25 -> “I don’t know.”
    - LLM‑level abstention: if generated text is an “I don’t know” variant, we flag abstained=true and suppress citations.
  
  Dedupe & filtering  : citations are de‑duplicated per file; very weak citations (<20% of best score) are dropped.

# 9. Troubleshooting
  streamlit : not recognized          -> python -m pip install streamlit; run via python -m streamlit run option_b_rag_qa\ui.py
  
  OpenAI 429 / quota                  -> you’ll see LLM_ERROR; app gracefully falls back to extractive
  
  HF model download big (~1.2GB)      -> only if --local_rewriter on. Default is off.
  
  FAISS/index errors                  -> re-run ingest to regenerate rag_index/
  
  Windows symlink warnings (HF cache) -> harmless; can ignore

# 10. Safety & secrets
  Never commit secrets. .env is ignored.
  If GitHub blocks a push for secrets:
    git rm --cached path/to/secret.file
    git commit --amend -C HEAD
    git push

# 11. Commands cheat‑sheet
    # Ingest
    python option_b_rag_qa\app.py ingest --corpus option_b_rag_qa\corpus --index option_b_rag_qa\rag_index
    
    # Ask (LLM off / on)
    python option_b_rag_qa\app.py ask "..." --index option_b_rag_qa\rag_index --llm off
    python option_b_rag_qa\app.py ask "..." --index option_b_rag_qa\rag_index --llm on --model gpt-4o-mini --temp 0
    
    # OOD → I don't know
    python option_b_rag_qa\app.py ask "Who won the Battle of Hastings?" --index option_b_rag_qa\rag_index --llm on --threshold 0.25
    
    # Eval
    python option_b_rag_qa\app.py eval --index option_b_rag_qa\rag_index --qas option_b_rag_qa\questions.json --k 5 --threshold 0.25 --llm on
    
    # Sweep
    python option_b_rag_qa\app.py eval --index option_b_rag_qa\rag_index --qas option_b_rag_qa\questions.json --k 5 --sweep 0.2,0.25,0.3 --llm on
    
    # UI
    python -m streamlit run option_b_rag_qa\ui.py

# 12. License
  Public domain texts courtesy of Project Gutenberg. This code is provided for assessment/demo purposes.
