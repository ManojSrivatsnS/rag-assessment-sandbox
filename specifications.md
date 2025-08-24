# Feature sepcifications - LAPS summary
  
  **Listen**       -> Answer questions over four public‑domain money classics with citations; abstain when unsure.
  
  **Aanalyse**
  * Corpus: .txt; remove Gutenberg boilerplate
  * Chunking: 750/150; embeddings: MiniLM; FAISS cosine (IP on L2‑norm).
  * Retrieval: top‑k with author boost + MMR; dedupe & filter citations.
  * Generation: LLM optional; system prompt forces “I don’t know.” when ungrounded.
  * Threshold‑based abstention + robust “I don’t know” detection.
    
  **Prioritise**   -> MVP first (RAG pipeline + CLI), then add: LLM, eval harness, threshold sweep, prompt logging, UI.
  
  **Ship**         -> CLI, README, questions.json, eval_report.md, sweep table, Loom.

  
