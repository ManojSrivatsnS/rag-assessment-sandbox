# Process Report — Option B (RAG Q&A)

## Tools Used
  * ChatGPT / GPT-4o-mini (codegen for CLI, eval harness, prompt shaping)
  * Local IDE (VS Code) for integration
  * FAISS + sentence-transformers (MiniLM)
  * Streamlit for quick UI
  * GitHub for version control; prompt logs in PROMPTLOG.md

## Key Prompts (3+ exact)
1) **System instruction refinement :**

   &emsp; &emsp; You are a careful assistant. Only answer using the provided sources. If the sources are insufficient, say exactly: 'I don't know.'
   
   &emsp; &emsp; Be concise, include short quotes, and cite sources like [1], [2].
   
   &emsp; &emsp; **Impact**
   LLM abstains cleanly; prevents hallucinations.
3) **User template with serialized chunks**

  &emsp; &emsp; Question: {question}
  
  &emsp; &emsp; Sources:{numbered [i] (file:start-end) chunk_text blocks} Answer clearly with citations like [1], [2].

  &emsp; &emsp; **Impact**: More grounded answers; better citations.

 3) **Abstention bug fix prompt**
   &emsp; &emsp; Detect 'I don’t know' variants from model output (smart quotes, with [1] trailers), strip citations, and set abstained=true. Return no citations when abstaining.

   &emsp; &emsp; **Impact**: Consistent abstain metrics; no misleading citations.

## What AI generated vs. what I hand-wrote
- AI: initial CLI scaffolding, prompt structure, error-handling patterns.
- Me: retrieval helpers (author boost, MMR), citation filtering, eval harness, threshold sweep, robust abstention detection, UI glue.

## Trade-offs
- MiniLM + FAISS chosen for speed & reproducibility vs. heavier rerankers.
- Threshold-based abstention is simple; recall could improve with cross-encoder reranking.
- Local rewriter disabled by default to avoid 1.2GB download; kept as fallback.

## Validation
- CLI ask (LLM on/off) + OOD abstention
- Eval harness with in-domain + OOD; sweep to pick threshold (0.25)
- Streamlit demo for UX and transparency

## Known limits / Next steps
- Add LLM reranker or cross-encoder
- Parent–child chunking for long-range answers
- Tiny entity graph for query expansion
- Result caching layer


   
