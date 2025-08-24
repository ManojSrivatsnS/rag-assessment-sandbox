import os, json
import streamlit as st

# If you run "streamlit run ui.py" from inside option_b_rag_qa/, this import works:
from app import Retriever, generate_answer, _retrieve_pipeline

st.set_page_config(page_title="RAG Q&A — Money Classics", layout="centered")
st.title("RAG Q&A — Money Classics")

index_dir = st.text_input("Index dir", "option_b_rag_qa/rag_index")
k = st.slider("Top‑k", 1, 10, 5)
thr = st.slider("Threshold (abstain)", 0.0, 0.8, 0.25, 0.05)
use_llm = st.toggle("Use LLM (needs OPENAI_API_KEY)", value=bool(os.getenv("OPENAI_API_KEY")))
author_boost = st.toggle("Author boost", True)
mmr = st.toggle("Diversity (MMR)", True)
dedupe = st.toggle("De‑dupe citations", True)
local_rewriter = st.toggle("Local rewriter (fallback)", False)

q = st.text_input("Ask a question", "")

if q and index_dir:
    try:
        r = Retriever(index_dir)
        hits = _retrieve_pipeline(r, q, k=k, author_boost=author_boost, mmr=mmr)
        ans = generate_answer(
            q, hits, threshold=thr, llm=use_llm,
            dedupe=dedupe, local_rewriter=local_rewriter, verbose=False
        )
        st.subheader("Answer")
        st.write(ans.answer)

        with st.expander("Citations"):
            for c in ans.citations:
                st.code(json.dumps(c, indent=2))

        with st.expander("Retrieved chunks (top pool)"):
            for h in hits:
                st.write(f"**{os.path.basename(h['path'])}**  score={h['score']:.3f}  span=({h['start']}-{h['end']})")
                st.text(h["text"][:400] + ("..." if len(h["text"])>400 else ""))

    except Exception as e:
        st.error(f"Error: {e}")
