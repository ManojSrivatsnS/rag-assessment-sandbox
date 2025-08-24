#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import faiss
from pydantic import BaseModel
from rich import print as rprint
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
env_path = Path(__file__).with_name(".env")
if env_path.exists():
    load_dotenv(env_path)

# ------------------------ Embeddings ------------------------
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # pragma: no cover

def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns (encode_fn, dim). encode_fn(texts)->np.ndarray (N,D), L2-normalized.
    """
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    model = SentenceTransformer(model_name)
    dim = int(model.get_sentence_embedding_dimension())
    def encode(texts: List[str]) -> np.ndarray:
        arr = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return arr.astype("float32")
    return encode, dim

# ------------------------ Cleaning & Chunking ------------------------
_GUT_START = re.compile(r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG", re.I)
_GUT_END   = re.compile(r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG", re.I)

def clean_gutenberg(text: str) -> str:
    """Remove Gutenberg boilerplate if markers are present."""
    start = _GUT_START.search(text)
    end   = _GUT_END.search(text)
    if start and end and end.start() > start.end():
        text = text[start.end(): end.start()]
    # normalize whitespace
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, chunk_size: int = 750, overlap: int = 150) -> List[Tuple[int,int,str]]:
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(i + chunk_size, n)
        out.append((i, j, text[i:j]))
        i = max(j - overlap, i + 1)
    return out

def iter_docs(corpus_dir: Path) -> Iterable[Dict[str, str]]:
    for p in corpus_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            yield {"path": str(p), "text": clean_gutenberg(txt)}

# ------------------------ Index build/load ------------------------
def _idx_paths(index_dir: str) -> Dict[str, Path]:
    root = Path(index_dir)
    return {
        "root": root,
        "faiss": root / "index.faiss",
        "meta": root / "meta.jsonl",
        "info": root / "embedder.json",
    }

def build_index(corpus_dir: str, index_dir: str, chunk_size: int = 750, overlap: int = 150) -> None:
    corpus = Path(corpus_dir)
    paths = _idx_paths(index_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)

    encode, dim = get_embedder()

    metas: List[Dict[str, Any]] = []
    payloads: List[str] = []

    for doc in iter_docs(corpus):
        for (s, e, ch) in chunk_text(doc["text"], chunk_size, overlap):
            metas.append({"path": doc["path"], "start": s, "end": e, "text": ch})
            payloads.append(ch)

    if not payloads:
        raise RuntimeError(f"No .txt/.md files found under {corpus}")

    embs = encode(payloads)  # (N, D), already normalized
    index = faiss.IndexFlatIP(dim)     # cosine via inner product on normalized vectors
    index.add(embs)

    faiss.write_index(index, str(paths["faiss"]))
    with open(paths["meta"], "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    paths["info"].write_text(
        json.dumps({"dim": dim, "chunk_size": chunk_size, "overlap": overlap}),
        encoding="utf-8"
    )
    rprint(f"[green]✅ Index built[/green] at {paths['root']}  (chunks: {len(metas)})")

class Retriever:
    def __init__(self, index_dir: str):
        paths = _idx_paths(index_dir)
        self.index = faiss.read_index(str(paths["faiss"]))
        self.meta  = [json.loads(l) for l in open(paths["meta"], encoding="utf-8")]
        info = json.loads(paths["info"].read_text(encoding="utf-8"))
        self.dim = int(info["dim"])
        self.encode, _ = get_embedder()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = self.encode([query])  # (1, D)
        D, I = self.index.search(q, k)
        hits: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if int(idx) < 0:
                continue
            m = self.meta[int(idx)]
            hits.append({"score": float(score), **m})
        return hits

# ------------------------ Answering ------------------------
class Answer(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    abstained: bool = False

_SYSTEM = ("You are a careful assistant. Only answer using the provided sources. "
           "If the sources are insufficient, say exactly: 'I don't know.' "
           "Be concise, include short quotes, and cite sources like [1], [2].")

def _fmt_chunks(chs: List[Dict[str, Any]]) -> str:
    lines = []
    for i, ch in enumerate(chs, 1):
        fname = os.path.basename(ch["path"])
        lines.append(f"[{i}] ({fname}:{ch['start']}-{ch['end']}) {ch['text']}")
    return "\n\n".join(lines)

def generate_answer(question: str, chunks: List[Dict[str, Any]], threshold: float = 0.25) -> Answer:
    top = chunks[0]["score"] if chunks else 0.0
    if (not chunks) or (top < threshold):
        return Answer(answer="I don't know.", citations=[], abstained=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI()
            msgs = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",
                 "content": f"Question: {question}\n\nSources:\n{_fmt_chunks(chunks)}\n\n"
                            f"Answer clearly with citations like [1], [2]."},
            ]
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=msgs, temperature=0
            )
            text = resp.choices[0].message.content.strip()
        except Exception:
            text = chunks[0]["text"][:280] + " [1]"
    else:
        text = chunks[0]["text"][:280] + " [1]"

    cits = [{"label": f"[{i}]",
             "path": h["path"],
             "span": [h["start"], h["end"]],
             "score": h["score"]}
            for i, h in enumerate(chunks[:3], 1)]
    return Answer(answer=text, citations=cits, abstained=False)

# ------------------------ Evaluation ------------------------
def run_eval(index: str, qas_path: str, k: int, threshold: float) -> Dict[str, float]:
    r = Retriever(index)
    cases = [json.loads(l) for l in open(qas_path, encoding="utf-8").read().splitlines() if l.strip()]

    tot = len(cases)
    doc_hit = got_ref = 0
    abstain_tp = abstain_fp = abstain_fn = 0

    def _base(p: str) -> str: return os.path.basename(p)

    for ex in cases:
        q = ex["question"]
        expect_doc: Optional[str] = ex.get("expect_doc")
        ood = bool(ex.get("ood", False))

        hits = r.search(q, k=k)
        ans = generate_answer(q, hits, threshold=threshold)

        if expect_doc:
            pool = {_base(h["path"]) for h in hits}
            if expect_doc in pool:
                doc_hit += 1
            if any(_base(c["path"]) == expect_doc for c in ans.citations):
                got_ref += 1

        if ood:
            if ans.abstained: abstain_tp += 1
            else:             abstain_fn += 1
        else:
            if ans.abstained: abstain_fp += 1

    denom_in  = max(1, sum(1 for ex in cases if ex.get("expect_doc")))
    results = {
        "n": float(tot),
        "retrieval_doc_hit@k": doc_hit / denom_in,
        "citation_matches_ref": got_ref / denom_in,
        "abstain_precision": (abstain_tp / max(1, abstain_tp + abstain_fp)),
        "abstain_recall": (abstain_tp / max(1, abstain_tp + abstain_fn)),
    }
    rprint({"eval": results})
    Path("eval_report.md").write_text(
        "\n".join([f"- {k}: {v:.3f}" for k, v in results.items()]),
        encoding="utf-8"
    )
    return results

# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser(description="RAG Q&A (Option B)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_ing = sub.add_parser("ingest", help="Build FAISS index from corpus")
    ap_ing.add_argument("--corpus", required=True)
    ap_ing.add_argument("--index", required=True)
    ap_ing.add_argument("--chunk_size", type=int, default=750)
    ap_ing.add_argument("--overlap", type=int, default=150)

    ap_ask = sub.add_parser("ask", help="Ask a question")
    ap_ask.add_argument("question", type=str)
    ap_ask.add_argument("--index", required=True)
    ap_ask.add_argument("--k", type=int, default=5)
    ap_ask.add_argument("--threshold", type=float, default=0.25)

    ap_eval = sub.add_parser("eval", help="Run mini evaluation from questions.json (JSON lines or array)")
    ap_eval.add_argument("--index", required=True)
    ap_eval.add_argument("--qas", required=True)
    ap_eval.add_argument("--k", type=int, default=5)
    ap_eval.add_argument("--threshold", type=float, default=0.25)

    args = ap.parse_args()

    if args.cmd == "ingest":
        build_index(args.corpus, args.index, args.chunk_size, args.overlap)
    elif args.cmd == "ask":
        r = Retriever(args.index)
        hits = r.search(args.question, k=args.k)
        ans = generate_answer(args.question, hits, threshold=args.threshold)
        print(json.dumps(ans.model_dump(), ensure_ascii=False, indent=2))
    elif args.cmd == "eval":
        # Accept either JSONL or a JSON array file for convenience
        qas_path = Path(args.qas)
        text = qas_path.read_text(encoding="utf-8")
        if text.strip().startswith("["):
            data = json.loads(text)
            tmp = qas_path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            args.qas = str(tmp)
        run_eval(args.index, args.qas, args.k, args.threshold)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
