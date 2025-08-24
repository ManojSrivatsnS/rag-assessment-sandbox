#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import faiss
from pydantic import BaseModel
from rich import print as rprint
from dotenv import load_dotenv
import httpx
from typing import Optional

# ──────────────────────────── .env ────────────────────────────
env_path = Path(__file__).with_name(".env")
if env_path.exists():
    load_dotenv(env_path)

# ───────────────────── Embeddings / Models ────────────────────
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # pragma: no cover

def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    model = SentenceTransformer(model_name)
    dim = int(model.get_sentence_embedding_dimension())
    def encode(texts: List[str]) -> np.ndarray:
        arr = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return arr.astype("float32")
    return encode, dim

# ───────────────────── Cleaning & Chunking ────────────────────
_GUT_START = re.compile(r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG", re.I)
_GUT_END   = re.compile(r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG", re.I)

def clean_gutenberg(text: str) -> str:
    start = _GUT_START.search(text)
    end   = _GUT_END.search(text)
    if start and end and end.start() > start.end():
        text = text[start.end(): end.start()]
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

# ──────────────────────── Index build/load ─────────────────────
def _idx_paths(index_dir: str) -> Dict[str, Path]:
    root = Path(index_dir)
    return {"root": root, "faiss": root/"index.faiss", "meta": root/"meta.jsonl", "info": root/"embedder.json"}

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

    embs = encode(payloads)  # (N, D) normalized
    index = faiss.IndexFlatIP(dim)  # cosine via inner product
    index.add(embs)

    faiss.write_index(index, str(paths["faiss"]))
    with open(paths["meta"], "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    paths["info"].write_text(json.dumps({"dim": dim, "chunk_size": chunk_size, "overlap": overlap}), encoding="utf-8")
    rprint(f"[green]✅ Index built[/green] at {paths['root']} (chunks: {len(metas)})")

class Retriever:
    def __init__(self, index_dir: str):
        paths = _idx_paths(index_dir)
        self.index = faiss.read_index(str(paths["faiss"]))
        self.meta  = [json.loads(l) for l in open(paths["meta"], encoding="utf-8")]
        info = json.loads(paths["info"].read_text(encoding="utf-8"))
        self.dim = int(info["dim"])
        self.encode, _ = get_embedder()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = self.encode([query])
        D, I = self.index.search(q, k)
        hits: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if int(idx) < 0: continue
            m = self.meta[int(idx)]
            hits.append({"score": float(score), **m})
        return hits

# ───────────────────── Retrieval helpers ───────────────────────
AUTHOR_HINTS = {
    "barnum": "barnum_art_of_money_getting.txt",
    "wattles": "wattles_science_of_getting_rich.txt",
    "smiles":  "smiles_self_help.txt",
    "conwell": "conwell_acres_of_diamonds.txt",
}

def apply_author_boost(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q = query.lower()
    target = None
    for k, fname in AUTHOR_HINTS.items():
        if k in q:
            target = fname
            break
    if not target:
        return hits
    out = []
    for h in hits:
        bname = os.path.basename(h["path"])
        h2 = dict(h)
        if bname == target:
            h2["score"] *= 1.05  # +5%
        out.append(h2)
    return sorted(out, key=lambda x: x["score"], reverse=True)

def mmr_select(hits: List[Dict[str, Any]], k=5, lambda_=0.7) -> List[Dict[str, Any]]:
    selected = []
    candidates = hits[:20]  # consider more
    while candidates and len(selected) < k:
        if not selected:
            selected.append(candidates.pop(0))
            continue
        def mmr_score(h):
            same_file_penalty = any(os.path.basename(h["path"]) == os.path.basename(s["path"]) for s in selected)
            return lambda_*h["score"] - (0.3 if same_file_penalty else 0.0)
        candidates.sort(key=mmr_score, reverse=True)
        selected.append(candidates.pop(0))
    return selected

def dedupe_citations(hits: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for h in hits[:10]:
        key = os.path.basename(h["path"])
        if key not in seen or h["score"] > seen[key]["score"]:
            seen[key] = h
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_n]

# ───────────────────── Prompt logging ──────────────────────────
def log_prompt(role_msgs, model: str, temperature: float, top_scores: List[float]) -> None:
    try:
        with open("PROMPTLOG.md", "a", encoding="utf-8") as f:
            f.write(
                f"\n---\n### {datetime.now(timezone.utc).isoformat()}  model={model}  temp={temperature}\n"
                f"Top scores: {top_scores}\n"
                "```json\n" + json.dumps(role_msgs, ensure_ascii=False, indent=2) + "\n```\n"
            )
    except Exception:
        pass

# ───────────────────────── Answering ───────────────────────────
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

_SUMMARY_PIPE = None
def local_summarize(text: str) -> str:
    """
    Optional local rewriter:
    - Tries a small public summarization model (no HF token required).
    - If unavailable (no net / blocked), uses a simple extractive 2-sentence fallback.
    """
    # Very safe extractive fallback (no network/deps)
    def _extractive_fallback(t: str, max_chars: int = 400) -> str:
        # Take first 2 sentences-ish
        parts = re.split(r'(?<=[.!?])\s+', t.strip())
        head = " ".join(parts[:2]) or t[:200]
        return head[:max_chars]

    global _SUMMARY_PIPE
    try:
        # Import transformers only when needed
        from transformers import pipeline
    except Exception:
        return _extractive_fallback(text)

    # Try to create the pipeline (wrap in try to catch 401/offline)
    if _SUMMARY_PIPE is None:
        try:
            # smaller public model; widely available
            _SUMMARY_PIPE = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            return _extractive_fallback(text)

    try:
        out = _SUMMARY_PIPE(text[:1200], max_length=90, min_length=30, do_sample=False)
        return out[0]["summary_text"]
    except Exception:
        return _extractive_fallback(text)

def generate_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    threshold: float = 0.25,
    llm: bool = True,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 350,
    dedupe: bool = True,
    local_rewriter: bool = False,
    verbose: bool = False,
) -> Answer:
    # 1) Retrieval-level abstention (no/weak hits)
    top = chunks[0]["score"] if chunks else 0.0
    if (not chunks) or (top < threshold):
        return Answer(answer="I don't know.", citations=[], abstained=True)

    # 2) Prep citations (dedupe + optional weak-cutoff)
    cit_pool = dedupe_citations(chunks, top_n=3) if dedupe else chunks[:3]
    if cit_pool:
        top_sc = cit_pool[0]["score"]
        cutoff = 0.20 * top_sc      # keep citations within 20% of best score
        cit_pool = [h for h in cit_pool if h["score"] >= cutoff]

    # 3) Build prompt
    api_key = os.getenv("OPENAI_API_KEY")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    msgs = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",
         "content": f"Question: {question}\n\nSources:\n{_fmt_chunks(cit_pool)}\n\n"
                    f"Answer clearly with citations like [1], [2]."},
    ]
    top_scores = [round(h["score"], 3) for h in cit_pool]

    # 4) Produce text (LLM or local fallback)
    if llm and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, http_client=httpx.Client(timeout=30.0))
            if verbose: print({"use_llm": True, "model": model, "top_score": round(top, 3)})
            log_prompt(msgs, model, temperature, top_scores)
            resp = client.chat.completions.create(
                model=model, messages=msgs, temperature=float(temperature), max_tokens=int(max_tokens)
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print("LLM_ERROR:", e)
            text = (local_summarize(chunks[0]["text"]) if local_rewriter else chunks[0]["text"][:280]) + " [1]"
    else:
        if verbose: print({"use_llm": False, "top_score": round(top, 3)})
        text = (local_summarize(chunks[0]["text"]) if local_rewriter else chunks[0]["text"][:280]) + " [1]"

    # 5) Robust LLM-level abstention detection (handles quotes/citations/punct)
    raw = (text or "").strip()
    norm = raw.replace("’", "'").replace("‘", "'").replace("`", "'")
    norm = re.sub(r"\[\d+\]", "", norm)       # drop [1], [2], …
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    norm_nopunct = re.sub(r"[^\w\s]", "", norm)

    abstain_phrases = {
        "i don't know", "i dont know", "idk", "cannot answer", "no answer", "i do not know"
    }
    if (norm_nopunct in abstain_phrases) or norm.startswith("i don't know") or norm.startswith("i dont know"):
        # If the final text is an abstention, do not show citations.
        return Answer(answer="I don't know.", citations=[], abstained=True)

    # 6) Emit answer with citations
    cits = [{
        "label": f"[{i}]",
        "path": h["path"],
        "span": [h["start"], h["end"]],
        "score": h["score"],
    } for i, h in enumerate(cit_pool, 1)]

    return Answer(answer=raw, citations=cits, abstained=False)


# ────────────────────────── Caching ────────────────────────────
_ANS_CACHE: Dict[Tuple[str, str, int, float, bool, str], Answer] = {}

def index_fingerprint(index_dir: str) -> str:
    m = hashlib.md5()
    for name in ["index.faiss", "meta.jsonl", "embedder.json"]:
        p = Path(index_dir) / name
        if p.exists():
            m.update(p.read_bytes())
    return m.hexdigest()

# ─────────────────────── Evaluation / Sweep ────────────────────
def _retrieve_pipeline(r: Retriever, q: str, k: int, author_boost: bool, mmr: bool) -> List[Dict[str, Any]]:
    # pull a larger pool for MMR/diversity
    pool_k = max(k * 4, 10)
    hits = r.search(q, k=pool_k)
    if author_boost:
        hits = apply_author_boost(q, hits)
    if mmr:
        hits = mmr_select(hits, k=k)
    else:
        hits = hits[:k]
    return hits

def run_eval(index: str, qas_path: str, k: int, threshold: float,
             llm: bool = True, model: Optional[str] = None,
             temp: float = 0.0, max_tokens: int = 350,
             author_boost: bool = True, mmr: bool = True,
             dedupe: bool = True, local_rewriter: bool = False) -> Dict[str, float]:
    r = Retriever(index)
    lines = [ln for ln in open(qas_path, encoding="utf-8").read().splitlines() if ln.strip()]
    try:
        if len(lines) == 1 and lines[0].strip().startswith("["):
            cases = json.loads(lines[0])
        else:
            cases = [json.loads(l) for l in lines]
    except Exception:
        cases = [json.loads(l) for l in lines]

    tot = len(cases)
    doc_hit = got_ref = 0
    abstain_tp = abstain_fp = abstain_fn = 0
    base = lambda p: os.path.basename(p)

    for ex in cases:
        q = ex["question"]
        expect_doc: Optional[str] = ex.get("expect_doc")
        ood = bool(ex.get("ood", False))

        hits = _retrieve_pipeline(r, q, k=k, author_boost=author_boost, mmr=mmr)
        ans = generate_answer(q, hits, threshold=threshold, llm=llm, model=model,
                              temperature=temp, max_tokens=max_tokens, dedupe=dedupe,
                              local_rewriter=local_rewriter, verbose=False)

        if expect_doc:
            pool = {base(h["path"]) for h in hits}
            if expect_doc in pool: doc_hit += 1
            if any(base(c["path"]) == expect_doc for c in ans.citations): got_ref += 1

        if ood:
            if ans.abstained: abstain_tp += 1
            else:             abstain_fn += 1
        else:
            if ans.abstained: abstain_fp += 1

    denom_in = max(1, sum(1 for ex in cases if ex.get("expect_doc")))
    results = {
        "n": float(tot),
        "retrieval_doc_hit@k": doc_hit / denom_in,
        "citation_matches_ref": got_ref / denom_in,
        "abstain_precision": (abstain_tp / max(1, abstain_tp + abstain_fp)),
        "abstain_recall": (abstain_tp / max(1, abstain_tp + abstain_fn)),
    }
    rprint({"eval": results})
    Path("eval_report.md").write_text("\n".join([f"- {k}: {v:.3f}" for k, v in results.items()]), encoding="utf-8")
    return results

def sweep_thresholds(index: str, qas: str, k: int, thr_list: List[float], **kwargs) -> None:
    rows = []
    for th in thr_list:
        res = run_eval(index, qas, k, th, **kwargs)
        rows.append({"threshold": th, **res})
    Path("eval_threshold_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = [
        "| thr | hit@k | cite@ref | abst.P | abst.R |",
        "|-----|-------|----------|--------|--------|",
    ]
    for r in rows:
        md.append(f"| {r['threshold']:.2f} | {r['retrieval_doc_hit@k']:.2f} | {r['citation_matches_ref']:.2f} | {r['abstain_precision']:.2f} | {r['abstain_recall']:.2f} |")
    Path("eval_threshold_sweep.md").write_text("\n".join(md), encoding="utf-8")
    rprint("[cyan]Wrote eval_threshold_sweep.md[/cyan]")

# ──────────────────────────── CLI ──────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="RAG Q&A (Option B)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ingest
    ap_ing = sub.add_parser("ingest", help="Build FAISS index from corpus")
    ap_ing.add_argument("--corpus", required=True)
    ap_ing.add_argument("--index", required=True)
    ap_ing.add_argument("--chunk_size", type=int, default=750)
    ap_ing.add_argument("--overlap", type=int, default=150)

    # ask
    ap_ask = sub.add_parser("ask", help="Ask a question")
    ap_ask.add_argument("question", type=str)
    ap_ask.add_argument("--index", required=True)
    ap_ask.add_argument("--k", type=int, default=5)
    ap_ask.add_argument("--threshold", type=float, default=0.25)
    ap_ask.add_argument("--llm", choices=["on", "off"], default=None)
    ap_ask.add_argument("--model", type=str, default=None)
    ap_ask.add_argument("--temp", type=float, default=0.0)
    ap_ask.add_argument("--max_tokens", type=int, default=350)
    ap_ask.add_argument("--author_boost", choices=["on", "off"], default="on")
    ap_ask.add_argument("--mmr", choices=["on", "off"], default="on")
    ap_ask.add_argument("--dedupe", choices=["on", "off"], default="on")
    ap_ask.add_argument("--local_rewriter", choices=["on", "off"], default="off")
    ap_ask.add_argument("--verbose", action="store_true")

    # eval
    ap_eval = sub.add_parser("eval", help="Run mini evaluation from questions.json (JSONL or JSON array)")
    ap_eval.add_argument("--index", required=True)
    ap_eval.add_argument("--qas", required=True)
    ap_eval.add_argument("--k", type=int, default=5)
    ap_eval.add_argument("--threshold", type=float, default=0.25)
    ap_eval.add_argument("--llm", choices=["on", "off"], default=None)
    ap_eval.add_argument("--model", type=str, default=None)
    ap_eval.add_argument("--temp", type=float, default=0.0)
    ap_eval.add_argument("--max_tokens", type=int, default=350)
    ap_eval.add_argument("--author_boost", choices=["on", "off"], default="on")
    ap_eval.add_argument("--mmr", choices=["on", "off"], default="on")
    ap_eval.add_argument("--dedupe", choices=["on", "off"], default="on")
    ap_eval.add_argument("--local_rewriter", choices=["on", "off"], default="off")
    ap_eval.add_argument("--sweep", type=str, default=None, help="Comma-separated thresholds, e.g., 0.15,0.2,0.25")

    args = ap.parse_args()

    if args.cmd == "ingest":
        build_index(args.corpus, args.index, args.chunk_size, args.overlap)
        return

    if args.cmd == "ask":
        r = Retriever(args.index)
        # Decide LLM usage
        use_llm = (args.llm == "on") if args.llm is not None else (os.getenv("OPENAI_API_KEY") is not None)
        # Retrieve with optional boosts & MMR
        hits = _retrieve_pipeline(
            r, args.question, k=args.k,
            author_boost=(args.author_boost == "on"),
            mmr=(args.mmr == "on")
        )
        # Optional cache (in-memory)
        fp = index_fingerprint(args.index)
        cache_key = (fp, args.question, args.k, round(args.threshold,3), use_llm, str(args.model or os.getenv("OPENAI_MODEL","")))
        if cache_key in _ANS_CACHE:
            ans = _ANS_CACHE[cache_key]
        else:
            ans = generate_answer(
                args.question, hits, threshold=args.threshold, llm=use_llm,
                model=args.model, temperature=args.temp, max_tokens=args.max_tokens,
                dedupe=(args.dedupe == "on"),
                local_rewriter=(args.local_rewriter == "on"),
                verbose=args.verbose,
            )
            _ANS_CACHE[cache_key] = ans
        print(json.dumps(ans.model_dump(), ensure_ascii=False, indent=2))
        return

    if args.cmd == "eval":
        # Accept JSON array or JSONL
        qas_path = Path(args.qas)
        text = qas_path.read_text(encoding="utf-8")
        if text.strip().startswith("["):
            data = json.loads(text)
            tmp = qas_path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            qas_path = tmp

        use_llm = (args.llm == "on") if args.llm is not None else (os.getenv("OPENAI_API_KEY") is not None)

        if args.sweep:
            # parse comma-separated list
            thr_list = []
            for tok in args.sweep.split(","):
                try:
                    thr_list.append(float(tok.strip()))
                except Exception:
                    pass
            sweep_thresholds(
                args.index, str(qas_path), args.k, thr_list,
                llm=use_llm, model=args.model, temp=args.temp, max_tokens=args.max_tokens,
                author_boost=(args.author_boost == "on"),
                mmr=(args.mmr == "on"),
                dedupe=(args.dedupe == "on"),
                local_rewriter=(args.local_rewriter == "on"),
            )
        else:
            run_eval(
                args.index, str(qas_path), args.k, args.threshold,
                llm=use_llm, model=args.model, temp=args.temp, max_tokens=args.max_tokens,
                author_boost=(args.author_boost == "on"),
                mmr=(args.mmr == "on"),
                dedupe=(args.dedupe == "on"),
                local_rewriter=(args.local_rewriter == "on"),
            )
        return

if __name__ == "__main__":
    main()