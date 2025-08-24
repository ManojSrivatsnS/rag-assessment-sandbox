import json, os
from option_b_rag_qa.app import Retriever, generate_answer

INDEX = "option_b_rag_qa/rag_index"

def test_retrieval_returns_hits():
    r = Retriever(INDEX)
    hits = r.search("Barnum on advertising", k=3)
    assert len(hits) > 0
    assert all("text" in h and "score" in h for h in hits)

def test_abstention_on_ood_low_threshold():
    r = Retriever(INDEX)
    hits = r.search("Who won the Battle of Hastings?", k=5)
    ans = generate_answer("Who won the Battle of Hastings?", hits, threshold=0.5, llm=False)
    assert ans.abstained is True
    assert ans.answer.lower().startswith("i don't know")

def test_in_domain_with_citations():
    r = Retriever(INDEX)
    hits = r.search("What does Barnum advise about frugality?", k=5)
    ans = generate_answer("What does Barnum advise about frugality?", hits, threshold=0.25, llm=False)
    assert ans.abstained is False
    assert len(ans.citations) >= 1
