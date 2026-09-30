from storage.base import SearchResult
from retrieval.reranker import CrossEncoderReranker

def test_cross_encoder_reranker_with_custom_scoring():
    candidates = [
        SearchResult(chunk_id="c1", document_id="d1", chunk_index=0, content="Low priority content", score=0.9),
        SearchResult(chunk_id="c2", document_id="d1", chunk_index=1, content="Exact high priority target content", score=0.5),
    ]

    # Custom scoring function that prioritizes 'high priority'
    def mock_scorer(query: str, doc: str) -> float:
        return 1.0 if "high priority" in doc else 0.1

    reranker = CrossEncoderReranker(scoring_fn=mock_scorer)
    reranked = reranker.rerank(query="find target", results=candidates, top_n=2)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score == 1.0
    assert reranked[1].chunk_id == "c1"

def test_cross_encoder_reranker_heuristic_with_section_boost():
    candidates = [
        SearchResult(
            chunk_id="c_intro",
            document_id="d1",
            chunk_index=0,
            content="Patellar tendinopathy is common among volleyball athletes.",
            section="Introduction",
            score=0.7
        ),
        SearchResult(
            chunk_id="c_results",
            document_id="d1",
            chunk_index=1,
            content="Heavy slow resistance training produced a 45% reduction in patellar tendon pain on the VISA-P scale.",
            section="Results",
            score=0.7
        )
    ]

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query="patellar tendinopathy pain reduction", results=candidates, top_n=2)

    assert len(reranked) == 2
    # c_results should be boosted due to higher keyword coverage and 'Results' section weight
    assert reranked[0].chunk_id == "c_results"
    assert reranked[0].score > reranked[1].score

def test_reranker_empty_results():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("query", [], top_n=5) == []
