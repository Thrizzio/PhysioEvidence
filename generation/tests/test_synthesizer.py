from storage.base import SearchResult
from generation.llm import MockLLMProvider
from generation.synthesizer import EvidenceSynthesizer

def test_synthesizer_success():
    mock_llm = MockLLMProvider(
        response_generator=lambda prompt: "Quadriceps strength symmetry index > 90% is required before sport clearance [Source 1]."
    )
    synthesizer = EvidenceSynthesizer(llm=mock_llm)

    results = [
        SearchResult(
            chunk_id="c1",
            document_id="d1",
            chunk_index=0,
            content="Limb symmetry index > 90% in quadriceps strength is recommended.",
            section="Results",
            document_title="ACL Return to Sport",
            score=0.95
        )
    ]

    answer = synthesizer.synthesize("What symmetry index is needed?", search_results=results)

    assert answer.insufficient_evidence is False
    assert "[Source 1]" in answer.answer
    assert len(answer.sources) == 1
    assert answer.sources[0].title == "ACL Return to Sport"
    assert answer.latency_ms > 0

    display_str = answer.format_for_display()
    assert "### Supporting Evidence Sources:" in display_str
    assert "ACL Return to Sport" in display_str

def test_synthesizer_zero_search_results():
    mock_llm = MockLLMProvider()
    synthesizer = EvidenceSynthesizer(llm=mock_llm)

    answer = synthesizer.synthesize("Obscure question with no hits", search_results=[])
    assert answer.insufficient_evidence is True
    assert "insufficient evidence" in answer.answer.lower()
    assert len(answer.sources) == 0

def test_synthesizer_llm_reports_insufficient_evidence():
    mock_llm = MockLLMProvider(
        response_generator=lambda prompt: "Based on the retrieved research literature, there is insufficient evidence to determine optimal dosage for this modality."
    )
    synthesizer = EvidenceSynthesizer(llm=mock_llm)

    results = [
        SearchResult(chunk_id="c1", document_id="d1", chunk_index=0, content="General discussion.", score=0.5)
    ]
    answer = synthesizer.synthesize("What is the exact dosage?", search_results=results)
    assert answer.insufficient_evidence is True
