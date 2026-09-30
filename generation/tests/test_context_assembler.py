from storage.base import SearchResult
from generation.context_assembler import ContextAssembler

def test_context_assembler_empty():
    assembler = ContextAssembler()
    context = assembler.assemble([])
    assert context.total_chunks == 0
    assert "No relevant research literature" in context.context_text
    assert context.sources == []

def test_context_assembler_formatting():
    assembler = ContextAssembler()
    results = [
        SearchResult(
            chunk_id="c1",
            document_id="d1",
            chunk_index=0,
            content="Isometric quadriceps contractions reduce anterior knee pain.",
            section="Results",
            page_start=4,
            page_end=4,
            document_title="Patellofemoral Pain Management",
            score=0.92
        ),
        SearchResult(
            chunk_id="c2",
            document_id="d2",
            chunk_index=1,
            content="Gradual return to running over 6 weeks minimizes patellar tendon overload.",
            section="Discussion",
            page_start=12,
            page_end=14,
            document_title="Running Injuries",
            score=0.85
        )
    ]

    context = assembler.assemble(results)
    assert context.total_chunks == 2
    assert len(context.sources) == 2

    # Verify source numbers and metadata
    assert context.sources[0].source_number == 1
    assert context.sources[0].title == "Patellofemoral Pain Management"
    assert context.sources[0].pages == "p. 4"

    assert context.sources[1].source_number == 2
    assert context.sources[1].title == "Running Injuries"
    assert context.sources[1].pages == "pp. 12-14"

    # Verify formatted text block contains headers and content
    assert "[Source 1]" in context.context_text
    assert "[Source 2]" in context.context_text
    assert "Isometric quadriceps contractions" in context.context_text

def test_context_assembler_character_limit():
    assembler = ContextAssembler(max_characters=150)
    results = [
        SearchResult(chunk_id=f"c_{i}", document_id=f"d_{i}", chunk_index=0, content="A" * 100, score=0.9)
        for i in range(5)
    ]
    context = assembler.assemble(results)
    # Only 1 chunk fits into 150 chars
    assert context.total_chunks == 1
