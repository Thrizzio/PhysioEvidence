import pytest
from document_processing.processors.section_detector import SectionDetector
from document_processing.models.document import ExtractedDocument, Page, DocumentMetadata

@pytest.fixture
def sample_document():
    text = (
        "Physiotherapy for back pain.\n\n"
        "Abstract\nThis is the summary.\n\n"
        "1. Introduction\nBack pain is common.\n\n"
        "Materials and Methods\nWe did a study.\n\n"
        "Results\nIt worked.\n\n"
        "Conclusion\nThe end."
    )
    return ExtractedDocument(
        document_id="123",
        source_path="test.pdf",
        filename="test.pdf",
        metadata=DocumentMetadata(),
        pages=[Page(page_number=1, text=text)]
    )

def test_section_detector(sample_document):
    detector = SectionDetector()
    sections = detector.detect_sections(sample_document)
    
    assert len(sections) == 6
    assert sections[0].name == "Preamble"
    assert sections[1].name == "Abstract"
    assert sections[2].name == "Introduction"
    assert sections[3].name == "Methods"  # Alias test
    assert sections[4].name == "Results"
    assert sections[5].name == "Conclusion"
    
    assert "We did a study" in sections[3].content

