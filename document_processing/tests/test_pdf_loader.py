import os
import pytest
from document_processing.loaders.pdf_loader import PDFLoader
import fitz

@pytest.fixture
def sample_pdf_path(tmp_path):
    path = tmp_path / "test_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello World! This is a test PDF. It has enough text to pass the OCR validation threshold which requires more than 50 characters per page on average. This should definitely be enough.")
    doc.save(str(path))
    doc.close()
    return str(path)
    
@pytest.fixture
def scanned_pdf_path(tmp_path):
    # Create a PDF with no text, simulating a scanned document
    path = tmp_path / "scanned_doc.pdf"
    doc = fitz.open()
    doc.new_page() # empty page
    doc.save(str(path))
    doc.close()
    return str(path)

def test_pdf_loader_success(sample_pdf_path):
    loader = PDFLoader()
    doc = loader.load(sample_pdf_path)
    
    assert doc.filename == "test_doc.pdf"
    assert len(doc.pages) == 1
    assert "Hello World!" in doc.pages[0].text
    assert doc.pages[0].page_number == 1
    
def test_pdf_loader_file_not_found():
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent.pdf")

def test_pdf_loader_invalid_extension():
    loader = PDFLoader()
    with pytest.raises(ValueError, match="not a PDF"):
        loader.load("test.txt")
        
def test_pdf_loader_needs_ocr(scanned_pdf_path):
    loader = PDFLoader()
    with pytest.raises(ValueError, match="OCR may be required"):
        loader.load(scanned_pdf_path)
