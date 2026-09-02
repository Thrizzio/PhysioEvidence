from document_processing.processors.cleaner import clean_text

def test_clean_whitespace():
    raw = "This   is  a\n\n\n\ntest.\r\nNext line."
    expected = "This is a\n\ntest.\nNext line."
    assert clean_text(raw) == expected

def test_repair_hyphenation():
    raw = "The treat-\nment was effective."
    expected = "The treatment was effective."
    assert clean_text(raw) == expected

