import re

def normalize_whitespace(text: str) -> str:
    """Replaces multiple spaces or newlines with appropriate spacing."""
    # Convert carriage returns to newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Replace 3 or more newlines with just 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def repair_hyphenation(text: str) -> str:
    """Repairs words that were hyphenated across line breaks."""
    # Matches a letter, a hyphen, a newline, and another letter
    # E.g. "treat-\nment" -> "treatment"
    return re.sub(r'([a-zA-Z])-\n([a-zA-Z])', r'\1\2', text)

def clean_text(text: str) -> str:
    """
    Applies standard cleaning operations to extracted text.
    Does NOT aggressively alter scientific meaning.
    """
    if not text:
        return ""
    
    text = repair_hyphenation(text)
    text = normalize_whitespace(text)
    return text

