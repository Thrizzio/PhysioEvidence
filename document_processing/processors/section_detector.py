import re
from typing import List, Dict, Optional
from document_processing.models.document import ExtractedDocument, Section
from document_processing.processors.cleaner import clean_text

class SectionDetector:
    """
    Detects sections in research papers using aliases and heuristics.
    """
    
    # Common aliases for standard sections
    SECTION_ALIASES = {
        "Abstract": ["abstract", "summary", "background and objective"],
        "Introduction": ["introduction", "background"],
        "Methods": ["methods", "materials and methods", "methodology", "study design", "experimental design"],
        "Results": ["results", "findings"],
        "Discussion": ["discussion", "limitations"],
        "Conclusion": ["conclusion", "conclusions", "summary and conclusions"],
        "References": ["references", "literature cited", "bibliography"]
    }
    
    def __init__(self):
        # Build a regex for detecting any of these aliases at the start of a line
        all_aliases = []
        for aliases in self.SECTION_ALIASES.values():
            all_aliases.extend(aliases)
            
        # Match pattern: Start of line, optional numbering (e.g. "1.", "II."), the alias, optional colon, end of line or space
        alias_pattern = r"(?:^|\n)(?:\d+\.?\s*|[IVX]+\.?\s*)?(" + "|".join(re.escape(a) for a in all_aliases) + r")\s*:?(?:\n|$)"
        self.heading_regex = re.compile(alias_pattern, re.IGNORECASE)

    def _map_to_standard_name(self, heading: str) -> str:
        """Map an alias back to the standard section name."""
        heading_clean = heading.strip().lower()
        for std_name, aliases in self.SECTION_ALIASES.items():
            if heading_clean in aliases:
                return std_name
        return heading.strip().title()

    def detect_sections(self, document: ExtractedDocument) -> List[Section]:
        """
        Splits a document into sections.
        """
        raw_text = document.raw_text
        
        # Find all heading matches
        matches = list(self.heading_regex.finditer(raw_text))
        
        sections = []
        
        if not matches:
            # Fallback: treat whole document as one section
            sections.append(Section(
                name="Body",
                content=clean_text(raw_text)
            ))
            return sections
            
        # Add content before first section as "Preamble" or "Title/Authors"
        first_match_start = matches[0].start()
        if first_match_start > 0:
            preamble = raw_text[:first_match_start].strip()
            if preamble:
                sections.append(Section(
                    name="Preamble",
                    content=clean_text(preamble)
                ))
                
        # Extract sections based on match boundaries
        for i, match in enumerate(matches):
            raw_heading = match.group(1)
            std_name = self._map_to_standard_name(raw_heading)
            
            start_idx = match.end()
            if i + 1 < len(matches):
                end_idx = matches[i+1].start()
            else:
                end_idx = len(raw_text)
                
            content = raw_text[start_idx:end_idx].strip()
            
            if content:
                sections.append(Section(
                    name=std_name,
                    content=clean_text(content)
                ))
                
        return sections

