import time
import logging
import re
from typing import List, Optional
from pydantic import BaseModel, Field

from storage.base import SearchResult
from generation.context_assembler import ContextAssembler, CitationSource, AssembledContext
from generation.llm import LLMProvider, LLMResponse
from generation.prompt_templates import PHYSIO_EVIDENCE_SYSTEM_PROMPT, USER_EVIDENCE_PROMPT_TEMPLATE
from observability.tracing import trace_operation

logger = logging.getLogger(__name__)

class EvidenceAnswer(BaseModel):
    """
    Final evidence-grounded response object.
    """
    question: str
    answer: str
    sources: List[CitationSource] = Field(default_factory=list)
    insufficient_evidence: bool = False
    citation_verified: bool = True
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: float = 0.0

    def format_for_display(self) -> str:
        """
        Formats answer and bibliography for clean user display.
        """
        lines = [self.answer, "\n\n### Supporting Evidence Sources:"]
        if not self.sources:
            lines.append("*(No sources cited)*")
        else:
            for s in self.sources:
                lines.append(f"- **[Source {s.source_number}]** {s.title} — *{s.section}* ({s.pages})")
        return "\n".join(lines)

class EvidenceSynthesizer:
    """
    Coordinates context assembly, grounded prompt construction, LLM generation,
    and structured source citation verification.
    """

    def __init__(
        self,
        llm: LLMProvider,
        context_assembler: Optional[ContextAssembler] = None,
        system_prompt: str = PHYSIO_EVIDENCE_SYSTEM_PROMPT
    ):
        self.llm = llm
        self.context_assembler = context_assembler or ContextAssembler()
        self.system_prompt = system_prompt

    @trace_operation(name="evidence_synthesizer", span_type="chain")
    def synthesize(self, question: str, search_results: List[SearchResult]) -> EvidenceAnswer:
        start_t = time.perf_counter()

        # Handle case where retriever found zero results
        if not search_results:
            latency = (time.perf_counter() - start_t) * 1000.0
            return EvidenceAnswer(
                question=question,
                answer="Based on the retrieved research literature, there is insufficient evidence to answer this inquiry. No relevant peer-reviewed studies were identified in the corpus matching the clinical question.",
                sources=[],
                insufficient_evidence=True,
                latency_ms=round(latency, 2)
            )

        # 1. Assemble structured evidence passages
        assembled: AssembledContext = self.context_assembler.assemble(search_results)

        # 2. Construct user prompt
        user_prompt = USER_EVIDENCE_PROMPT_TEMPLATE.format(
            context=assembled.context_text,
            question=question.strip()
        )

        # 3. Generate response via LLM
        response: LLMResponse = self.llm.generate(
            prompt=user_prompt,
            system_prompt=self.system_prompt
        )

        latency = (time.perf_counter() - start_t) * 1000.0

        # A source list alone is not enough: reject answers that cite a source
        # number which was not supplied to the model, or that omit citations
        # entirely. This keeps the UI from presenting unverifiable synthesis.
        cited_numbers = {
            int(number)
            for number in re.findall(r"\[Source\s+(\d+)\]", response.content, flags=re.IGNORECASE)
        }
        available_numbers = {source.source_number for source in assembled.sources}
        citations_valid = bool(cited_numbers) and cited_numbers.issubset(available_numbers)
        is_insufficient = "insufficient evidence" in response.content.lower()
        answer_text = response.content.strip()
        if not citations_valid and not is_insufficient:
            logger.warning("LLM response failed citation validation: cited=%s available=%s", cited_numbers, available_numbers)
            answer_text = (
                "The generated response could not be verified against the retrieved sources. "
                "Please review the supporting evidence or broaden the indexed corpus."
            )
            is_insufficient = True

        logger.info(
            f"Synthesized evidence answer in {latency:.1f}ms: "
            f"{len(assembled.sources)} sources, insufficient={is_insufficient}"
        )

        return EvidenceAnswer(
            question=question,
            answer=answer_text,
            sources=assembled.sources,
            insufficient_evidence=is_insufficient,
            citation_verified=citations_valid or is_insufficient,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=round(latency, 2)
        )
