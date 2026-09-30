from abc import ABC, abstractmethod
from typing import List, Optional
from storage.base import SearchResult, FilterCriteria

class BaseRetriever(ABC):
    """
    Abstract interface for evidence retrieval components.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_criteria: Optional[FilterCriteria] = None
    ) -> List[SearchResult]:
        """
        Retrieves top_k most relevant chunks matching the query and filter criteria.
        """
        pass
