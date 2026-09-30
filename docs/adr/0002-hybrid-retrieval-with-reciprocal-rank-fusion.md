# ADR 0002: Hybrid Retrieval Architecture and Reciprocal Rank Fusion (RRF)

## Context
In physiotherapy research literature, users search with both high-level semantic intent (e.g. "early weightbearing safety following ACL graft reconstruction") and exact clinical terminology, specialized acronyms, or specific anatomical tests (e.g. "Lachman test", "VAS pain score", "BFR", "Nordic hamstring").
Pure semantic vector search excels at conceptual matching and paraphrasing, but can fail on specialized clinical acronyms or exact author names. Conversely, keyword full-text search (BM25 / PostgreSQL `tsvector`) excels at exact phrase matching, but fails with synonyms ("eccentric training" vs "negative repetitions").

## Decision
We implement a hybrid retrieval system combining dense semantic vector search and sparse full-text lexical search using **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF Score}(d) = \sum_{m \in \{\text{vector}, \text{text}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

Where:
- $k$ is a smoothing constant (standard default $k = 60$).
- $w_m$ is an optional channel weight (default $w_{\text{vector}} = 1.0$, $w_{\text{text}} = 1.0$).
- $\text{rank}_m(d)$ is the 1-based rank position of chunk $d$ in the respective retrieval candidate list.

### Why RRF over Score Normalization (Min-Max / Z-Score)
1. **Distribution Independence**: Dense vector search yields cosine similarities bounded in $[-1, 1]$ (or $[0, 1]$), while sparse full-text search yields unbounded TF-IDF / BM25 / `ts_rank_cd` scores. Linear combination of uncalibrated scores requires arbitrary tuning and breaks across disparate query distributions.
2. **Robustness to Outliers**: Rank-based fusion prevents an abnormally high keyword score or vector match from dominating the results list.
3. **Interview Defensibility**: RRF is an established, industry-standard consensus algorithm for search result fusion.

## Alternatives Considered
1. **Pure Dense Vector Search**:
   - *Limitation*: Can miss exact anatomical terms, drug names, specific surgical acronyms (e.g., "BFR", "ACL", "PTTD").
2. **Linear Score Blending ($\alpha \cdot \text{Score}_{\text{dense}} + (1-\alpha) \cdot \text{Score}_{\text{sparse}}$)**:
   - *Limitation*: Requires normalizing scores across disparate non-linear distributions, causing instability when one score distribution is skewed.

## Consequences
- **Positive**: Queries containing both clinical concepts and exact medical terms achieve superior retrieval recall.
- **Positive**: Seamlessly runs on top of `PostgresVectorStore` (which already possesses both `hnsw` vector index and `gin(tsv)` full-text index) and `InMemoryVectorStore`.
- **Considerations**: Requires two query executions per search (one dense vector embedding + search, one full-text query) followed by an in-memory fusion pass (sub-millisecond overhead).
