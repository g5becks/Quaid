# 02 - Architecture and Design

**Comprehensive system design leveraging Python ecosystem for intelligent knowledge management**

---

## Executive Summary

Quaid is implemented as a pure Python application leveraging a powerful combination of libraries for structured markdown processing, high-performance search, natural language understanding, and optional AI enhancement. The architecture achieves vector-database-level search quality through a unique combination of:

1. **Markdown-Query**: Structured markdown parsing and semantic understanding
2. **Tantivy**: Full-text search with git-storable indexes
3. **Polars**: High-performance dataframe operations for metadata
4. **SpaCy**: Natural language processing and entity recognition
5. **Optional AI**: Local classification models or cloud providers for enhanced capabilities

**Core Innovation**: Achieve enterprise-grade search quality through a FastMCP-based server that enables efficient multi-agent coordination, code-based tool composition, and intelligent memory management while keeping all data human-readable and version-controlled.

**Framework Choice**: FastMCP provides comprehensive MCP server capabilities with LLM-friendly documentation, async support, and production-ready features that align perfectly with Quaid's requirements.

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Quaid MCP Server (Python)                │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           MCP Server Layer (FastMCP)                │   │
│  │  - Tool registration and discovery                  │   │
│  │  - Agent communication and context                  │   │
│  │  - Code execution environment                      │   │
│  │  - Skill management and persistence                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Code Execution Framework                 │   │
│  │                                                     │   │
│  │  • Progressive tool discovery                       │   │
│  │  • Context-efficient data processing                │   │
│  │  • Privacy-preserving operations                    │   │
│  │  • State persistence across executions              │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Query Processing Pipeline                │   │
│  │                                                     │   │
│  │  1. Intent Analysis (spaCy)                        │   │
│  │  2. Full-Text Search (tantivy)                     │   │
│  │  3. Structural Query (markdown-query)              │   │
│  │  4. Metadata Filter (polars)                       │   │
│  │  5. Reranking (spaCy similarity + optional AI)     │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Storage Layer (Git-Native)             │   │
│  │                                                     │   │
│  │  • Markdown fragments (.md files)                  │   │
│  │  • JSONL indexes (polars-compatible)               │   │
│  │  • Tantivy search index (binary, git-storable)     │   │
│  │  • spaCy custom NER models                         │   │
│  │  • Agent skills and code workspace                 │   │
│  │  • TOML configuration                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Core Components Deep Dive

## 1. Markdown-Query: Structural Understanding

**Purpose**: Parse and query markdown structure for rich semantic understanding

**Key Capabilities**:

```python
import mq

# Extract all code blocks
code_blocks = mq.run("select(.code) | to_text()", fragment_content, None)

# Get headings hierarchy
headings = mq.run("select(or(.h1, .h2, .h3))", fragment_content, None)

# Extract decision blocks (using blockquotes as conventions)
decisions = mq.run("select(.blockquote) | to_text()", fragment_content, None)

# Get code with language filter
python_code = mq.run('select(.code("python")) | to_text()', fragment_content, None)

# Complex queries
implementations = mq.run(
    "select(.h2) | filter(contains('Implementation')) | next(.code)",
    fragment_content,
    None
)
```

**Structural Element Recognition**:

1. **Headings Hierarchy**: Understand document structure through H1, H2, H3 headings
2. **Code Blocks**: Extract code with language identification
3. **Decision Blocks**: Identify decisions in blockquotes ("> **Decision**: ...")
4. **Link Analysis**: Parse internal and external links
5. **Lists and Tasks**: Extract actionable items and their completion status
6. **Admonitions**: Recognize important notes, warnings, and tips

**Benefits of Markdown-Query**:
- Preserves document semantics and structure
- Enables precise content extraction
- Supports complex query patterns
- Language-agnostic content processing

## 2. Tantivy: Full-Text Search Engine

**Purpose**: Blazing-fast full-text search with git-storable indexes

**Index Structure**:

```python
from tantivy import Document, Index, SchemaBuilder

# Create comprehensive schema
schema_builder = SchemaBuilder()
schema_builder.add_text_field("id", stored=True)
schema_builder.add_text_field("title", stored=True)
schema_builder.add_text_field("content", stored=True)  # Full markdown content
schema_builder.add_text_field("code", stored=True)     # Extracted code blocks
schema_builder.add_text_field("decisions", stored=True) # Decision text
schema_builder.add_text_field("tags", stored=True)
schema_builder.add_date_field("created", stored=True)
schema_builder.add_facet_field("type")
schema_builder.add_facet_field("importance")
schema = schema_builder.build()

# Create index in git-storable location
index = Index(schema, path=".quaid/memory/indexes/tantivy/")

# Index a fragment
writer = index.writer()
doc = Document()
doc.add_text("id", fragment.id)
doc.add_text("title", fragment.title)
doc.add_text("content", fragment.full_content)
doc.add_text("code", "\n".join(fragment.code_blocks))
doc.add_text("decisions", "\n".join(fragment.decisions))
doc.add_text("tags", " ".join(fragment.tags))
doc.add_date("created", fragment.created)
doc.add_facet("type", fragment.type)
doc.add_facet("importance", fragment.importance)
writer.add_document(doc)
writer.commit()
```

**Query Capabilities**:

```python
# Basic full-text search
searcher = index.searcher()
query = index.parse_query("authentication JWT", ["content", "code", "decisions"])
results = searcher.search(query, 10)

# Field-specific search
query = index.parse_query("title:authentication OR code:verify_jwt")

# Phrase search for exact matches
query = index.parse_query('"token expiry"')

# Faceted search by type and importance
query = index.parse_query("authentication", facets={
    "type": ["concept", "implementation"],
    "importance": ["high"]
})

# Generate snippets with highlighting
snippet_generator = SnippetGenerator.create(searcher, query, "content")
for doc in results:
    snippet = snippet_generator.snippet_from_doc(doc)
    print(f"Match: {snippet}")
```

**Why Tantivy**:

- **Git-Storable**: Index files are compact binaries that can be committed to version control
- **Rust Performance**: Microsecond search latency, comparable to Elasticsearch
- **No Server**: Embedded library, no daemon or external dependencies
- **Rich Features**: Snippets, highlighting, faceting, phrase search
- **Python Bindings**: Native Python integration via tantivy-py

## 3. Polars: High-Performance Data Analysis

**Purpose**: Lightning-fast operations on structured metadata with lazy evaluation

**Data Structures**:

```python
import polars as pl

# fragments.jsonl - Main metadata index
fragments_df = pl.read_ndjson(".quaid/memory/indexes/fragments.jsonl")

# Schema includes:
# {
#   "id": str,
#   "type": str,
#   "title": str,
#   "tags": list[str],
#   "entities": list[str],
#   "created": datetime,
#   "updated": datetime,
#   "path": str,
#   "code_languages": list[str],
#   "has_decision": bool,
#   "heading_count": int,
#   "code_block_count": int,
#   "importance": str,
#   "completeness": str,
#   "related_ids": list[str]
# }

# tags.jsonl - Tag metadata and statistics
tags_df = pl.read_ndjson(".quaid/memory/indexes/tags.jsonl")

# graph.jsonl - Knowledge graph relationships
graph_df = pl.read_ndjson(".quaid/memory/indexes/graph.jsonl")
```

**Query Operations**:

```python
# Filter by metadata with complex conditions
auth_fragments = (
    fragments_df
    .filter(pl.col("tags").list.contains("authentication"))
    .filter(pl.col("type") == "concept")
    .filter(pl.col("importance") == "high")
    .sort("created", descending=True)
)

# Complex aggregations for insights
tag_stats = (
    fragments_df
    .explode("tags")
    .groupby("tags")
    .agg([
        pl.count().alias("fragment_count"),
        pl.col("type").value_counts().alias("type_distribution"),
        pl.col("created").min().alias("first_seen"),
        pl.col("created").max().alias("last_seen")
    ])
    .sort("fragment_count", descending=True)
)

# Join operations for graph traversal
related_fragments = (
    fragments_df
    .join(
        graph_df.filter(pl.col("from_id") == "auth-001"),
        left_on="id",
        right_on="to_id"
    )
)

# Temporal analysis for trends
recent_auth = (
    fragments_df
    .filter(pl.col("tags").list.contains("authentication"))
    .filter(pl.col("created") >= pl.datetime(2025, 1, 1))
    .groupby_dynamic("created", every="1w")
    .agg(pl.count().alias("fragments_per_week"))
)

# Entity co-occurrence analysis
entity_pairs = (
    fragments_df
    .select(["id", "entities"])
    .explode("entities")
    .join(
        fragments_df.select(["id", "entities"]).explode("entities"),
        on="id",
        suffix="_pair"
    )
    .filter(pl.col("entities") < pl.col("entities_pair"))
    .groupby(["entities", "entities_pair"])
    .count()
)
```

**Lazy Evaluation for Large Datasets**:

```python
# Lazy reading for datasets > 100MB
fragments_lazy = pl.scan_ndjson(".quaid/memory/indexes/fragments.jsonl")

result = (
    fragments_lazy
    .filter(pl.col("tags").list.contains("api"))
    .filter(pl.col("type") == "implementation")
    .select(["id", "title", "created", "tags"])
    .collect()  # Only execute when needed
)
```

**Benefits of Polars**:
- **Performance**: Multi-threaded, vectorized operations
- **Memory Efficiency**: Lazy evaluation and streaming
- **Expressive API**: Complex operations in readable code
- **Type Safety**: Strict typing catches errors early

## 4. SpaCy: Natural Language Processing

**Purpose**: Linguistic understanding, entity recognition, and similarity scoring

### 4.1 Query Intent Analysis

```python
import spacy

nlp = spacy.load("en_core_web_md")

def analyze_query_intent(query: str):
    """Determine structural elements to search based on user intent"""
    doc = nlp(query)

    intent = {
        "search_code": False,
        "search_concepts": False,
        "search_decisions": False,
        "search_patterns": False,
        "entities": [],
        "focus_verbs": [],
        "key_terms": []
    }

    # Detect intent from verbs and patterns
    for token in doc:
        if token.lemma_ in ["implement", "code", "write", "build", "create"]:
            intent["search_code"] = True
        elif token.lemma_ in ["decide", "choose", "select", "why"]:
            intent["search_decisions"] = True
        elif token.lemma_ in ["concept", "understand", "explain", "theory"]:
            intent["search_concepts"] = True
        elif token.lemma_ in ["pattern", "template", "reusable"]:
            intent["search_patterns"] = True

    # Extract named entities
    intent["entities"] = [ent.text for ent in doc.ents]

    # Extract key terms (nouns, proper nouns, adjectives)
    intent["key_terms"] = [
        token.text for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop
    ]

    return intent

# Example usage
query = "show me JWT implementation code examples"
intent = analyze_query_intent(query)
# Returns:
# {
#   "search_code": True,
#   "entities": ["JWT"],
#   "key_terms": ["JWT", "implementation", "code", "examples"]
# }
```

### 4.2 Entity Recognition and Linking

```python
# Custom NER model for project-specific entities
from spacy.training import Example

# Train custom NER to recognize:
# - Technology names (JWT, OAuth2, PostgreSQL)
# - Project-specific terms
# - File paths and function names

# Extract entities from fragments
def extract_entities(text: str):
    doc = nlp(text)
    entities = {
        "technologies": [],
        "products": [],
        "organizations": [],
        "custom_entities": [],
        "file_references": []
    }

    for ent in doc.ents:
        if ent.label_ == "TECH":
            entities["technologies"].append(ent.text)
        elif ent.label_ == "PRODUCT":
            entities["products"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(ent.text)

    # Custom pattern matching for file references
    file_pattern = r'\b[\w\-/]+\.(py|js|ts|md|json|yaml|yml)\b'
    import re
    file_matches = re.findall(file_pattern, text)
    entities["file_references"] = file_matches

    return entities
```

### 4.3 Similarity Scoring for Reranking

```python
def rerank_results(query: str, results: list[dict], top_k: int = 10):
    """Rerank search results using spaCy similarity"""
    query_doc = nlp(query)

    scored_results = []
    for result in results:
        # Combine title and snippet for similarity
        result_text = f"{result['title']} {result['snippet']}"
        result_doc = nlp(result_text)

        # Calculate semantic similarity
        similarity = query_doc.similarity(result_doc)

        # Boost score based on structural factors
        boost = 1.0
        if result.get("has_code") and "implement" in query.lower():
            boost *= 1.5
        if result.get("has_decision") and "why" in query.lower():
            boost *= 1.3
        if result.get("importance") == "high":
            boost *= 1.2

        final_score = similarity * boost
        scored_results.append((final_score, result))

    # Sort and return top k
    scored_results.sort(reverse=True, key=lambda x: x[0])
    return [result for score, result in scored_results[:top_k]]
```

### 4.4 Keyphrase Extraction

```python
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

def extract_keyphrases(text: str, top_n: int = 10):
    """Extract important phrases for indexing and tagging"""
    doc = nlp(text)

    # Extract noun phrases (key concepts)
    noun_chunks = [
        chunk.text.lower()
        for chunk in doc.noun_chunks
        if chunk.text.lower() not in STOP_WORDS
        and len(chunk.text.split()) <= 3  # Avoid overly long phrases
    ]

    # Count frequencies
    phrase_freq = Counter(noun_chunks)

    # Weight by position and importance
    weighted_phrases = {}
    for i, chunk in enumerate(doc.noun_chunks):
        phrase = chunk.text.lower()
        if phrase in phrase_freq:
            # Earlier mentions get higher weight
            position_weight = 1.0 - (i / len(doc.noun_chunks))
            weighted_phrases[phrase] = phrase_freq[phrase] * position_weight

    # Return top weighted phrases
    sorted_phrases = sorted(weighted_phrases.items(), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, _ in sorted_phrases[:top_n]]

# Use for auto-tagging and content summarization
keyphrases = extract_keyphrases(fragment_content)
# Example output: ["jwt authentication", "token validation", "api security"]
```

## 5. Integrated Search Pipeline

### Multi-Stage Search Architecture

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchResult:
    fragment_id: str
    title: str
    snippet: str
    score: float
    matched_elements: List[str]  # ["code", "heading", "decision"]
    entities: List[str]
    tags: List[str]
    type: str
    path: str

class QuaidSearch:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.tantivy_index = Index.open(".quaid/memory/indexes/tantivy/")
        self.fragments_df = pl.read_ndjson(".quaid/memory/indexes/fragments.jsonl")

        # Optional AI components
        self.semantic_search = None  # sentence-transformers
        self.cross_encoder = None   # FlashRank reranker
        self.classifier = None      # Hybrid classifier

    def search(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Multi-stage search pipeline with intelligent result processing
        """
        # STAGE 1: Query Intent Analysis (spaCy)
        intent = self._analyze_intent(query)

        # STAGE 2: Full-Text Search (Tantivy)
        tantivy_results = self._fulltext_search(query, intent, top_k * 3)

        # STAGE 3: Structural Analysis (markdown-query)
        enriched_results = self._structural_analysis(tantivy_results, intent)

        # STAGE 4: Metadata Filtering (Polars)
        filtered_results = self._metadata_filter(enriched_results, filters)

        # STAGE 5: Semantic Similarity (Optional AI)
        if self.semantic_search and len(filtered_results) > 0:
            filtered_results = self._semantic_rerank(query, filtered_results)

        # STAGE 6: Cross-Encoder Reranking (Optional AI)
        if self.cross_encoder and len(filtered_results) > 10:
            filtered_results = self._cross_encoder_rerank(query, filtered_results[:20])

        # STAGE 7: Final Processing and Formatting
        final_results = self._format_results(query, intent, filtered_results)

        return final_results[:top_k]

    def _analyze_intent(self, query: str) -> dict:
        """Use spaCy to understand query intent and extract entities"""
        doc = self.nlp(query)

        return {
            "search_code": any(t.lemma_ in ["implement", "code", "write"] for t in doc),
            "search_decisions": any(t.lemma_ in ["decide", "why", "rationale"] for t in doc),
            "search_patterns": any(t.lemma_ in ["pattern", "template", "approach"] for t in doc),
            "entities": [ent.text for ent in doc.ents],
            "key_terms": [t.text for t in doc if t.pos_ in ["NOUN", "PROPN"]],
            "urgency": any(t.text.lower() in ["urgent", "critical", "important"] for t in doc)
        }

    def _fulltext_search(self, query: str, intent: dict, limit: int) -> list:
        """Search with Tantivy using intent-aware field selection"""
        searcher = self.tantivy_index.searcher()

        # Build query based on intent
        fields = ["content"]
        if intent["search_code"]:
            fields.append("code")
        if intent["search_decisions"]:
            fields.append("decisions")

        # Add entity boosting
        query_str = query
        for entity in intent["entities"]:
            query_str += f' "{entity}"^2.0'  # Boost exact entity matches

        tantivy_query = self.tantivy_index.parse_query(query_str, fields)
        results = searcher.search(tantivy_query, limit)

        # Generate snippets with highlighting
        snippet_gen = SnippetGenerator.create(searcher, tantivy_query, "content")

        return [
            {
                "fragment_id": doc["id"],
                "title": doc["title"],
                "snippet": snippet_gen.snippet_from_doc(doc),
                "score": score,
                "path": doc.path
            }
            for score, doc in results
        ]

    def _structural_analysis(self, results: list, intent: dict) -> list:
        """Use markdown-query to analyze fragment structure"""
        enriched = []

        for result in results:
            # Load fragment content
            with open(result["path"]) as f:
                content = f.read()

            # Extract structural elements based on intent
            matched_elements = []

            if intent["search_code"]:
                code_blocks = mq.run("select(.code) | to_text()", content, None)
                if code_blocks.values:
                    matched_elements.append("code")
                    result["code_snippets"] = code_blocks.values[:2]  # First 2 code blocks

            if intent["search_decisions"]:
                decisions = mq.run("select(.blockquote) | to_text()", content, None)
                if decisions.values:
                    matched_elements.append("decision")
                    result["decision_text"] = decisions.values[0]  # First decision

            # Extract headings for context
            headings = mq.run("select(or(.h1, .h2)) | to_text()", content, None)
            result["headings"] = headings.values[:3]  # First 3 headings

            result["matched_elements"] = matched_elements
            enriched.append(result)

        return enriched

    def _metadata_filter(self, results: list, filters: Optional[dict]) -> list:
        """Filter using Polars dataframe operations"""
        if not filters:
            return results

        # Get fragment IDs from search results
        fragment_ids = [r["fragment_id"] for r in results]

        # Build Polars filter expression
        df_filter = pl.col("id").is_in(fragment_ids)

        if "type" in filters:
            df_filter = df_filter & (pl.col("type") == filters["type"])

        if "tags" in filters:
            df_filter = df_filter & pl.col("tags").list.contains(filters["tags"])

        if "importance" in filters:
            df_filter = df_filter & (pl.col("importance") == filters["importance"])

        if "date_range" in filters:
            start, end = filters["date_range"]
            df_filter = df_filter & pl.col("created").is_between(start, end)

        # Apply filter
        filtered_df = self.fragments_df.filter(df_filter)
        valid_ids = set(filtered_df["id"].to_list())

        return [r for r in results if r["fragment_id"] in valid_ids]

    def _semantic_rerank(self, query: str, results: list) -> list:
        """Optional semantic reranking using sentence transformers"""
        if not self.semantic_search:
            return results

        # This would use sentence-transformers for semantic similarity
        # Implementation details depend on the chosen semantic search library
        return results  # Placeholder

    def _cross_encoder_rerank(self, query: str, results: list) -> list:
        """Optional cross-encoder reranking using FlashRank"""
        if not self.cross_encoder:
            return results

        # This would use FlashRank or similar for query-document scoring
        # Implementation details depend on the chosen reranking library
        return results  # Placeholder

    def _format_results(self, query: str, intent: dict, results: list) -> List[SearchResult]:
        """Format results with rich metadata"""
        formatted_results = []

        for result in results:
            # Get fragment metadata
            fragment_data = self.fragments_df.filter(
                pl.col("id") == result["fragment_id"]
            ).to_dicts()[0]

            formatted_result = SearchResult(
                fragment_id=result["fragment_id"],
                title=result["title"],
                snippet=result["snippet"],
                score=result.get("semantic_score", result["score"]),
                matched_elements=result.get("matched_elements", []),
                entities=fragment_data.get("entities", []),
                tags=fragment_data.get("tags", []),
                type=fragment_data.get("type", "unknown"),
                path=result["path"]
            )

            formatted_results.append(formatted_result)

        # Final sorting by composite score
        formatted_results.sort(key=lambda x: x.score, reverse=True)
        return formatted_results
```

## 6. Storage Architecture

### File-Based Storage System

**Core Philosophy**: Everything stored as human-readable text files that can be version controlled

```
.quaid/
├── config.toml                    # Configuration with environment interpolation
├── memory/
│   ├── fragments/                 # Markdown fragments
│   │   ├── 2025-11-08-auth-001.md
│   │   └── ...
│   │
│   ├── indexes/                   # All indexes (git-storable)
│   │   ├── fragments.jsonl        # Fragment metadata (Polars-compatible)
│   │   ├── tags.jsonl             # Tag statistics
│   │   ├── graph.jsonl            # Knowledge graph relationships
│   │   │
│   │   └── tantivy/               # Tantivy search index
│   │       ├── meta.json
│   │       └── *.bin files
│   │
│   └── context/                   # Session context
│       ├── current.md             # Active session
│       ├── 2025-11-08-session-1.md # Archived summaries
│       └── sessions.jsonl         # Session index
│
├── models/                        # spaCy custom models
│   └── ner_project/               # Custom NER model
│
├── prompts/                       # AI prompt templates
│   ├── store_memory.prompt.md
│   ├── classify.prompt.md
│   └── search.prompt.md
│
└── cache/                         # Runtime cache (gitignored)
    ├── models/                    # Downloaded AI models
    └── query_cache/
```

### Fragment Storage Format

**Rich Structured Markdown with Comprehensive Metadata**:

```markdown
---
# Core Identity
id: "2025-11-08-auth-001"
type: concept
title: "JWT Authentication Strategy"
created: 2025-11-08T12:00:00Z
updated: 2025-11-08T14:30:00Z
author: "developer"
worktree: "main"

# Classification
tags: [authentication, jwt, security, api]
importance: high
confidence: 0.95
completeness: complete

# Extracted Entities
entities: ["JWT", "OAuth2", "Redis", "Express"]

# Structural Analysis
has_code: true
code_languages: ["python", "javascript"]
has_decision: true
heading_count: 5
code_block_count: 3
link_count: 4
word_count: 1250

# Relationships
related_ids: ["2025-11-07-session-mgmt-002", "2025-11-05-api-security-003"]
implements: []
referenced_by: ["2025-11-09-auth-middleware-004"]

# Processing Metadata
processed_by: "hybrid-classifier-v1"
extracted_entities: true
classified_at: "2025-11-08T12:05:00Z"
---

# JWT Authentication Strategy

> **Decision**: Use JWT tokens for API authentication instead of session cookies
> **Date**: 2025-11-08
> **Status**: Approved
> **Rationale**: Need for stateless authentication to support horizontal scaling

## Context

Our microservices architecture requires stateless authentication that can:
- Scale horizontally without session affinity
- Support mobile applications
- Enable independent service validation

## Implementation

### Token Generation

```python
import jwt
from datetime import datetime, timedelta

def generate_token(user_id: str, secret: str) -> str:
    """Generate JWT token with user claims"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, secret, algorithm='RS256')
```

### Token Validation Middleware

```javascript
// Express.js middleware
const validateJWT = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  // validation logic
};
```

## Trade-offs

**Advantages**:
- Stateless - no server-side session storage
- Scalable - works across multiple servers
- Mobile-friendly - easy token storage

**Disadvantages**:
- Token revocation complexity
- Larger payload size vs session ID
- Requires secure key management

## Related Concepts

- [Session Management](2025-11-07-session-mgmt-002.md)
- [API Security](2025-11-05-api-security-003.md)

## References

- [RFC 7519 - JWT Standard](https://tools.ietf.org/html/rfc7519)
- [OWASP JWT Security](https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html)
```

### Index Storage

**JSONL Format for Efficient Processing**:

```jsonl
{"id":"2025-11-08-auth-001","type":"concept","title":"JWT Authentication Strategy","tags":["authentication","jwt","security","api"],"entities":["JWT","OAuth2","Redis"],"created":"2025-11-08T12:00:00Z","updated":"2025-11-08T14:30:00Z","path":"memory/fragments/2025-11-08-auth-001.md","has_code":true,"code_languages":["python","javascript"],"has_decision":true,"importance":"high","completeness":"complete","word_count":1250}
{"id":"2025-11-08-auth-002","type":"implementation","title":"JWT Token Validation","tags":["authentication","jwt","python"],"entities":["JWT","Python"],"created":"2025-11-08T13:15:00Z","updated":"2025-11-08T13:15:00Z","path":"memory/fragments/2025-11-08-auth-002.md","has_code":true,"code_languages":["python"],"has_decision":false,"importance":"medium","completeness":"complete","word_count":850}
```

## 7. Textualize TUI Interface (Interactive Knowledge Management)

### Purpose
A rich, reactive terminal user interface that enables comprehensive interaction with the Quaid knowledge base for manual editing, viewing, and management operations.

### TUI Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Quaid TUI (Textualize)                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Main Layout (3-Panel)                   │   │
│  │                                                     │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │   │
│  │  │   Search    │ │  Content    │ │  Metadata   │     │   │
│  │  │   Panel     │ │   Panel     │ │   Panel     │     │   │
│  │  │             │ │             │ │             │     │   │
│  │  │ • Live      │ │ • Markdown  │ │ • Tags      │     │   │
│  │  │   Search    │ │   Preview   │ │ • Entities  │     │   │
│  │  │ • Filters   │ │ • Code      │ │ • Relations │     │   │
│  │  │ • History   │ │   Highlight │ │ • Timeline  │     │   │
│  │  │             │ │ • Editing   │ │             │     │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Interactive Components                 │   │
│  │                                                     │   │
│  │  • Knowledge Graph Visualizer                      │   │
│  │  • Timeline Explorer                               │   │
│  │  • Batch Operations Dashboard                      │   │
│  │  • Session Context Manager                         │   │
│  │  • Import/Export Wizard                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Core TUI Features

#### 7.1 Interactive Search Panel

```python
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, ListView, DataTable, Static
from textual.binding import Binding

class QuaidTUI(App):
    """Main TUI application for Quaid knowledge management"""

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+f", "focus_search", "Search"),
        Binding("ctrl+n", "new_fragment", "New Fragment"),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("f1", "help", "Help"),
        Binding("f2", "toggle_graph", "Toggle Graph"),
        Binding("f3", "toggle_timeline", "Toggle Timeline"),
    ]

    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 1;
    }

    #search-panel {
        width: 30%;
        border: solid $primary;
    }

    #content-panel {
        width: 50%;
        border: solid $secondary;
    }

    #metadata-panel {
        width: 20%;
        border: solid $accent;
    }
    """

    def compose(self) -> ComposeResult:
        """Create main layout"""
        with Horizontal():
            with Vertical(id="search-panel"):
                yield Input(placeholder="Search fragments...", id="search-input")
                yield ListView(id="search-results")
                yield Static("Filters:", classes="panel-header")
                # Filter widgets here...

            with Vertical(id="content-panel"):
                yield Static("Content Preview", classes="panel-header")
                yield Static("Select a fragment to view", id="content-preview")
                # Rich markdown viewer here...

            with Vertical(id="metadata-panel"):
                yield Static("Metadata", classes="panel-header")
                yield DataTable(id="metadata-table")
                # Metadata widgets here...
```

#### 7.2 Rich Content Viewing

```python
from textual.widgets import Markdown
from rich.syntax import Syntax
from rich.markdown import Markdown as RichMarkdown

class ContentViewer(Static):
    """Rich content viewer with syntax highlighting"""

    def show_fragment(self, fragment_path: str):
        """Display a fragment with rich formatting"""
        with open(fragment_path) as f:
            content = f.read()

        # Parse frontmatter and markdown
        frontmatter, markdown_body = self._parse_markdown(content)

        # Create rich display
        rendered = RichMarkdown(markdown_body)
        self.update(rendered)

    def show_code_snippet(self, code: str, language: str):
        """Display code with syntax highlighting"""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.update(syntax)

    def show_search_results(self, results: list[SearchResult]):
        """Display search results with highlighting"""
        # Custom rendering for search results
        pass
```

#### 7.3 Knowledge Graph Visualization

```python
from textual.containers import Container
from textual.widget import Widget
import networkx as nx

class KnowledgeGraphWidget(Container):
    """Interactive knowledge graph visualization"""

    def __init__(self):
        super().__init__()
        self.graph = nx.Graph()
        self.selected_node = None

    def build_graph_from_data(self, fragments_df, graph_df):
        """Build networkx graph from fragment data"""
        # Add nodes (fragments)
        for _, fragment in fragments_df.iterrows():
            self.graph.add_node(
                fragment['id'],
                title=fragment['title'],
                type=fragment['type'],
                tags=fragment['tags']
            )

        # Add edges (relationships)
        for _, edge in graph_df.iterrows():
            self.graph.add_edge(
                edge['from_id'],
                edge['to_id'],
                relationship=edge['relationship']
            )

    def render_graph(self):
        """Render graph using ASCII/Unicode art"""
        # Create visual representation of the graph
        # Use colors for different node types
        # Show connections between related concepts
        pass

    def on_click(self, event):
        """Handle node selection"""
        # Navigate to selected fragment
        pass
```

#### 7.4 Timeline Explorer

```python
from textual.widgets import DataTable
from datetime import datetime, timedelta

class TimelineExplorer(DataTable):
    """Interactive timeline for temporal knowledge exploration"""

    def __init__(self):
        super().__init__()
        self.cursor_type = "row"
        self.zebra_stripes = True

    def load_timeline_data(self, fragments_df):
        """Load fragments into timeline view"""
        self.clear(columns=True)

        # Add columns
        self.add_column("Date", key="date")
        self.add_column("Time", key="time")
        self.add_column("Title", key="title")
        self.add_column("Type", key="type")
        self.add_column("Tags", key="tags")

        # Add rows sorted by date
        sorted_fragments = fragments_df.sort("created", descending=True)
        for _, fragment in sorted_fragments.iterrows():
            created = fragment['created']
            self.add_row(
                (
                    created.strftime("%Y-%m-%d"),
                    created.strftime("%H:%M"),
                    fragment['title'][:50] + "..." if len(fragment['title']) > 50 else fragment['title'],
                    fragment['type'],
                    ", ".join(fragment['tags'][:3])
                ),
                key=fragment['id']
            )

    def on_select(self, event):
        """Handle timeline selection"""
        fragment_id = event.row_key.value
        # Trigger fragment display in content panel
        self.app.query_one("#content-panel").show_fragment(fragment_id)
```

#### 7.5 Batch Operations Dashboard

```python
from textual.screen import ModalScreen
from textual.widgets import ProgressBar, Label

class BatchOperationsScreen(ModalScreen):
    """Modal screen for batch operations"""

    def compose(self) -> ComposeResult:
        yield Label("Batch Operations")
        yield ProgressBar(id="progress-bar")
        yield Label("Processing fragments...", id="status-label")
        yield DataTable(id="operations-log")

    def run_batch_reindex(self):
        """Run batch reindexing with progress"""
        # Show progress bar
        # Update status
        # Log operations
        pass

    def run_batch_classify(self):
        """Run batch classification with AI models"""
        # Similar progress tracking
        pass

    def run_batch_export(self):
        """Run batch export to various formats"""
        # Export progress and completion
        pass
```

#### 7.6 Integration with MCP Server

```python
class TUIServerBridge:
    """Bridge between TUI and MCP server"""

    def __init__(self, mcp_client):
        self.mcp = mcp_client

    async def search_fragments(self, query: str, filters: dict = None):
        """Search via MCP server"""
        return await self.mcp.call("search_fragments", {
            "query": query,
            "filters": filters or {}
        })

    async def get_fragment(self, fragment_id: str):
        """Get fragment via MCP server"""
        return await self.mcp.call("get_fragment", {
            "id": fragment_id
        })

    async def update_fragment(self, fragment_id: str, content: str, metadata: dict):
        """Update fragment via MCP server"""
        return await self.mcp.call("update_fragment", {
            "id": fragment_id,
            "content": content,
            "metadata": metadata
        })

    async def create_fragment(self, content: str, metadata: dict):
        """Create fragment via MCP server"""
        return await self.mcp.call("create_fragment", {
            "content": content,
            "metadata": metadata
        })
```

### TUI Benefits

1. **Rich Interaction**: Direct manipulation of knowledge base with immediate visual feedback
2. **Discovery Mode**: Visual exploration of knowledge graphs and timelines
3. **Efficient Editing**: Quick fragment creation and editing with markdown preview
4. **Batch Management**: Visual interface for bulk operations and maintenance
5. **Context Awareness**: Always-visible metadata and relationships
6. **Keyboard-Driven**: Fast navigation for power users
7. **Offline Capability**: Full functionality without internet connection

### Development Approach

1. **Modular Components**: Each TUI feature as separate widget
2. **Async Integration**: Non-blocking operations with MCP server
3. **Responsive Design**: Adaptive layout for different terminal sizes
4. **Theme Support**: Customizable appearance with rich color schemes
5. **Accessibility**: Screen reader friendly and keyboard navigation

## 8. Performance Architecture

### Resource Requirements

**Minimal Setup (Core Features Only)**:
- **Disk**: 100MB (indexes + basic models)
- **RAM**: 500MB (in-memory indexes + processing)
- **CPU**: Any modern CPU
- **Search Latency**: <50ms for typical queries

**Recommended Setup (With Local Intelligence)**:
- **Disk**: 250MB (includes semantic models)
- **RAM**: 1GB (models + caches)
- **CPU**: Modern CPU with multiple cores
- **Search Latency**: <120ms with semantic search

**Maximum Setup (Full AI Features)**:
- **Disk**: 2GB (includes all models and embeddings)
- **RAM**: 4GB (all models in memory)
- **CPU**: Modern CPU, GPU optional
- **Search Latency**: <500ms with full AI pipeline

### Performance Optimizations

1. **Lazy Loading**: Use Polars lazy evaluation for large datasets
2. **Index Caching**: Keep search indexes in memory
3. **Model Caching**: Cache NLP models after first load
4. **Batch Processing**: Process multiple fragments together
5. **Incremental Updates**: Update indexes incrementally, not full rebuild

### Scalability Considerations

1. **Index Partitioning**: Split large indexes by date or category
2. **Query Optimization**: Use intent analysis to limit search scope
3. **Caching Strategy**: Cache common queries and results
4. **Resource Management**: Limit memory usage with streaming

---

## Benefits of This Architecture

### 1. No Vector Database Required
- **Tantivy** provides fast full-text search with BM25 ranking
- **spaCy** provides semantic similarity via word vectors
- **Structural metadata** enables precise filtering and scoring
- **Result quality** rivals vector databases without infrastructure complexity

### 2. Git-Native and Version Controlled
- All data committed to git with full history
- Indexes are compact and efficiently stored
- Merge-friendly when conflicts occur (indexes rebuild from source)
- No external database dependencies or backup requirements

### 3. High Performance
- **Tantivy**: Microsecond search latency
- **Polars**: Million-row filtering in milliseconds
- **spaCy**: Cached model loading for fast NLP
- **markdown-query**: Native Rust performance via Python bindings

### 4. Rich Query Capabilities
- Full-text search across all content types
- Structural queries (headings, code, decisions, links)
- Entity-based search and relationship tracking
- Temporal queries and trend analysis
- Faceted search and filtering

### 5. Extensible and Modular
- Plugin architecture for custom processors
- Custom NER models for domain-specific entities
- User-defined markdown conventions
- Flexible configuration system

### 6. Privacy and Control
- All processing can be done locally
- No data leaves the user's machine
- Complete control over models and processing
- Compliance-friendly for sensitive codebases

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)
- [ ] Setup Python project structure with pyproject.toml
- [ ] Implement storage layer (markdown + JSONL)
- [ ] Integrate markdown-query for parsing
- [ ] Build basic CLI with Click/Typer
- [ ] Create fragment management system

### Phase 2: Search Engine (Weeks 4-6)
- [ ] Integrate Tantivy-py for full-text search
- [ ] Build index management and maintenance
- [ ] Implement basic search with filtering
- [ ] Add snippet generation and highlighting
- [ ] Integrate Polars for metadata queries

### Phase 3: Intelligence Layer (Weeks 7-9)
- [ ] Integrate spaCy for NLP processing
- [ ] Implement query intent analysis
- [ ] Build entity extraction and linking
- [ ] Create similarity-based reranking
- [ ] Develop custom NER training pipeline

### Phase 4: Advanced Features (Weeks 10-12)
- [ ] Multi-stage search pipeline integration
- [ ] Knowledge graph relationship detection
- [ ] Analytics and statistics dashboard
- [ ] Worktree support and isolation
- [ ] Import/export capabilities

### Phase 5: Advanced Features (Weeks 13-15)
- [ ] Performance optimization and caching
- [ ] Analytics and statistics dashboard
- [ ] Worktree support and isolation
- [ ] Import/export capabilities
- [ ] **Textualize TUI Development**
  - [ ] Interactive knowledge browser with Textualize
  - [ ] Real-time search interface with live results
  - [ ] Fragment editor with markdown preview
  - [ ] Knowledge graph visualization
  - [ ] Batch operations and management tools
  - [ ] Session management and context viewer

### Phase 6: Polish and Distribution (Weeks 16-17)
- [ ] Comprehensive testing and validation
- [ ] Documentation and examples
- [ ] Package for PyPI distribution
- [ ] CI/CD pipeline setup
- [ ] TUI usability testing and refinement

---

## Conclusion

This architecture leverages Python's rich ecosystem to build a powerful, git-native knowledge management system that achieves enterprise-grade search quality without external infrastructure. The combination of Tantivy's speed, markdown-query's structural understanding, Polars' analytical power, and spaCy's linguistic intelligence creates a uniquely capable system for managing development knowledge.

**Key Architectural Innovations**:

1. **Multi-Modal Search**: Combines full-text, structural, semantic, and metadata signals
2. **Git-Native Storage**: All data human-readable and version-controlled
3. **Intelligent Processing**: AI-enhanced classification without vendor lock-in
4. **Performance First**: Sub-100ms query latency with large datasets
5. **Developer Centric**: Designed for developer workflows and tools

The result is a comprehensive solution that addresses the critical knowledge management challenges in modern software development while maintaining simplicity, performance, and extensibility.

---

**Previous**: [01-Vision-and-Problem.md](01-Vision-and-Problem.md) | **Next**: [03-Installation-and-Setup.md](03-Installation-and-Setup.md)