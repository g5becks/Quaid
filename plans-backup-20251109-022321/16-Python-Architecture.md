# 16 - Python-Based Architecture

**Comprehensive system design leveraging Python ecosystem for intelligent knowledge management**

---

## Executive Summary

Quaid is reimagined as a pure Python application leveraging a powerful combination of libraries:
- **markdown-query**: Structured markdown parsing and querying
- **tantivy-py**: Full-text search with git-storable indexes
- **polars**: High-performance dataframe operations
- **spaCy**: Natural language processing and entity recognition
- **Optional AI**: Classification and reranking (OpenAI, Anthropic, local models)

**Core Innovation**: Achieve vector-database-level search quality through:
1. Rich markdown structure (markdown-query)
2. Full-text indexing (tantivy)
3. Linguistic analysis (spaCy)
4. Structured metadata (polars)
5. Smart reranking (AI or rule-based)

**All data lives in git** - no external databases required.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Quaid Core (Python)                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              CLI Layer (Click/Typer)                │   │
│  │  - Command parsing and routing                      │   │
│  │  - User interaction and output formatting           │   │
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
│  │  • TOML configuration                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components Deep Dive

### 1. Markdown-Query: Structural Understanding

**Purpose**: Parse and query markdown structure for rich semantic understanding

**Capabilities**:
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

**Fragment Structure Convention**:
```markdown
---
id: "2025-11-08-auth-001"
type: concept
tags: [authentication, jwt, security]
entities: ["JWT", "OAuth2", "Redis"]  # Extracted by spaCy
created: 2025-11-08T12:00:00Z
---

# JWT Authentication

> **Decision**: Use JWT for stateless authentication
> **Rationale**: Enables horizontal scaling and mobile app support

## Context
Authentication requirements for microservices architecture...

## Implementation
```python
def verify_jwt(token: str) -> dict:
    """Verify JWT token signature and expiry"""
    # ...
```

## Related Concepts
- Session Management
- Token Refresh Strategy

## References
- [RFC 7519](https://tools.ietf.org/html/rfc7519)
```

**Markdown-Query Benefits**:
- Extract decisions from blockquotes
- Pull code implementations
- Navigate heading hierarchy
- Filter by code language
- Preserve document structure

---

### 2. Tantivy: Full-Text Search Engine

**Purpose**: Blazing-fast full-text search with git-storable indexes

**Index Structure**:
```python
from tantivy import Document, Index, SchemaBuilder

# Create schema
schema_builder = SchemaBuilder()
schema_builder.add_text_field("id", stored=True)
schema_builder.add_text_field("title", stored=True)
schema_builder.add_text_field("content", stored=True)  # Full markdown content
schema_builder.add_text_field("code", stored=True)     # Extracted code blocks
schema_builder.add_text_field("tags", stored=True)
schema_builder.add_date_field("created", stored=True)
schema_builder.add_facet_field("type")
schema = schema_builder.build()

# Create index in git-storable location
index = Index(schema, path=".quaid/indexes/tantivy/")

# Index a fragment
writer = index.writer()
doc = Document()
doc.add_text("id", fragment.id)
doc.add_text("title", fragment.title)
doc.add_text("content", fragment.full_content)
doc.add_text("code", "\n".join(fragment.code_blocks))
doc.add_text("tags", " ".join(fragment.tags))
doc.add_date("created", fragment.created)
doc.add_facet("type", fragment.type)
writer.add_document(doc)
writer.commit()
```

**Query Capabilities**:
```python
# Basic search
searcher = index.searcher()
query = index.parse_query("authentication JWT", ["content", "code"])
results = searcher.search(query, 10)

# Field-specific search
query = index.parse_query("title:authentication OR code:verify_jwt")

# Phrase search
query = index.parse_query('"token expiry"')

# Faceted search
query = index.parse_query("authentication", facets={"type": ["concept", "implementation"]})

# Generate snippets with highlighting
snippet_generator = SnippetGenerator.create(searcher, query, "content")
for doc in results:
    snippet = snippet_generator.snippet_from_doc(doc)
```

**Why Tantivy**:
- **Git-storable**: Index files are compact binaries that can be committed
- **Fast**: Rust-based, comparable to Elasticsearch
- **No server**: Embedded library, no daemon needed
- **Python bindings**: Native Python integration via tantivy-py
- **Advanced features**: Snippets, highlighting, faceting, phrase search

---

### 3. Polars: High-Performance Data Analysis

**Purpose**: Lightning-fast operations on structured metadata

**Data Structures**:
```python
import polars as pl

# fragments.jsonl - Main index
fragments_df = pl.read_ndjson(".quaid/memory/indexes/fragments.jsonl")

# Schema:
# {
#   "id": str,
#   "type": str,
#   "title": str, 
#   "tags": list[str],
#   "entities": list[str],
#   "created": datetime,
#   "updated": datetime,
#   "path": str,
#   "code_language": list[str],
#   "has_decision": bool,
#   "heading_count": int,
#   "code_block_count": int,
#   "related_ids": list[str]
# }

# tags.jsonl - Tag metadata
tags_df = pl.read_ndjson(".quaid/memory/indexes/tags.jsonl")

# graph.jsonl - Relationships
graph_df = pl.read_ndjson(".quaid/memory/indexes/graph.jsonl")
```

**Query Operations**:
```python
# Filter by metadata
auth_fragments = (
    fragments_df
    .filter(pl.col("tags").list.contains("authentication"))
    .filter(pl.col("type") == "concept")
    .sort("created", descending=True)
)

# Complex aggregations
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

# Temporal analysis
recent_auth = (
    fragments_df
    .filter(pl.col("tags").list.contains("authentication"))
    .filter(pl.col("created") >= pl.datetime(2025, 1, 1))
    .groupby_dynamic("created", every="1w")
    .agg(pl.count().alias("fragments_per_week"))
)

# Entity co-occurrence
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

**Lazy Execution for Large Datasets**:
```python
# Lazy reading for datasets > 100MB
fragments_lazy = pl.scan_ndjson(".quaid/memory/indexes/fragments.jsonl")

result = (
    fragments_lazy
    .filter(pl.col("tags").list.contains("api"))
    .select(["id", "title", "created"])
    .collect()  # Only execute when needed
)
```

---

### 4. SpaCy: Natural Language Processing

**Purpose**: Linguistic understanding, entity recognition, and similarity scoring

**Core Capabilities**:

#### 4.1 Query Intent Analysis
```python
import spacy

nlp = spacy.load("en_core_web_md")

def analyze_query_intent(query: str):
    """Determine structural elements to search"""
    doc = nlp(query)
    
    intent = {
        "search_code": False,
        "search_concepts": False,
        "search_decisions": False,
        "entities": [],
        "focus_verbs": [],
        "key_terms": []
    }
    
    # Detect intent from verbs and patterns
    for token in doc:
        if token.lemma_ in ["implement", "code", "write", "build"]:
            intent["search_code"] = True
        elif token.lemma_ in ["decide", "choose", "select"]:
            intent["search_decisions"] = True
        elif token.lemma_ in ["concept", "understand", "explain"]:
            intent["search_concepts"] = True
    
    # Extract entities
    intent["entities"] = [ent.text for ent in doc.ents]
    
    # Extract key terms (nouns, proper nouns)
    intent["key_terms"] = [
        token.text for token in doc 
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop
    ]
    
    return intent

# Example
query = "show me JWT implementation code"
intent = analyze_query_intent(query)
# {
#   "search_code": True,
#   "entities": ["JWT"],
#   "key_terms": ["JWT", "implementation", "code"]
# }
```

#### 4.2 Entity Recognition and Linking
```python
# Custom NER model for project-specific entities
from spacy.training import Example

# Train custom NER to recognize:
# - Class names
# - Function names  
# - Technology names (JWT, OAuth2, PostgreSQL)
# - Project-specific terms

# Example training data
TRAIN_DATA = [
    ("Use JWT for authentication", {"entities": [(4, 7, "TECH")]}),
    ("Implement OAuth2 flow", {"entities": [(10, 16, "TECH")]}),
]

# Extract entities from fragments
def extract_entities(text: str):
    doc = nlp(text)
    entities = {
        "technologies": [],
        "persons": [],
        "organizations": [],
        "custom": []
    }
    
    for ent in doc.ents:
        if ent.label_ == "TECH":
            entities["technologies"].append(ent.text)
        elif ent.label_ == "PERSON":
            entities["persons"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(ent.text)
    
    return entities
```

#### 4.3 Similarity Scoring for Reranking
```python
def rerank_results(query: str, results: list[dict], top_k: int = 10):
    """Rerank search results using spaCy similarity"""
    query_doc = nlp(query)
    
    scored_results = []
    for result in results:
        # Combine title and snippet for similarity
        result_text = f"{result['title']} {result['snippet']}"
        result_doc = nlp(result_text)
        
        # Calculate similarity
        similarity = query_doc.similarity(result_doc)
        
        # Boost score based on structural factors
        boost = 1.0
        if result.get("has_code") and "implement" in query.lower():
            boost *= 1.5
        if result.get("is_decision") and "why" in query.lower():
            boost *= 1.3
        
        final_score = similarity * boost
        scored_results.append((final_score, result))
    
    # Sort and return top k
    scored_results.sort(reverse=True, key=lambda x: x[0])
    return [result for score, result in scored_results[:top_k]]
```

#### 4.4 Keyphrase Extraction
```python
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

def extract_keyphrases(text: str, top_n: int = 10):
    """Extract important phrases for indexing"""
    doc = nlp(text)
    
    # Extract noun chunks
    noun_chunks = [
        chunk.text.lower() 
        for chunk in doc.noun_chunks 
        if chunk.text.lower() not in STOP_WORDS
    ]
    
    # Count frequencies
    phrase_freq = Counter(noun_chunks)
    
    return [phrase for phrase, _ in phrase_freq.most_common(top_n)]

# Use for auto-tagging
keyphrases = extract_keyphrases(fragment_content)
# ["jwt authentication", "token validation", "api security"]
```

---

### 5. Integrated Search Pipeline

**Multi-Stage Search with All Components**:

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
    path: str

class QuaidSearch:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.tantivy_index = Index.open(".quaid/indexes/tantivy/")
        self.fragments_df = pl.read_ndjson(".quaid/memory/indexes/fragments.jsonl")
    
    def search(
        self, 
        query: str, 
        filters: Optional[dict] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Multi-stage search pipeline
        """
        # STAGE 1: Query Intent Analysis (spaCy)
        intent = self._analyze_intent(query)
        
        # STAGE 2: Full-Text Search (tantivy)
        tantivy_results = self._fulltext_search(query, intent, top_k * 3)
        
        # STAGE 3: Structural Analysis (markdown-query)
        enriched_results = self._structural_analysis(tantivy_results, intent)
        
        # STAGE 4: Metadata Filtering (polars)
        filtered_results = self._metadata_filter(enriched_results, filters)
        
        # STAGE 5: Reranking (spaCy similarity + structural features)
        final_results = self._rerank(query, filtered_results, top_k)
        
        return final_results
    
    def _analyze_intent(self, query: str) -> dict:
        """Use spaCy to understand query intent"""
        doc = self.nlp(query)
        
        return {
            "search_code": any(t.lemma_ in ["implement", "code"] for t in doc),
            "search_decisions": any(t.lemma_ in ["decide", "why", "rationale"] for t in doc),
            "entities": [ent.text for ent in doc.ents],
            "key_terms": [t.text for t in doc if t.pos_ in ["NOUN", "PROPN"]],
            "verbs": [t.lemma_ for t in doc if t.pos_ == "VERB"]
        }
    
    def _fulltext_search(self, query: str, intent: dict, limit: int) -> list:
        """Search with tantivy"""
        searcher = self.tantivy_index.searcher()
        
        # Build query based on intent
        fields = ["content"]
        if intent["search_code"]:
            fields.append("code")
        
        # Add entity boosting
        query_str = query
        for entity in intent["entities"]:
            query_str += f" {entity}^2.0"  # Boost entities
        
        tantivy_query = self.tantivy_index.parse_query(query_str, fields)
        results = searcher.search(tantivy_query, limit)
        
        # Generate snippets
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
            # Load fragment
            with open(result["path"]) as f:
                content = f.read()
            
            # Extract structural elements based on intent
            matched_elements = []
            
            if intent["search_code"]:
                code_blocks = mq.run("select(.code) | to_text()", content, None)
                if code_blocks.values:
                    matched_elements.append("code")
                    result["code_snippets"] = code_blocks.values
            
            if intent["search_decisions"]:
                decisions = mq.run("select(.blockquote) | to_text()", content, None)
                if decisions.values:
                    matched_elements.append("decision")
                    result["decision_text"] = decisions.values
            
            # Extract headings for context
            headings = mq.run("select(or(.h1, .h2)) | to_text()", content, None)
            result["headings"] = headings.values
            
            result["matched_elements"] = matched_elements
            enriched.append(result)
        
        return enriched
    
    def _metadata_filter(self, results: list, filters: Optional[dict]) -> list:
        """Filter using polars dataframe operations"""
        if not filters:
            return results
        
        # Get fragment IDs
        fragment_ids = [r["fragment_id"] for r in results]
        
        # Build polars filter
        df_filter = pl.col("id").is_in(fragment_ids)
        
        if "type" in filters:
            df_filter = df_filter & (pl.col("type") == filters["type"])
        
        if "tags" in filters:
            df_filter = df_filter & pl.col("tags").list.contains(filters["tags"])
        
        if "date_range" in filters:
            start, end = filters["date_range"]
            df_filter = df_filter & pl.col("created").is_between(start, end)
        
        # Apply filter
        filtered_df = self.fragments_df.filter(df_filter)
        valid_ids = set(filtered_df["id"].to_list())
        
        return [r for r in results if r["fragment_id"] in valid_ids]
    
    def _rerank(self, query: str, results: list, top_k: int) -> List[SearchResult]:
        """Rerank using spaCy similarity and structural features"""
        query_doc = self.nlp(query)
        
        scored = []
        for result in results:
            # Base similarity score
            result_text = f"{result['title']} {result['snippet']}"
            result_doc = self.nlp(result_text)
            similarity = query_doc.similarity(result_doc)
            
            # Structural boosting
            boost = 1.0
            if "code" in result.get("matched_elements", []):
                boost *= 1.5
            if "decision" in result.get("matched_elements", []):
                boost *= 1.3
            
            final_score = similarity * boost * result["score"]
            
            scored.append((final_score, SearchResult(
                fragment_id=result["fragment_id"],
                title=result["title"],
                snippet=result["snippet"],
                score=final_score,
                matched_elements=result.get("matched_elements", []),
                entities=result.get("entities", []),
                path=result["path"]
            )))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [result for _, result in scored[:top_k]]
```

---

## Fragment Storage Format

**Rich Structured Markdown**:

```markdown
---
# Metadata (YAML frontmatter)
id: "2025-11-08-auth-001"
type: concept  # concept, implementation, decision, reference
title: "JWT Authentication Strategy"
tags: [authentication, jwt, security, api]
entities: ["JWT", "OAuth2", "Redis", "Express"]  # Auto-extracted by spaCy
created: 2025-11-08T12:00:00Z
updated: 2025-11-08T14:30:00Z
author: "developer"
worktree: "main"

# Structural metadata (auto-generated)
has_code: true
code_languages: ["python", "javascript"]
has_decision: true
heading_count: 5
code_block_count: 3
link_count: 4

# Relationships
related_ids: ["2025-11-07-session-mgmt-002", "2025-11-05-api-security-003"]
implements: []
referenced_by: ["2025-11-09-auth-middleware-004"]
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

## Tags Reference

#authentication #jwt #security #api #decision #microservices
```

**Markdown Conventions for Enhanced Search**:

1. **Decisions use blockquotes**: `> **Decision**: ...`
2. **Code blocks tagged with language**: ` ```python`
3. **Headings create hierarchy**: `# > ## > ###`
4. **Links show relationships**: `[Related](other-fragment.md)`
5. **Tags at bottom**: `#tag1 #tag2`

---

## Directory Structure

```
project/
└── .quaid/
    ├── config.toml                    # Configuration
    │
    ├── memory/
    │   ├── fragments/                 # Markdown fragments
    │   │   ├── 2025-11-08-auth-001.md
    │   │   └── ...
    │   │
    │   └── indexes/                   # All indexes (git-storable)
    │       ├── fragments.jsonl        # Polars-compatible metadata
    │       ├── tags.jsonl             # Tag statistics
    │       ├── graph.jsonl            # Relationships
    │       │
    │       └── tantivy/               # Tantivy search index
    │           ├── meta.json
    │           └── *.bin files
    │
    ├── models/                        # spaCy custom models
    │   └── ner_project/               # Custom NER model
    │       ├── config.cfg
    │       └── model files
    │
    └── cache/                         # Runtime cache (gitignored)
        ├── spacy_cache/
        └── query_cache/
```

**Git Considerations**:
- All indexes are binary but compact and diff-friendly
- Tantivy indexes rebuild from fragments if corrupted
- spaCy models can be versioned with git LFS if needed
- Cache directory gitignored for performance

---

## CLI API

### Core Commands

```bash
# Initialize project
quaid init <project-name>

# Store fragment
quaid store <content>
quaid store --file path/to/file.md
quaid store --type concept "JWT provides stateless auth"

# Search (multi-stage pipeline)
quaid search "how does JWT authentication work"
quaid search "show me authentication code" --type implementation
quaid search "why JWT" --type decision

# Advanced queries
quaid query --selector ".code | to_text()"  # Direct markdown-query
quaid query --sql "SELECT * FROM fragments WHERE type='decision'"  # SQL over polars

# Entity management
quaid entities list
quaid entities link "JWT" "OAuth2"  # Manual entity relationship

# Graph operations
quaid graph show <fragment-id>
quaid graph related <fragment-id> --depth 2

# Analytics
quaid stats
quaid stats --by-tag
quaid stats --timeline

# Model management
quaid models train-ner  # Train custom entity recognition
quaid models export
```

---

## Configuration

**`.quaid/config.toml`**:

```toml
[project]
name = "my-awesome-project"
initialized = "2025-11-08T12:00:00Z"

[search]
# Search pipeline configuration
enable_spacy = true
enable_ai_rerank = false  # Optional AI reranking
default_top_k = 10

[spacy]
model = "en_core_web_md"  # or en_core_web_lg for better similarity
custom_ner_path = ".quaid/models/ner_project"
enable_entity_linking = true

[tantivy]
index_path = ".quaid/memory/indexes/tantivy"
default_fields = ["content", "code", "title"]
snippet_length = 150

[polars]
lazy_loading = true  # Use scan_ndjson for large datasets
cache_enabled = true

[fragments]
# Structural conventions
decision_marker = "> **Decision**:"
code_block_languages = ["python", "javascript", "rust", "go"]
auto_extract_entities = true
auto_generate_tags = true

[ai]
# Optional AI provider for classification/reranking
enabled = false
provider = "openai"  # or anthropic, local
model = "gpt-4"
api_key = "#{OPENAI_API_KEY}"

[worktree]
auto_detect = true
isolation = "partial"  # full, partial, none
```

---

## Benefits of This Architecture

### 1. No Vector Database Needed
- **Tantivy** provides fast full-text search
- **spaCy** provides semantic similarity via word vectors
- **Structural metadata** enables precise filtering
- **Result quality** rivals vector databases without the infrastructure

### 2. Git-Native
- All data committed to git
- Indexes are compact and efficient
- Merge-friendly (indexes rebuild from source)
- No external database to backup/restore

### 3. Performance
- **Tantivy**: Microsecond search latency
- **Polars**: Million-row filtering in milliseconds
- **spaCy**: Cached model loading
- **markdown-query**: Native Rust performance via bindings

### 4. Rich Query Capabilities
- Full-text search across all content
- Structural queries (headings, code, decisions)
- Entity-based search
- Temporal queries
- Graph traversal

### 5. Extensibility
- Custom spaCy NER models
- Plugin architecture for processors
- Custom tantivy analyzers
- User-defined markdown conventions

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)
- [ ] Setup Python project structure
- [ ] Implement storage layer (markdown + JSONL)
- [ ] Integrate markdown-query for parsing
- [ ] Basic CLI with Click/Typer
- [ ] Fragment creation and management

### Phase 2: Search Foundation (Weeks 4-6)
- [ ] Integrate tantivy-py
- [ ] Build index management
- [ ] Implement basic full-text search
- [ ] Create snippet generation
- [ ] Add polars for metadata queries

### Phase 3: NLP Enhancement (Weeks 7-9)
- [ ] Integrate spaCy
- [ ] Implement query intent analysis
- [ ] Build entity extraction
- [ ] Create similarity-based reranking
- [ ] Develop custom NER training pipeline

### Phase 4: Advanced Features (Weeks 10-12)
- [ ] Multi-stage search pipeline
- [ ] Graph relationship detection
- [ ] Analytics and statistics
- [ ] Custom query language
- [ ] Worktree support

### Phase 5: Polish & Distribution (Weeks 13-14)
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Package for PyPI
- [ ] CI/CD pipeline

---

## Performance Targets

- **Search latency**: < 50ms for typical queries
- **Index build**: < 5s for 1000 fragments
- **Memory usage**: < 500MB for 10,000 fragments
- **Git repo size**: < 50MB for 5,000 fragments (including indexes)

---

## Conclusion

This architecture leverages Python's rich ecosystem to build a powerful, git-native knowledge management system that achieves vector-database-level search quality without external infrastructure. The combination of tantivy's speed, markdown-query's structural understanding, polars' analytical power, and spaCy's linguistic intelligence creates a uniquely capable system.

---

**Previous**: [15-CLI-API-Final.md](15-CLI-API-Final.md)

**Version**: 1.0  
**Last Updated**: 2025-11-09
