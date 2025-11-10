# 06 - Search and Intelligence

**Comprehensive guide to Quaid's advanced search capabilities and AI-powered intelligence features**

---

## Overview

Quaid's search and intelligence system combines multiple advanced technologies to provide exceptional information retrieval and understanding capabilities. This guide covers the full spectrum from basic search to AI-enhanced intelligence features.

### Core Intelligence Components

1. **Multi-Stage Search Pipeline**: Combines text, structural, and semantic search
2. **Local AI Classification**: Privacy-first automatic categorization and tagging
3. **Hybrid Intelligence**: Local processing with optional cloud enhancement
4. **Intent Understanding**: Natural language query processing
5. **Smart Reranking**: Context-aware result ordering

---

## Multi-Stage Search Pipeline

### Search Architecture Overview

Quaid uses a sophisticated multi-stage search pipeline that combines different search technologies:

```
Query Input
    ↓
┌─────────────────────────────────┐
│   Stage 1: Intent Analysis      │ ← spaCy NLP
│   - Parse query intent           │
│   - Extract entities            │
│   - Determine search scope      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Stage 2: Full-Text Search     │ ← Tantivy BM25
│   - Text matching               │
│   - Phrase search               │
│   - Field-specific search       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Stage 3: Structural Query     │ ← markdown-query
│   - Heading extraction          │
│   - Code block analysis         │
│   - Decision detection          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Stage 4: Metadata Filter      │ ← Polars
│   - Type filtering              │
│   - Tag filtering              │
│   - Date filtering              │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Stage 5: Semantic Rerank     │ ← Local AI
│   - Sentence similarity        │
│   - Cross-encoder reranking    │
│   - Context awareness          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Stage 6: Score Combination    │
│   - Weight signal combination   │
│   - Final ranking              │
│   - Result formatting          │
└─────────────────────────────────┘
    ↓
Ranked Results
```

### Basic Search Usage

```bash
# Simple text search
quaid recall "authentication"
quaid recall "JWT token validation"
quaid recall "database connection pooling"

# Natural language queries
quaid recall "how do I implement JWT authentication"
quaid recall "show me examples of error handling"
quaid recall "why did we choose PostgreSQL"

# Phrase search
quaid recall '"token expiration"'
quaid recall '"database migration"'
```

### Advanced Search Features

#### Field-Specific Search

```bash
# Search in specific fields
quaid recall "title:authentication"
quaid recall "code:validate_jwt"
quaid recall "tags:security"
quaid recall "type:decision"

# Multiple field search
quaid recall "title:authentication AND code:python"
quaid recall "tags:security OR tags:jwt"
```

#### Filtering and Faceting

```bash
# Filter by fragment type
quaid recall "authentication" --type concept
quaid recall "database" --type implementation
quaid recall "migration" --type decision

# Filter by tags
quaid recall "API" --tags authentication,security
quaid recall "error" --tags troubleshooting,debugging

# Filter by importance
quaid recall "security" --importance high
quaid recall "example" --importance medium

# Filter by date range
quaid recall "JWT" --created-since 2025-11-01
quaid recall "database" --updated-within 7d
```

#### Semantic Search

```bash
# Enable semantic search
quaid config set search.enable_semantic_search true

# Semantic search queries
quaid recall "ways to handle user authentication" --semantic
quaid recall "database optimization techniques" --semantic
quaid recall "API security best practices" --semantic

# Hybrid search (text + semantic)
quaid recall "authentication" --hybrid
```

---

## Local AI Classification

### Classification Strategy

Quaid uses a hybrid classification approach that combines multiple techniques:

1. **Rule-Based Analysis**: Structural and pattern-based classification
2. **Zero-Shot Classification**: AI-based categorization without training data
3. **Local Language Models**: Small, efficient models for privacy-first processing
4. **Entity Recognition**: Automatic extraction of key entities and concepts

### Classification Configuration

```toml
# .quaid/config.toml
[classification]
# Overall strategy
strategy = "hybrid"  # hybrid, ai-only, rule-based-only

# Type classification
[classification.type]
mode = "hybrid"  # user, auto, hybrid
validate = true
validation_threshold = 0.3
conflict_resolution = "prompt"

# Tag classification
[classification.tags]
enabled = true
max_tags = 10
min_confidence = 0.3
include_technologies = true
include_entities = true

# Importance classification
[classification.importance]
mode = "auto"
boost_from_admonitions = true
boost_from_decisions = true
boost_from_length = true

# Local AI configuration
[classification.local]
classification_mode = "rule-based"  # rule-based, zero-shot, llm
llm_model = "phi-2"  # phi-2, tinyllama, stablelm-zephyr
enable_semantic_search = true
embedding_model = "all-MiniLM-L6-v2"
```

### Type Classification

#### Fragment Types

Quaid supports these built-in fragment types:

- **concept**: Conceptual explanations and theories
- **implementation**: Code implementations and examples
- **decision**: Architectural decisions and ADRs
- **reference**: External references and links
- **pattern**: Reusable patterns and templates
- **troubleshooting**: Problem-solving guides
- **api-doc**: API documentation

#### Classification Examples

```bash
# Automatic classification
quaid store "JWT provides stateless authentication for microservices"
# → Automatically classified as "concept"

quaid store "```python\ndef validate_jwt(token):\n    # implementation\n```"
# → Automatically classified as "implementation"

quaid store "> **Decision**: Use JWT over sessions\n**Rationale**: Stateless scaling"
# → Automatically classified as "decision"

# Manual type specification
quaid store --type=pattern "Always validate JWT signature before processing claims"

# Type validation
quaid store --type=concept "```python\ndef validate_jwt(token):\n    # code\n```"
# → Warning: Content looks like implementation, not concept
```

### Tag Classification

#### Automatic Tag Extraction

Quaid automatically extracts tags from content using multiple methods:

```bash
# Store content with automatic tagging
quaid store "Implement JWT authentication using Python and Redis"

# Resulting tags:
# - jwt (from content)
# - authentication (from content)
# - python (from code language detection)
# - redis (from entity recognition)
# - security (from semantic classification)
```

#### Tag Categories

1. **Concept Tags**: Semantic topics (authentication, database, api)
2. **Technology Tags**: Technologies and frameworks (python, react, postgresql)
3. **Entity Tags**: Named entities and concepts (JWT, OAuth2, Redis)
4. **Code Language Tags**: Programming languages (python, javascript, typescript)

#### Custom Tag Rules

```toml
# Define custom tag extraction rules
[custom_rules.tag_extraction]
# File extension patterns
file_extensions = {pattern = r"\.(py|js|ts)$", tags = ["code"]}

# Framework detection
framework_names = {
    patterns = ["React", "Vue", "Angular", "Django", "Flask"],
    tags = ["framework"]
}

# Cloud provider detection
cloud_providers = {
    patterns = ["AWS", "Azure", "GCP"],
    tags = ["cloud"]
}

# Technology detection
technology_keywords = {
    jwt = ["authentication", "security"],
    docker = ["containerization", "devops"],
    redis = ["caching", "database"]
}
```

### Importance Classification

#### Importance Levels

- **high**: Critical information affecting architecture, security, or major decisions
- **medium**: Important information for daily work and implementation
- **low**: Nice-to-have information, references, or minor details

#### Importance Detection Rules

```toml
# Importance classification rules
[classification.importance]
# High importance indicators
high_importance_words = ["critical", "urgent", "security", "production", "breaking"]
high_importance_admonitions = ["!!! warning", "!!! danger", "!!! important"]
high_importance_tags = ["security", "production", "deployment"]

# Medium importance indicators
medium_importance_words = ["important", "recommended", "best practice"]
medium_importance_structure = ["multiple sections", "code examples", "comprehensive"]

# Low importance indicators
low_importance_words = ["note", "reference", "optional"]
low_importance_structure = ["single section", "short content", "link-only"]
```

---

## Local-Only Intelligence

### Privacy-First AI Processing

Quaid can perform all AI processing locally without sending data to external services:

```toml
# Configure local-only AI
[ai]
enabled = true
mode = "local"

[ai.local]
# Classification mode
classification_mode = "rule-based"  # rule-based, zero-shot, llm

# Local models
llm_model = "phi-2"  # Small, efficient local model
embedding_model = "all-MiniLM-L6-v2"
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Performance settings
max_workers = 4
use_gpu = false
low_memory_mode = false
```

### Local Classification Models

#### Rule-Based Classification (Recommended)

Fast, reliable classification using structural analysis:

```python
# Rule-based classification example
def classify_fragment(content: str) -> dict:
    """Classify fragment using rules and patterns"""

    # Detect type from structural markers
    if "> **Decision**:" in content or "**Rationale**:" in content:
        fragment_type = "decision"
    elif content.count("```") >= 2:
        fragment_type = "implementation"
    elif "concept" in content.lower() or "theory" in content.lower():
        fragment_type = "concept"
    else:
        fragment_type = "concept"  # Default

    # Detect importance from content
    importance_score = 0
    if any(word in content.lower() for word in ["critical", "important", "urgent"]):
        importance_score += 3
    if len(content) > 2000:
        importance_score += 2
    if "!!!" in content:  # Admonitions
        importance_score += 2

    if importance_score >= 5:
        importance = "high"
    elif importance_score >= 2:
        importance = "medium"
    else:
        importance = "low"

    return {
        "type": fragment_type,
        "importance": importance,
        "tags": extract_tags(content)
    }
```

#### Zero-Shot Classification

Using lightweight transformer models for classification:

```bash
# Install zero-shot classification
quaid models install zero-shot-classifier

# Configure zero-shot mode
quaid config set classification.local.classification_mode zero-shot

# Usage
quaid store "Complex implementation details"  # Automatically classified
```

#### Local Language Models

Using small local models for more sophisticated classification:

```bash
# Install Ollama and local model
quaid models install ollama
quaid models pull phi-2

# Configure local LLM mode
quaid config set classification.local.classification_mode llm
quaid config set classification.local.llm_model phi-2

# Usage
quaid store "Need to understand this complex system architecture"
```

### Local Semantic Search

#### Sentence Transformers

Use sentence-transformers for semantic similarity without external APIs:

```bash
# Install semantic search model
quaid models install sentence-transformers
quaid models download all-MiniLM-L6-v2

# Enable local semantic search
quaid config set search.enable_semantic_search true
quaid config set search.semantic_model all-MiniLM-L6-v2

# Semantic search
quaid recall "ways to handle user sessions" --semantic
```

#### Cross-Encoder Reranking

Use cross-encoder models for advanced reranking:

```bash
# Install FlashRank for reranking
quaid models install flashrank
quaid models download ms-marco-MiniLM-L-12-v2

# Enable reranking
quaid config set search.enable_reranking true
quaid config set search.reranker_model ms-marco-MiniLM-L-12-v2

# Reranked search
quaid recall "authentication patterns" --rerank
```

### Performance Comparison

| Feature | Rule-Based | Zero-Shot | Local LLM | Cloud API |
|---------|------------|-----------|-----------|-----------|
| Speed | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡ Slow | ⚡⚡ Medium |
| Accuracy | 75-80% | 80-85% | 85-90% | 95-98% |
| Privacy | ✅ 100% | ✅ 100% | ✅ 100% | ❌ No |
| Cost | Free | Free | Free | $$ |
| Resources | 50MB | 300MB | 2GB | Minimal |

**Recommended Setup**: Rule-based + semantic search (85-90% quality, 100% privacy)

---

## Hybrid Classification Strategy

### Three-Tier Classification System

Quaid uses a sophisticated hybrid approach for optimal results:

#### Tier 1: Structural Analysis (Rule-Based)

Fast, deterministic analysis of fragment structure:

```python
def assess_completeness(fragment: dict) -> str:
    """Assess fragment completeness based on structural signals"""
    score = 0

    # Heading structure
    if len(fragment['headings']) >= 3:
        score += 2
    elif len(fragment['headings']) >= 1:
        score += 1

    # Code examples
    if len(fragment['code_blocks']) >= 2:
        score += 2
    elif len(fragment['code_blocks']) >= 1:
        score += 1

    # External references
    if len(fragment['external_links']) >= 1:
        score += 1

    # Decision records
    if fragment['has_decision']:
        score += 1

    # Content length
    if len(fragment['content']) > 2000:
        score += 1

    # Determine completeness
    if score >= 7:
        return "complete"
    elif score >= 4:
        return "partial"
    else:
        return "stub"
```

#### Tier 2: Zero-Shot Classification (AI)

Semantic classification using pre-trained models:

```python
# Zero-shot tag classification
def classify_tags(content: str) -> list[str]:
    """Classify tags using zero-shot learning"""

    candidate_labels = [
        "authentication", "authorization", "security",
        "database", "caching", "api", "networking",
        "frontend", "backend", "devops", "testing",
        "performance", "error-handling", "data-processing"
    ]

    # Use zero-shot classifier
    result = classifier(content[:512], candidate_labels, multi_label=True)

    # Filter by confidence threshold
    tags = [
        label for label, score in zip(result['labels'], result['scores'])
        if score > 0.3
    ][:10]  # Max 10 tags

    return tags
```

#### Tier 3: User Override (Optional)

Allow users to override or refine automatic classifications:

```bash
# User-specified type (overrides AI)
quaid store --type=decision "We decided to use JWT"

# User-specified tags (adds to AI-generated)
quaid store --tags authentication,security "JWT implementation"

# Importance override
quaid store --importance=high "Critical security update"
```

### Validation and Confidence Scoring

```python
def validate_classification(fragment: dict, user_type: str = None) -> dict:
    """Validate and score classification confidence"""

    validation = {
        "type": fragment['type'],
        "type_confidence": 0.0,
        "tags": fragment['tags'],
        "tag_confidence": 0.0,
        "importance": fragment['importance'],
        "warnings": [],
        "suggestions": []
    }

    # Validate type with zero-shot classifier
    if user_type:
        # User provided type - validate it
        type_scores = classifier.classify_type(fragment['content'], user_type)
        validation["type_confidence"] = type_scores

        if type_scores < 0.3:
            validation["warnings"].append(
                f"Type '{user_type}' has low confidence ({type_scores:.2f})"
            )
            # Suggest alternative
            suggested_type, suggested_score = classifier.suggest_type(fragment['content'])
            validation["suggestions"].append(
                f"Consider type '{suggested_type}' (confidence: {suggested_score:.2f})"
            )
    else:
        # AI-determined type - confidence from classification
        validation["type_confidence"] = fragment.get('type_confidence', 0.5)

    # Validate tags
    tag_confidences = classifier.validate_tags(fragment['content'], fragment['tags'])
    avg_tag_confidence = sum(tag_confidences.values()) / len(tag_confidences)
    validation["tag_confidence"] = avg_tag_confidence

    # Add suggestions for low-confidence tags
    for tag, confidence in tag_confidences.items():
        if confidence < 0.3:
            validation["warnings"].append(
                f"Tag '{tag}' has low confidence ({confidence:.2f})"
            )

    return validation
```

---

## Intent Understanding

### Query Intent Analysis

Quaid analyzes search queries to understand user intent and optimize search results:

```python
def analyze_query_intent(query: str) -> dict:
    """Analyze query to understand search intent"""
    doc = nlp(query)

    intent = {
        "primary_intent": "search",  # search, implement, debug, understand
        "search_code": False,
        "search_decisions": False,
        "search_patterns": False,
        "search_concepts": False,
        "entities": [],
        "key_terms": [],
        "urgency": False,
        "complexity": "simple"  # simple, medium, complex
    }

    # Detect intent from verbs and patterns
    intent_verbs = {
        "implement": "implement", "code": "implement", "write": "implement",
        "debug": "debug", "fix": "debug", "error": "debug",
        "understand": "understand", "explain": "understand", "concept": "understand",
        "decide": "decide", "choose": "decide", "why": "decide"
    }

    for token in doc:
        if token.lemma_ in intent_verbs:
            intent["primary_intent"] = intent_verbs[token.lemma_]

    # Detect search targets
    if any(token.lemma_ in ["implement", "code", "write"] for token in doc):
        intent["search_code"] = True
    if any(token.lemma_ in ["decide", "why", "rationale"] for token in doc):
        intent["search_decisions"] = True
    if any(token.lemma_ in ["pattern", "template", "approach"] for token in doc):
        intent["search_patterns"] = True
    if any(token.lemma_ in ["concept", "theory", "understand"] for token in doc):
        intent["search_concepts"] = True

    # Extract entities
    intent["entities"] = [ent.text for ent in doc.ents]

    # Extract key terms
    intent["key_terms"] = [
        token.text for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop
    ]

    # Detect urgency
    urgency_words = ["urgent", "critical", "asap", "immediately"]
    if any(word in query.lower() for word in urgency_words):
        intent["urgency"] = True

    # Detect complexity
    if len(doc) > 10:
        intent["complexity"] = "complex"
    elif len(doc) > 5:
        intent["complexity"] = "medium"

    return intent
```

### Intent-Based Search Optimization

```bash
# Examples of intent-based search optimization

# Implementation intent
quaid recall "how to implement JWT authentication"
# → Prioritizes implementation fragments with code examples
# → Boosts results with Python/JavaScript code blocks
# → Filters for "implementation" type

# Debugging intent
quaid recall "JWT token is not validating properly"
# → Prioritizes troubleshooting fragments
# → Boosts fragments with error handling
# → Searches for common JWT errors

# Decision intent
quaid recall "why should we use JWT instead of sessions"
# → Prioritizes decision fragments
# → Boosts fragments with rationale sections
# → Searches for comparison content

# Concept intent
quaid recall "explain how JWT tokens work"
# → Prioritizes concept fragments
# → Boosts fragments with explanations
# → Searches for educational content
```

---

## Smart Reranking

### Reranking Pipeline

Quaid uses multiple reranking techniques to improve search result quality:

#### 1. Semantic Similarity Reranking

```python
def semantic_rerank(query: str, results: list) -> list:
    """Rerank results using semantic similarity"""

    # Encode query
    query_embedding = semantic_model.encode(query)

    # Encode result snippets
    result_texts = [f"{r['title']} {r['snippet']}" for r in results]
    result_embeddings = semantic_model.encode(result_texts)

    # Calculate similarities
    similarities = cosine_similarity([query_embedding], result_embeddings)[0]

    # Update scores
    for i, result in enumerate(results):
        result['semantic_score'] = float(similarities[i])
        result['final_score'] = (
            result['base_score'] * 0.4 +  # Original score
            result['semantic_score'] * 0.6   # Semantic score
        )

    # Sort by final score
    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results
```

#### 2. Cross-Encoder Reranking

```python
def cross_encoder_rerank(query: str, results: list, top_k: int = 10) -> list:
    """Rerank using cross-encoder for query-document pairs"""

    # Create query-document pairs
    pairs = [
        [query, f"{r['title']} {r['snippet']}"]
        for r in results[:top_k * 2]  # Rerank more candidates
    ]

    # Score with cross-encoder
    scores = cross_encoder.predict(pairs)

    # Update scores and sort
    for i, result in enumerate(results[:top_k * 2]):
        result['cross_encoder_score'] = float(scores[i])
        result['final_score'] = (
            result['base_score'] * 0.25 +
            result['semantic_score'] * 0.35 +
            result['cross_encoder_score'] * 0.40
        )

    # Sort and return top k
    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results[:top_k]
```

#### 3. Structural Scoring

```python
def structural_score(result: dict, query_intent: dict) -> float:
    """Calculate structural score based on intent and fragment structure"""

    score = 1.0  # Base score

    # Boost based on matched structural elements
    if query_intent["search_code"] and result.get("has_code"):
        score *= 1.5
    if query_intent["search_decisions"] and result.get("has_decision"):
        score *= 1.3
    if query_intent["search_patterns"] and result.get("is_pattern"):
        score *= 1.4

    # Boost based on importance
    if result.get("importance") == "high":
        score *= 1.2
    elif result.get("importance") == "low":
        score *= 0.8

    # Boost based on completeness
    if result.get("completeness") == "complete":
        score *= 1.1
    elif result.get("completeness") == "stub":
        score *= 0.7

    # Boost based on recency (for certain queries)
    if query_intent.get("urgency"):
        days_old = (datetime.now() - result["created"]).days
        if days_old < 7:
            score *= 1.1
        elif days_old > 90:
            score *= 0.9

    return score
```

### Adaptive Scoring

Quaid adapts scoring based on query characteristics and user feedback:

```python
def adaptive_scoring(results: list, query: str, user_feedback: dict = None) -> list:
    """Adapt scoring based on query and feedback"""

    # Analyze query characteristics
    query_complexity = len(query.split())
    has_specific_terms = any(term.isupper() for term in query.split())

    # Adjust weights based on query type
    if query_complexity > 10:
        # Complex query - emphasize semantic understanding
        weights = {"base": 0.2, "semantic": 0.5, "cross_encoder": 0.3}
    elif has_specific_terms:
        # Specific terms - emphasize exact matching
        weights = {"base": 0.5, "semantic": 0.3, "cross_encoder": 0.2}
    else:
        # General query - balanced approach
        weights = {"base": 0.3, "semantic": 0.4, "cross_encoder": 0.3}

    # Apply adaptive scoring
    for result in results:
        result["adaptive_score"] = (
            result.get("base_score", 0) * weights["base"] +
            result.get("semantic_score", 0) * weights["semantic"] +
            result.get("cross_encoder_score", 0) * weights["cross_encoder"]
        )

    # Sort by adaptive score
    results.sort(key=lambda x: x["adaptive_score"], reverse=True)
    return results
```

---

## Performance Optimization

### Search Performance

#### Caching Strategy

```toml
# Configure search caching
[search.cache]
# Enable result caching
enabled = true

# Cache TTL (seconds)
default_ttl = 3600

# Cache size limits
max_cache_size = "100MB"
max_entries = 10000

# Cache invalidation
invalidate_on_index_update = true
invalidate_on_fragment_change = true
```

#### Index Optimization

```bash
# Optimize search indexes
quaid index optimize

# Rebuild indexes for better performance
quaid index rebuild --optimize

# Monitor index performance
quaid index stats

# Output:
# Index Statistics:
# Documents: 1,247
# Index size: 45MB
# Average query time: 23ms
# Cache hit rate: 67%
```

### Model Performance

#### Local Model Optimization

```toml
# Optimize local AI models
[ai.local.performance]
# Parallel processing
max_workers = 4
batch_size = 32

# Memory optimization
low_memory_mode = false
model_cache_size = "500MB"

# GPU acceleration (if available)
use_gpu = true
gpu_memory_fraction = 0.5

# Model quantization (for memory efficiency)
quantize_models = true
```

#### Model Selection Guide

| Use Case | Recommended Model | Size | Quality | Speed |
|----------|------------------|------|---------|-------|
| Basic Classification | Rule-based | 50MB | 75-80% | ⚡⚡⚡ |
| Standard Use | Zero-shot | 300MB | 80-85% | ⚡⚡ |
| High Quality | Local LLM | 2GB | 85-90% | ⚡ |
| Maximum Quality | Cloud API | - | 95-98% | ⚡⚡ |

---

## Search Examples and Workflows

### Example 1: Finding Implementation Patterns

```bash
# Search for specific implementation pattern
quaid recall "database connection pooling" --type implementation

# Expand search with semantic understanding
quaid recall "ways to manage database connections efficiently" --semantic

# Filter by technology
quaid recall "database connection" --tags postgresql,pooling

# Results ranked by:
# 1. Exact text matches
# 2. Code presence and relevance
# 3. Semantic similarity
# 4. Implementation quality (completeness, examples)
```

### Example 2: Decision Research

```bash
# Search for architectural decisions
quaid recall "database choice decision" --type decision

# Include related concepts
quaid recall "why postgresql" --include-related

# Find comparisons
quaid recall "postgresql vs mongodb" --type comparison

# Results prioritized by:
# 1. Decision fragments
# 2. Rationale and consequences
# 3. Related concepts and alternatives
# 4. Implementation details
```

### Example 3: Debugging and Troubleshooting

```bash
# Search for specific error
quaid recall "JWT token expiration error" --type troubleshooting

# Find related debugging info
quaid recall "token validation issues" --semantic

# Include code examples
quaid recall "JWT error handling" --include-code

# Results enhanced with:
# 1. Error pattern matching
# 2. Solution approaches
# 3. Code fixes and examples
# 4. Related troubleshooting guides
```

---

## Next Steps

After mastering search and intelligence features:

1. **Learn CLI Commands**: [07-CLI-and-API-Reference.md](07-CLI-and-API-Reference.md)
2. **Explore Advanced Features**: [08-Advanced-Features.md](08-Advanced-Features.md)

---

**Previous**: [05-Core-Features.md](05-Core-Features.md) | **Next**: [07-CLI-and-API-Reference.md](07-CLI-and-API-Reference.md)