# 17 - Custom Scoring Framework

**AI-Friendly Markdown API and Structure-Based Relevance Scoring**

---

## Executive Summary

Quaid introduces a **structure-based scoring system** that leverages markdown conventions to compute relevance scores. By defining a clear markdown API that AI models can easily understand and follow, we enable precise control over search relevance without machine learning.

**Core Innovation**: Semantic markup through markdown structure creates a rich signal system for scoring:
- Document structure (headings, sections, nesting)
- Content classification (via special markers)
- Relationship signals (links, references)
- Importance indicators (emphasis, position)
- Metadata richness (tags, entities, completeness)

---

## The Markdown API for AI Models

### Design Philosophy

Create a **simple, consistent markdown syntax** that:
1. AI models can easily learn and apply
2. Humans can read and understand naturally
3. Tools can parse reliably
4. Provides rich scoring signals

### Core Conventions

```markdown
---
# Standard YAML frontmatter
id: "2025-11-09-concept-001"
type: concept  # concept | implementation | decision | reference | pattern
importance: high  # high | medium | low
completeness: complete  # complete | partial | stub
---

# Main Title [SCORE: 10]

> **Type**: Concept
> **Importance**: High  
> **Keywords**: authentication, JWT, security

## Overview [SCORE: 8]

Brief introduction...

## Core Concept [SCORE: 9]

!!! important "Key Insight"
    This is the most important point. [BOOST: +5]
!!!

### Implementation Details [SCORE: 7]

```python
# Code blocks get automatic relevance
def important_function():
    pass
```

## Decision Rationale [SCORE: 8]

> **Decision**: Use JWT for authentication
> **Date**: 2025-11-09
> **Status**: Approved
> **Impact**: High

[RELEVANCE: This section contains decision-making rationale]

## Related Topics [SCORE: 5]

- [[high-priority]] [Session Management](./session-001.md)
- [[medium-priority]] [OAuth Integration](./oauth-002.md)
- [[reference]] [JWT RFC 7519](https://tools.ietf.org/html/rfc7519)

## Examples [SCORE: 7]

### Real-World Usage [BOOST: +3]

Concrete examples boost relevance...

## Tags & Metadata [SCORE: 6]

#primary:authentication #primary:jwt #secondary:security #secondary:api

---
**Confidence**: 95%
**Last Updated**: 2025-11-09
**Reviewed By**: security-team
```

---

## Scoring Signal Taxonomy

### 1. Structural Signals

#### 1.1 Heading Hierarchy Score
```python
HEADING_SCORES = {
    "h1": 10,  # Main title - highest importance
    "h2": 8,   # Major sections
    "h3": 7,   # Subsections
    "h4": 5,   # Minor sections
    "h5": 3,   # Details
    "h6": 2,   # Fine details
}

def score_heading_match(query: str, heading: str, level: int) -> float:
    """
    Score based on heading level and match quality
    """
    base_score = HEADING_SCORES.get(f"h{level}", 1)
    
    # Exact match in high-level heading = very relevant
    if query.lower() in heading.lower():
        match_boost = 1.0 if level <= 2 else 0.5
        return base_score * (1.0 + match_boost)
    
    return base_score * 0.3  # Partial credit
```

#### 1.2 Position-Based Scoring
```python
def score_by_position(match_position: int, total_length: int) -> float:
    """
    Content near the beginning is more relevant
    """
    relative_position = match_position / total_length
    
    if relative_position < 0.1:  # First 10%
        return 1.5
    elif relative_position < 0.3:  # First 30%
        return 1.2
    elif relative_position < 0.5:  # First half
        return 1.0
    else:
        return 0.8
```

#### 1.3 Section Type Scoring
```markdown
## Overview [SECTION-TYPE: overview]
<!-- Introductory content - moderate relevance -->

## Core Concept [SECTION-TYPE: core]
<!-- Central concept - highest relevance -->

## Implementation [SECTION-TYPE: implementation]
<!-- Code and practical details - high for "how" queries -->

## Examples [SECTION-TYPE: examples]
<!-- Concrete examples - high for practical queries -->

## Background [SECTION-TYPE: background]
<!-- Context - lower relevance for specific queries -->

## References [SECTION-TYPE: references]
<!-- Links and citations - low direct relevance -->
```

```python
SECTION_TYPE_SCORES = {
    "core": 10,
    "implementation": 9,
    "examples": 8,
    "overview": 7,
    "decision": 9,
    "rationale": 8,
    "background": 5,
    "references": 3,
    "related": 4,
}

def extract_section_type(heading: str) -> str:
    """
    Infer section type from heading text
    """
    heading_lower = heading.lower()
    
    type_keywords = {
        "core": ["core", "key", "essential", "fundamental"],
        "implementation": ["implementation", "code", "how to", "usage"],
        "examples": ["example", "sample", "demo", "tutorial"],
        "overview": ["overview", "introduction", "summary"],
        "decision": ["decision", "rationale", "why"],
        "background": ["background", "history", "context"],
        "references": ["reference", "link", "resource", "see also"],
        "related": ["related", "similar", "see also"],
    }
    
    for type_name, keywords in type_keywords.items():
        if any(kw in heading_lower for kw in keywords):
            return type_name
    
    return "general"
```

---

### 2. Content Classification Signals

#### 2.1 Explicit Type Markers
```markdown
> **Type**: Concept | Implementation | Decision | Pattern | Reference
> **Importance**: High | Medium | Low
> **Completeness**: Complete | Partial | Stub

TYPE_SCORES = {
    ("concept", "high", "complete"): 10,
    ("implementation", "high", "complete"): 9,
    ("decision", "high", "complete"): 9,
    ("pattern", "medium", "complete"): 7,
    ("reference", "low", "complete"): 5,
    ("concept", "high", "partial"): 7,
    ("implementation", "medium", "stub"): 3,
}
```

#### 2.2 Special Block Types
```markdown
!!! important "Critical Information"
    This is crucial. [BOOST: +5]
!!!

!!! warning "Potential Issues"
    Be careful here. [BOOST: +3]
!!!

!!! tip "Best Practice"
    Recommended approach. [BOOST: +4]
!!!

!!! note "Additional Context"
    Extra information. [BOOST: +1]
!!!

!!! deprecated "Outdated"
    No longer recommended. [PENALTY: -5]
!!!
```

```python
ADMONITION_SCORES = {
    "important": 5,
    "critical": 5,
    "warning": 3,
    "tip": 4,
    "best-practice": 4,
    "note": 1,
    "deprecated": -5,
    "obsolete": -8,
}
```

---

### 3. Emphasis and Importance Signals

#### 3.1 Markdown Emphasis
```python
def score_emphasis(text: str, match: str) -> float:
    """
    Score based on text emphasis around match
    """
    score = 1.0
    
    # Check if match is in bold
    if f"**{match}**" in text or f"__{match}__" in text:
        score *= 1.5
    
    # Check if match is in italic
    if f"*{match}*" in text or f"_{match}_" in text:
        score *= 1.2
    
    # Check if match is in heading
    if text.startswith("#"):
        score *= 2.0
    
    # Check if match is in code
    if f"`{match}`" in text:
        score *= 1.3
    
    return score
```

#### 3.2 Priority Tags
```markdown
- [[high-priority]] [Critical Feature](./feature.md)
- [[medium-priority]] [Nice to Have](./enhancement.md)
- [[low-priority]] [Future Consideration](./future.md)

#primary:authentication  # Primary topic
#secondary:security      # Secondary topic
#tertiary:logging        # Related but not central
```

```python
PRIORITY_SCORES = {
    "high-priority": 3.0,
    "medium-priority": 1.5,
    "low-priority": 0.8,
}

TAG_PREFIX_SCORES = {
    "primary": 2.0,
    "secondary": 1.2,
    "tertiary": 0.8,
    "deprecated": 0.3,
}
```

---

### 4. Code Block Signals

#### 4.1 Code Block Metadata
```markdown
```python
# IMPORTANCE: high
# COMPLEXITY: medium
# STATUS: production
def authenticate_user(token: str) -> User:
    """
    Core authentication logic.
    KEYWORDS: authentication, jwt, security
    """
    pass
```

```python
CODE_BLOCK_SCORES = {
    "importance": {
        "high": 2.0,
        "medium": 1.3,
        "low": 0.8,
    },
    "status": {
        "production": 1.5,
        "tested": 1.3,
        "experimental": 0.9,
        "deprecated": 0.3,
    },
    "complexity": {
        "high": 0.9,  # Complex code less relevant for quick lookup
        "medium": 1.0,
        "low": 1.1,   # Simple code examples more accessible
    }
}
```

#### 4.2 Language-Specific Scoring
```python
def score_code_language(language: str, query_context: dict) -> float:
    """
    Boost code blocks in relevant language
    """
    query_langs = query_context.get("languages", [])
    
    if language in query_langs:
        return 2.0
    
    # Generic code is still relevant
    return 1.0
```

---

### 5. Relationship Signals

#### 5.1 Link Classification
```markdown
## Related Topics

- [[implements]] [Base Authentication](./base-auth.md)
- [[extends]] [OAuth2 Flow](./oauth2.md)
- [[referenced-by]] [API Security Guide](./api-security.md)
- [[depends-on]] [Token Storage](./token-storage.md)
- [[alternative-to]] [Session-Based Auth](./session-auth.md)
```

```python
RELATIONSHIP_SCORES = {
    "implements": 1.5,      # Strong implementation relationship
    "extends": 1.4,         # Extension/specialization
    "referenced-by": 1.2,   # Used by other components
    "depends-on": 1.3,      # Dependency relationship
    "alternative-to": 0.9,  # Alternative approach (less relevant)
    "supersedes": 1.6,      # Newer/better approach
    "deprecated-by": 0.5,   # Outdated (penalty)
}
```

#### 5.2 Cross-Reference Density
```python
def score_cross_references(fragment: dict) -> float:
    """
    Well-connected fragments are often more authoritative
    """
    incoming_links = len(fragment.get("referenced_by", []))
    outgoing_links = len(fragment.get("related_ids", []))
    
    # Balance: too few = isolated, too many = link spam
    total_links = incoming_links + outgoing_links
    
    if total_links == 0:
        return 0.8  # Isolated
    elif total_links <= 5:
        return 1.2  # Well connected
    elif total_links <= 10:
        return 1.0  # Highly connected
    else:
        return 0.9  # Over-connected (diminishing returns)
```

---

### 6. Metadata Completeness Signals

```python
def score_metadata_completeness(frontmatter: dict) -> float:
    """
    More complete metadata suggests higher quality
    """
    score = 1.0
    
    # Required fields
    required = ["id", "type", "title", "created"]
    if all(field in frontmatter for field in required):
        score += 0.2
    
    # Optional quality indicators
    if frontmatter.get("tags") and len(frontmatter["tags"]) >= 3:
        score += 0.15
    
    if frontmatter.get("entities") and len(frontmatter["entities"]) > 0:
        score += 0.15
    
    if frontmatter.get("importance") == "high":
        score += 0.3
    
    if frontmatter.get("completeness") == "complete":
        score += 0.2
    
    if frontmatter.get("reviewed_by"):
        score += 0.25  # Reviewed content is higher quality
    
    if frontmatter.get("confidence") and frontmatter["confidence"] >= 90:
        score += 0.15
    
    return score
```

---

## Composite Scoring Algorithm

### Multi-Factor Scoring Function

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ScoringContext:
    """Context for computing relevance score"""
    query: str
    query_intent: dict  # From spaCy analysis
    matched_text: str
    match_position: int
    total_doc_length: int
    fragment_metadata: dict
    structural_context: dict  # Heading level, section type, etc.

class StructuralScorer:
    """
    Compute relevance scores based on document structure
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.weights = config.get("scoring_weights", {
            "heading_match": 0.20,
            "section_type": 0.15,
            "position": 0.10,
            "emphasis": 0.10,
            "metadata_completeness": 0.10,
            "code_relevance": 0.15,
            "relationship": 0.10,
            "content_type": 0.10,
        })
    
    def compute_score(self, ctx: ScoringContext) -> Dict[str, float]:
        """
        Compute multi-factor relevance score
        """
        scores = {}
        
        # 1. Heading hierarchy score
        if ctx.structural_context.get("in_heading"):
            heading_level = ctx.structural_context["heading_level"]
            scores["heading_match"] = self._score_heading_match(
                ctx.query, 
                ctx.matched_text, 
                heading_level
            )
        else:
            scores["heading_match"] = 0
        
        # 2. Section type relevance
        section_type = ctx.structural_context.get("section_type", "general")
        scores["section_type"] = self._score_section_type(
            section_type, 
            ctx.query_intent
        )
        
        # 3. Position in document
        scores["position"] = self._score_position(
            ctx.match_position, 
            ctx.total_doc_length
        )
        
        # 4. Text emphasis
        scores["emphasis"] = self._score_emphasis(
            ctx.matched_text, 
            ctx.query
        )
        
        # 5. Metadata completeness
        scores["metadata_completeness"] = self._score_metadata(
            ctx.fragment_metadata
        )
        
        # 6. Code relevance
        if ctx.structural_context.get("in_code_block"):
            scores["code_relevance"] = self._score_code_block(
                ctx.structural_context.get("code_metadata", {}),
                ctx.query_intent
            )
        else:
            scores["code_relevance"] = 0
        
        # 7. Relationship strength
        scores["relationship"] = self._score_relationships(
            ctx.fragment_metadata
        )
        
        # 8. Content type alignment
        scores["content_type"] = self._score_content_type(
            ctx.fragment_metadata.get("type"),
            ctx.fragment_metadata.get("importance"),
            ctx.query_intent
        )
        
        # Compute weighted total
        total_score = sum(
            score * self.weights.get(factor, 0)
            for factor, score in scores.items()
        )
        
        # Add individual scores for debugging
        scores["total"] = total_score
        
        return scores
    
    def _score_heading_match(
        self, 
        query: str, 
        heading: str, 
        level: int
    ) -> float:
        """Score based on heading level and match"""
        base_score = HEADING_SCORES.get(f"h{level}", 1)
        
        # Exact phrase match
        if query.lower() in heading.lower():
            return base_score * 2.0
        
        # Partial word match
        query_words = set(query.lower().split())
        heading_words = set(heading.lower().split())
        overlap = query_words & heading_words
        
        if overlap:
            match_ratio = len(overlap) / len(query_words)
            return base_score * (1.0 + match_ratio)
        
        return base_score * 0.3
    
    def _score_section_type(
        self, 
        section_type: str, 
        query_intent: dict
    ) -> float:
        """Score based on section type and query intent"""
        base_score = SECTION_TYPE_SCORES.get(section_type, 5)
        
        # Boost if section type matches intent
        if query_intent.get("search_code") and section_type == "implementation":
            base_score *= 1.5
        elif query_intent.get("search_decisions") and section_type == "decision":
            base_score *= 1.5
        elif query_intent.get("search_concepts") and section_type == "core":
            base_score *= 1.3
        
        return base_score / 10.0  # Normalize to 0-1
    
    def _score_position(self, position: int, total: int) -> float:
        """Score based on position in document"""
        relative_pos = position / max(total, 1)
        
        if relative_pos < 0.1:
            return 1.0
        elif relative_pos < 0.3:
            return 0.8
        elif relative_pos < 0.5:
            return 0.6
        else:
            return 0.4
    
    def _score_emphasis(self, text: str, query: str) -> float:
        """Score based on text emphasis"""
        score = 0.5  # Base score
        
        if f"**{query}**" in text or f"__{query}__" in text:
            score += 0.3
        if f"*{query}*" in text or f"_{query}_" in text:
            score += 0.2
        if f"`{query}`" in text:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_metadata(self, metadata: dict) -> float:
        """Score based on metadata completeness"""
        score = 0.5
        
        if metadata.get("importance") == "high":
            score += 0.2
        if metadata.get("completeness") == "complete":
            score += 0.15
        if metadata.get("tags") and len(metadata["tags"]) >= 3:
            score += 0.1
        if metadata.get("reviewed_by"):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_code_block(
        self, 
        code_metadata: dict, 
        query_intent: dict
    ) -> float:
        """Score code blocks based on metadata"""
        if not query_intent.get("search_code"):
            return 0.5  # Not looking for code
        
        importance = code_metadata.get("importance", "medium")
        status = code_metadata.get("status", "unknown")
        
        score = CODE_BLOCK_SCORES["importance"].get(importance, 1.0)
        score *= CODE_BLOCK_SCORES["status"].get(status, 1.0)
        
        return min(score / 2.0, 1.0)  # Normalize
    
    def _score_relationships(self, metadata: dict) -> float:
        """Score based on relationship density"""
        incoming = len(metadata.get("referenced_by", []))
        outgoing = len(metadata.get("related_ids", []))
        total = incoming + outgoing
        
        if total == 0:
            return 0.5
        elif total <= 5:
            return 1.0
        elif total <= 10:
            return 0.8
        else:
            return 0.6
    
    def _score_content_type(
        self, 
        content_type: str, 
        importance: str,
        query_intent: dict
    ) -> float:
        """Score based on content type and importance"""
        base_scores = {
            "concept": 0.8,
            "implementation": 0.9,
            "decision": 0.9,
            "pattern": 0.7,
            "reference": 0.5,
        }
        
        score = base_scores.get(content_type, 0.6)
        
        if importance == "high":
            score *= 1.3
        elif importance == "low":
            score *= 0.8
        
        return min(score, 1.0)
```

---

## AI Model Guidelines

### Markdown Structure Template

**Provide this to AI models when creating fragments**:

```markdown
# Creating Quaid Fragments - AI Model Guide

## Essential Structure

Every fragment should follow this template:

---
id: "YYYY-MM-DD-topic-NNN"
type: concept | implementation | decision | reference | pattern
title: "Clear Descriptive Title"
importance: high | medium | low
completeness: complete | partial | stub
tags: [primary-tag, secondary-tag, ...]
entities: [Entity1, Entity2, ...]
---

# Main Title

> **Type**: [Match the frontmatter type]
> **Importance**: [Match the frontmatter importance]
> **Keywords**: keyword1, keyword2, keyword3

## Overview [SECTION-TYPE: overview]

Brief introduction (1-2 paragraphs). Keep this concise.

## Core Concept [SECTION-TYPE: core]

!!! important "Key Insight"
    The single most important point goes here.
    This will receive the highest relevance boost.
!!!

Main concept explanation with detailed information.

## Implementation [SECTION-TYPE: implementation]

For code-focused content:

```python
# IMPORTANCE: high
# STATUS: production
def important_function():
    """
    KEYWORDS: relevant, terms, here
    """
    pass
```

## Decision Rationale [SECTION-TYPE: decision]

> **Decision**: What was decided
> **Date**: YYYY-MM-DD
> **Status**: Approved | Proposed | Rejected
> **Impact**: High | Medium | Low

Explanation of why this decision was made.

## Examples [SECTION-TYPE: examples]

### Real-World Usage

Concrete examples that show practical application.

## Related Topics [SECTION-TYPE: related]

- [[high-priority]] [Related Topic 1](./topic1.md)
- [[medium-priority]] [Related Topic 2](./topic2.md)

## References [SECTION-TYPE: references]

- [External Link 1](https://example.com)

## Tags

#primary:main-topic #secondary:related-topic

---
**Confidence**: 95%
**Last Updated**: YYYY-MM-DD
**Reviewed By**: team-name (optional)
```

### Scoring Optimization Tips for AI Models

**Include in prompts to AI assistants**:

```
When creating Quaid fragments, optimize for search relevance:

1. HEADING STRUCTURE
   - Put most important info in H2 headings (score: 8)
   - Use descriptive heading text with keywords
   - H1 is the title (score: 10)

2. SECTION TYPES
   - "Core Concept" sections = highest relevance (10)
   - "Implementation" sections = high for code queries (9)
   - "Decision" sections = high for "why" queries (9)
   - "Examples" sections = high for practical queries (8)

3. EMPHASIS
   - Use !!! important for critical information (+5 boost)
   - Bold (**text**) for key terms (+50% boost)
   - Code blocks for implementation details (+30% boost)

4. METADATA
   - Always include: type, importance, completeness
   - Add 3+ relevant tags
   - Extract entities (technologies, names, concepts)
   - Set importance=high for critical content

5. CODE BLOCKS
   - Add metadata comments at top:
     # IMPORTANCE: high
     # STATUS: production
     # KEYWORDS: auth, jwt, validate
   - Use specific language tags (python, javascript, etc.)

6. RELATIONSHIPS
   - Link 3-5 related documents
   - Use relationship markers: [[implements]], [[extends]]
   - Avoid over-linking (>10 links)

7. POSITION
   - Put most relevant content in first 30% of document
   - Core concepts near the top
   - Examples and references toward the end
```

---

## Configuration

**`.quaid/config.toml` scoring section**:

```toml
[scoring]
# Enable custom structural scoring
enabled = true

# Scoring weights (must sum to 1.0)
[scoring.weights]
heading_match = 0.20
section_type = 0.15
position = 0.10
emphasis = 0.10
metadata_completeness = 0.10
code_relevance = 0.15
relationship = 0.10
content_type = 0.10

# Heading scores
[scoring.heading_scores]
h1 = 10
h2 = 8
h3 = 7
h4 = 5
h5 = 3
h6 = 2

# Section type scores
[scoring.section_types]
core = 10
implementation = 9
decision = 9
examples = 8
overview = 7
background = 5
references = 3
related = 4

# Admonition boosts
[scoring.admonitions]
important = 5
critical = 5
warning = 3
tip = 4
note = 1
deprecated = -5

# Priority tag multipliers
[scoring.priorities]
high-priority = 3.0
medium-priority = 1.5
low-priority = 0.8

# Tag prefix multipliers
[scoring.tag_prefixes]
primary = 2.0
secondary = 1.2
tertiary = 0.8
deprecated = 0.3
```

---

## Search Integration

### Using Structural Scores with Search Pipeline

```python
class QuaidSearch:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.tantivy_index = Index.open(".quaid/indexes/tantivy/")
        self.fragments_df = pl.read_ndjson(".quaid/memory/indexes/fragments.jsonl")
        self.scorer = StructuralScorer(load_config())
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Multi-stage search with structural scoring
        """
        # Stage 1-4: As before (intent, tantivy, markdown-query, polars)
        intent = self._analyze_intent(query)
        tantivy_results = self._fulltext_search(query, intent, top_k * 3)
        enriched_results = self._structural_analysis(tantivy_results, intent)
        filtered_results = self._metadata_filter(enriched_results)
        
        # Stage 5: Apply structural scoring
        scored_results = []
        for result in filtered_results:
            # Create scoring context
            ctx = ScoringContext(
                query=query,
                query_intent=intent,
                matched_text=result["snippet"],
                match_position=result.get("position", 0),
                total_doc_length=result.get("doc_length", 1000),
                fragment_metadata=result.get("metadata", {}),
                structural_context=result.get("structure", {})
            )
            
            # Compute structural score
            structural_scores = self.scorer.compute_score(ctx)
            
            # Combine with tantivy score
            final_score = (
                result["tantivy_score"] * 0.4 +  # Full-text relevance
                structural_scores["total"] * 0.6  # Structural relevance
            )
            
            result["score"] = final_score
            result["score_breakdown"] = structural_scores
            scored_results.append(result)
        
        # Sort by final score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_results[:top_k]
```

---

## Example: Query Analysis with Structural Scoring

### Query: "how to implement JWT authentication"

**Step 1: Intent Analysis (spaCy)**
```python
{
    "search_code": True,  # "implement" detected
    "search_concepts": False,
    "search_decisions": False,
    "entities": ["JWT"],
    "key_terms": ["implement", "authentication"]
}
```

**Step 2: Tantivy Full-Text Search**
Returns 30 candidates with text match scores.

**Step 3: Markdown-Query Structural Analysis**
Extract from each candidate:
- Heading hierarchy
- Section types
- Code blocks
- Blockquote decisions
- Link relationships

**Step 4: Structural Scoring**

Fragment A (scores):
```json
{
    "heading_match": 0.8,      // "JWT" in H2 heading
    "section_type": 0.9,       // "Implementation" section
    "position": 1.0,           // Match in first 10%
    "emphasis": 0.7,           // Some bold text
    "metadata_completeness": 0.9,  // Complete metadata
    "code_relevance": 1.0,     // Python code, importance=high
    "relationship": 0.8,       // 4 related links
    "content_type": 0.9,       // type=implementation, importance=high
    "total": 0.86
}
```

Fragment B (scores):
```json
{
    "heading_match": 0.5,      // "authentication" in H3
    "section_type": 0.7,       // "Background" section
    "position": 0.6,           // Match in middle of doc
    "emphasis": 0.5,           // No emphasis
    "metadata_completeness": 0.6,  // Partial metadata
    "code_relevance": 0.0,     // No code blocks
    "relationship": 0.6,       // 8 links (too many)
    "content_type": 0.5,       // type=reference, importance=medium
    "total": 0.54
}
```

**Final Ranking**:
1. Fragment A (final: 0.86 * 0.6 + tantivy_score * 0.4)
2. Fragment B (final: 0.54 * 0.6 + tantivy_score * 0.4)

---

## Benefits

1. **AI-Friendly**: Clear, learnable conventions
2. **Interpretable**: Scores are explainable
3. **Tunable**: Adjust weights per project
4. **No ML Required**: Pure rule-based scoring
5. **Git-Native**: All scores computed from markdown structure
6. **Human-Readable**: Same markdown humans write naturally

---

## Implementation Roadmap

### Phase 1: Basic Structural Scoring (Week 1)
- [ ] Implement heading hierarchy scoring
- [ ] Implement position-based scoring
- [ ] Implement section type detection
- [ ] Basic configuration support

### Phase 2: Advanced Signals (Week 2)
- [ ] Emphasis detection (bold, italic, code)
- [ ] Admonition block parsing
- [ ] Priority tag system
- [ ] Metadata completeness scoring

### Phase 3: Code & Relationships (Week 3)
- [ ] Code block metadata extraction
- [ ] Relationship type detection
- [ ] Cross-reference density analysis
- [ ] Language-specific boosting

### Phase 4: Integration & Tuning (Week 4)
- [ ] Integrate with search pipeline
- [ ] Score breakdown visualization
- [ ] Configuration UI
- [ ] A/B testing framework

---

## Conclusion

This structure-based scoring framework provides a powerful, interpretable alternative to ML-based relevance ranking. By defining clear markdown conventions that AI models can easily learn, we create a virtuous cycle: better-structured content leads to better search results, which encourages better content creation.

The system is:
- **Transparent**: Every score is explainable
- **Controllable**: Tune weights and rules per project
- **Maintainable**: No training data or models needed
- **Effective**: Leverages rich markdown structure for precise relevance

---

**Previous**: [16-Python-Architecture.md](16-Python-Architecture.md)

**Version**: 1.0  
**Last Updated**: 2025-11-09
