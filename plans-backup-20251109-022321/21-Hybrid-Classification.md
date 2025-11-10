# 21 - Hybrid Classification Strategy

**Zero-Shot Classification for Tags, Structured Extraction for Type**

---

## Executive Summary

After analyzing the requirements, here's the optimal approach:

**Classification should handle BOTH, but differently**:

1. **Type** → User/LLM provides, classifier validates/suggests
2. **Tags** → Zero-shot classifier generates automatically
3. **Importance** → Zero-shot classifier determines
4. **Completeness** → Rule-based analysis (structural signals)

**Why this hybrid approach**:
- **Type** is high-level and domain-specific (user knows best)
- **Tags** are granular and numerous (classifier excels here)
- **Importance** benefits from semantic understanding
- **Completeness** is structural (code present? examples? tests?)

---

## The Classification Strategy

### Three-Tier Classification System

```
┌─────────────────────────────────────────────────────────────┐
│                   Fragment Classification                    │
│                                                             │
│  Input: Markdown Fragment + Optional User Hints            │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Tier 1: Structural Analysis (Rule-Based)         │     │
│  │   - Completeness (complete/partial/stub)          │     │
│  │   - Section detection                             │     │
│  │   - Code block count & languages                  │     │
│  │   - Heading structure                             │     │
│  │   - Task completion ratio                         │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Tier 2: Zero-Shot Classification (BART-MNLI)     │     │
│  │   - Tags (5-10 granular tags)                     │     │
│  │   - Importance (high/medium/low)                  │     │
│  │   - Type validation/suggestion (if user provided) │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Tier 3: User Override (Optional)                 │     │
│  │   - Type (concept/implementation/decision/etc)    │     │
│  │   - Additional tags                               │     │
│  │   - Importance override                           │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  Final Classification Metadata                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 1: Type Classification

### Why LLM Must Provide Type

**Reasoning**:
1. **Domain-Specific** - "decision" vs "pattern" vs "concept" are project-specific
2. **Context-Dependent** - Same content could be "reference" in one project, "implementation" in another
3. **High-Level Intent** - LLM/user interaction determines intent
4. **Limited Set** - Only 5-7 types, perfect for constrained LLM output

**Type Taxonomy** (LLM must choose from this list):
```python
FRAGMENT_TYPES = [
    "concept",        # Conceptual explanation, theory
    "implementation", # Code implementation, how-to
    "decision",       # Decision record, ADR
    "reference",      # Links to external resources, docs
    "pattern",        # Reusable pattern, template
    "troubleshooting", # Problem-solving guide
    "api-doc"         # API documentation
]
```

### Type Control Flow

**The ONLY way to control type is via slash commands that trigger LLM prompts**:

```
User: /store_memory JWT authentication implementation with refresh tokens
                     
                     ↓
                     
Quaid triggers prompt (using Promptdown):
"You are storing a memory fragment. You MUST classify it with a type 
from this list: concept, implementation, decision, reference, pattern, 
troubleshooting, api-doc.

Content: {user_message}

Return JSON:
{
  "type": "implementation",  # REQUIRED - must be from list above
  "title": "JWT Authentication",
  "tags": ["jwt", "auth", "security"]  # Auto-generated or suggested
}"

                     ↓
                     
LLM Response:
{
  "type": "implementation",
  "title": "JWT Authentication with Refresh Tokens",
  "tags": ["jwt", "authentication", "security", "tokens"]
}

                     ↓
                     
Zero-shot classifier validates LLM's choice:
scores = classifier.classify(content, candidate_labels=FRAGMENT_TYPES)
if scores[llm_type] < 0.3:
    warn(f"⚠ Type '{llm_type}' seems unusual (confidence: {scores[llm_type]:.2f})")
    suggest(f"  Classifier suggests: '{top_type}' (confidence: {scores[top_type]:.2f})")
```

### Slash Command Examples

**Explicit type control** (user specifies type):
```bash
# User explicitly says "decision"
/store_memory decision to use JWT over sessions for auth

→ Prompt includes: "User specified type: 'decision'. Use this type."
→ LLM response: {"type": "decision", ...}
```

**Implicit type** (LLM infers from context):
```bash
# User doesn't specify type
/store_memory here's how to implement JWT validation

→ Prompt: "Analyze content and choose appropriate type from list"
→ LLM response: {"type": "implementation", ...}
```

**Type override via slash command**:
```bash
# User can force a specific type
/store_memory --type=troubleshooting JWT token expiration issues

→ Prompt includes: "REQUIRED type: 'troubleshooting'"
→ LLM response: {"type": "troubleshooting", ...}
```

### No Frontmatter Control

**This is WRONG** ❌:
```markdown
---
type: implementation  # User cannot control via frontmatter
title: JWT Auth
---
```

**This is CORRECT** ✅:
```
User → /slash_command → LLM prompt → LLM provides type → Validated by classifier
```

The markdown file is the **output** of the LLM, not the input. The LLM generates the frontmatter based on the slash command prompt.

---

## Part 2: Tag Classification (Zero-Shot)

### Why Classifier Should Generate Tags

**Reasoning**:
1. **Many Tags** - Could have 5-10 tags per fragment (tedious for user)
2. **Granular** - Specific technologies, concepts, patterns
3. **Semantic Understanding** - Classifier detects implied topics
4. **Consistent** - Same terms across fragments

### Tag Extraction Strategy

**Two-Stage Tag Generation**:

**Stage 1: Broad Topic Detection**
```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Detect broad topics
broad_labels = [
    "authentication", "authorization", "security",
    "database", "caching", "api",
    "frontend", "backend", "devops",
    "testing", "monitoring", "performance",
    "error-handling", "data-processing"
]

result = classifier(
    fragment_content[:512],  # Truncate for speed
    candidate_labels=broad_labels,
    multi_label=True
)

# Get top 3-5 broad topics (score > 0.3)
broad_tags = [
    label for label, score in zip(result['labels'], result['scores'])
    if score > 0.3
][:5]
```

**Stage 2: Technology/Library Detection** (spaCy NER + rules)
```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Extract technology entities
doc = nlp(fragment_content)
tech_entities = []

# Known tech patterns
TECH_PATTERNS = {
    "JWT", "OAuth", "Redis", "PostgreSQL", "MongoDB",
    "React", "Vue", "Angular", "Django", "Flask",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP"
}

for ent in doc.ents:
    if ent.label_ in ["PRODUCT", "ORG"] and ent.text in TECH_PATTERNS:
        tech_entities.append(ent.text.lower())

# Also check code blocks for language tags
code_languages = extract_code_languages(fragment_content)

# Combine
all_tags = broad_tags + tech_entities + code_languages
```

### Tag Taxonomy

**Three Tag Categories**:

1. **Concept Tags** (from zero-shot) - `authentication`, `caching`, `security`
2. **Technology Tags** (from NER + code) - `jwt`, `redis`, `postgresql`, `python`
3. **Project Tags** (user-provided) - `payment-service`, `v2-migration`, `deprecated`

---

## Part 3: Importance Classification (Zero-Shot)

### Why Classifier Should Determine Importance

**Reasoning**:
1. **Semantic Signals** - Words like "critical", "deprecated", "experimental"
2. **Structural Signals** - Length, admonitions, code blocks
3. **Consistent** - Avoid user bias

```python
# Detect importance level
importance_labels = ["critical", "high-priority", "medium-priority", "low-priority"]

importance_result = classifier(
    f"{fragment_title}. {fragment_content[:500]}",
    candidate_labels=importance_labels
)

# Map to high/medium/low
importance_score = importance_result['scores'][0]
if importance_score > 0.6:
    importance = "high"
elif importance_score > 0.3:
    importance = "medium"
else:
    importance = "low"

# Boost based on structural signals
if has_admonition_important or has_admonition_critical:
    importance = "high"
```

---

## Part 4: Completeness (Rule-Based)

### Why Rules Work Best for Completeness

**Reasoning**:
1. **Structural** - Presence of sections, code, examples
2. **Objective** - Not semantic, just "does it have X?"
3. **Fast** - No model inference needed

```python
def assess_completeness(fragment: dict) -> str:
    """
    Assess fragment completeness based on structural signals
    """
    score = 0
    max_score = 10
    
    # Has multiple headings? (structure)
    if len(fragment['headings']) >= 3:
        score += 2
    elif len(fragment['headings']) >= 1:
        score += 1
    
    # Has code examples?
    if len(fragment['code_blocks']) >= 2:
        score += 2
    elif len(fragment['code_blocks']) >= 1:
        score += 1
    
    # Has external references?
    if len(fragment['links']['external']) >= 1:
        score += 1
    
    # Has internal links?
    if len(fragment['links']['internal']) >= 2:
        score += 1
    
    # Has decision record?
    if fragment['decisions']:
        score += 1
    
    # Has tasks?
    if fragment['tasks']['total'] > 0:
        score += 1
        # All tasks completed?
        if fragment['tasks']['pending'] == 0:
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

---

## Implementation

### Unified Classifier

```python
from transformers import pipeline
import spacy
from typing import Dict, List, Any

class HybridClassifier:
    """
    Hybrid classification system combining zero-shot, NER, and rules
    """
    
    def __init__(self):
        # Zero-shot classifier
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )
        
        # NER for entity extraction
        self.nlp = spacy.load("en_core_web_sm")
        
        # Known technology patterns
        self.tech_patterns = {
            "jwt", "oauth", "redis", "postgresql", "mongodb",
            "react", "vue", "angular", "django", "flask",
            "docker", "kubernetes", "aws", "azure", "gcp",
            "python", "javascript", "typescript", "rust", "go"
        }
        
        # Fragment type taxonomy
        self.fragment_types = [
            "concept", "implementation", "decision", 
            "reference", "pattern", "troubleshooting", "api-doc"
        ]
        
        # Broad topic labels for tagging
        self.topic_labels = [
            "authentication", "authorization", "security",
            "database", "caching", "api", "networking",
            "frontend", "backend", "devops", "infrastructure",
            "testing", "monitoring", "performance", "optimization",
            "error-handling", "logging", "data-processing",
            "configuration", "deployment", "migration"
        ]
    
    def classify(
        self,
        fragment: Dict[str, Any],
        user_provided_type: str = None
    ) -> Dict[str, Any]:
        """
        Classify a fragment using hybrid approach
        """
        content = fragment['content']
        title = fragment.get('title', '')
        
        # Stage 1: Structural analysis (completeness)
        completeness = self._assess_completeness(fragment)
        
        # Stage 2: Zero-shot classification
        # 2a. Type (if not provided by user)
        if user_provided_type:
            fragment_type = user_provided_type
            type_confidence = self._validate_type(content, user_provided_type)
        else:
            fragment_type, type_confidence = self._classify_type(content, title)
        
        # 2b. Tags (always auto-generate)
        tags = self._extract_tags(content, fragment)
        
        # 2c. Importance
        importance = self._classify_importance(content, title, fragment)
        
        return {
            "type": fragment_type,
            "type_confidence": type_confidence,
            "tags": tags,
            "importance": importance,
            "completeness": completeness
        }
    
    def _classify_type(
        self, 
        content: str, 
        title: str
    ) -> tuple[str, float]:
        """
        Classify fragment type using zero-shot
        """
        # Use title + first 500 chars for type detection
        text = f"{title}. {content[:500]}"
        
        result = self.classifier(
            text,
            candidate_labels=self.fragment_types,
            hypothesis_template="This is a {} document."
        )
        
        return result['labels'][0], result['scores'][0]
    
    def _validate_type(
        self, 
        content: str, 
        user_type: str
    ) -> float:
        """
        Validate user-provided type, return confidence
        """
        result = self.classifier(
            content[:500],
            candidate_labels=self.fragment_types,
            hypothesis_template="This is a {} document."
        )
        
        # Find score for user's type
        type_index = result['labels'].index(user_type)
        return result['scores'][type_index]
    
    def _extract_tags(
        self,
        content: str,
        fragment: Dict[str, Any]
    ) -> List[str]:
        """
        Extract tags using multi-stage approach
        """
        tags = set()
        
        # Stage 1: Zero-shot topic detection
        result = self.classifier(
            content[:512],
            candidate_labels=self.topic_labels,
            multi_label=True
        )
        
        # Get topics with score > 0.3
        topic_tags = [
            label for label, score in zip(result['labels'], result['scores'])
            if score > 0.3
        ][:5]
        tags.update(topic_tags)
        
        # Stage 2: Technology/library extraction (NER)
        doc = self.nlp(content[:1000])
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"]:
                ent_lower = ent.text.lower()
                if ent_lower in self.tech_patterns:
                    tags.add(ent_lower)
        
        # Stage 3: Code languages
        code_langs = [cb['language'] for cb in fragment.get('code_blocks', [])]
        tags.update(code_langs)
        
        # Stage 4: Extract from title
        title_words = fragment.get('title', '').lower().split()
        for word in title_words:
            if word in self.tech_patterns:
                tags.add(word)
        
        return sorted(list(tags))[:10]  # Max 10 tags
    
    def _classify_importance(
        self,
        content: str,
        title: str,
        fragment: Dict[str, Any]
    ) -> str:
        """
        Classify importance using zero-shot + structural signals
        """
        # Zero-shot importance detection
        importance_labels = [
            "critical", "high-priority", "medium-priority", "low-priority"
        ]
        
        result = self.classifier(
            f"{title}. {content[:500]}",
            candidate_labels=importance_labels,
            hypothesis_template="This is a {} topic."
        )
        
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        # Map to high/medium/low
        if top_label in ["critical", "high-priority"] and top_score > 0.5:
            base_importance = "high"
        elif top_label == "low-priority" and top_score > 0.5:
            base_importance = "low"
        else:
            base_importance = "medium"
        
        # Boost based on structural signals
        admonitions = fragment.get('admonitions', [])
        has_important = any(
            adm['type'] in ['important', 'critical'] 
            for adm in admonitions
        )
        
        if has_important:
            base_importance = "high"
        
        # Boost if it's a decision with high impact
        decisions = fragment.get('decisions', [])
        has_high_impact = any(
            dec.get('impact') == 'high' 
            for dec in decisions
        )
        
        if has_high_impact:
            base_importance = "high"
        
        return base_importance
    
    def _assess_completeness(self, fragment: Dict[str, Any]) -> str:
        """
        Assess completeness using structural signals
        """
        score = 0
        max_score = 10
        
        # Multiple headings?
        headings_count = len(fragment.get('headings', []))
        if headings_count >= 3:
            score += 2
        elif headings_count >= 1:
            score += 1
        
        # Code examples?
        code_blocks_count = len(fragment.get('code_blocks', []))
        if code_blocks_count >= 2:
            score += 2
        elif code_blocks_count >= 1:
            score += 1
        
        # External references?
        if len(fragment.get('links', {}).get('external', [])) >= 1:
            score += 1
        
        # Internal links?
        if len(fragment.get('links', {}).get('internal', [])) >= 2:
            score += 1
        
        # Decisions?
        if fragment.get('decisions'):
            score += 1
        
        # Tasks?
        tasks = fragment.get('tasks', {})
        if tasks.get('total', 0) > 0:
            score += 1
            if tasks.get('pending', 0) == 0:
                score += 1
        
        # Content length
        if len(fragment.get('content', '')) > 2000:
            score += 1
        
        # Determine completeness
        if score >= 7:
            return "complete"
        elif score >= 4:
            return "partial"
        else:
            return "stub"
```

---

## Usage Examples

### Example 1: Slash Command with Explicit Type

```bash
# User command
/store_memory --type=implementation JWT authentication with refresh tokens

# Quaid generates prompt using Promptdown
```
```python
from promptdown import StructuredPrompt

prompt = StructuredPrompt.from_promptdown_file(".quaid/prompts/store_memory.prompt.md")
prompt.apply_template_values({
    "user_message": "JWT authentication with refresh tokens",
    "required_type": "implementation",
    "available_types": FRAGMENT_TYPES
})

# LLM processes and returns
llm_response = {
    "type": "implementation",
    "title": "JWT Authentication with Refresh Tokens",
    "content": "# JWT Authentication...",
    "tags": ["jwt", "authentication", "security", "tokens"]
}

# Classifier validates LLM's choice
classifier = HybridClassifier()
validation = classifier.validate_llm_classification(llm_response)

# Output:
{
    "type": "implementation",
    "type_confidence": 0.89,  # High confidence ✓
    "tags": ["authentication", "jwt", "security", "python", "tokens"],
    "importance": "high",
    "completeness": "complete"
}
```

### Example 2: Slash Command with Implicit Type

```bash
# User doesn't specify type
/store_memory here's how to troubleshoot JWT token expiration errors

# Quaid prompts LLM to infer type
```
```python
prompt.apply_template_values({
    "user_message": "here's how to troubleshoot JWT token expiration errors",
    "required_type": None,  # LLM must choose
    "available_types": FRAGMENT_TYPES
})

# LLM infers from content
llm_response = {
    "type": "troubleshooting",  # LLM chose this
    "title": "JWT Token Expiration Errors",
    "content": "...",
    "tags": ["jwt", "troubleshooting", "errors", "tokens"]
}

# Classifier validates
validation = classifier.validate_llm_classification(llm_response)

# Output:
{
    "type": "troubleshooting",
    "type_confidence": 0.91,  # High confidence ✓
    "tags": ["troubleshooting", "jwt", "errors", "tokens", "debugging"],
    "importance": "medium",
    "completeness": "partial"
}
```

### Example 3: Type Validation Warning

```bash
# User specifies type that doesn't match content
/store_memory --type=concept [provides actual implementation code]

# LLM uses specified type but classifier detects mismatch
```
```python
llm_response = {
    "type": "concept",  # User forced this
    "title": "JWT Authentication",
    "content": "```python\ndef validate_jwt(token):\n  ...",  # Implementation code!
    "tags": ["jwt", "authentication"]
}

# Classifier validates
validation = classifier.validate_llm_classification(llm_response)

# Output:
{
    "type": "concept",
    "type_confidence": 0.15,  # LOW confidence ⚠
    "validation_warning": "Type 'concept' seems unusual for this content",
    "suggested_type": "implementation",
    "suggested_confidence": 0.89,
    "tags": ["authentication", "jwt", "security", "python"],
    "importance": "high",
    "completeness": "complete"
}

# Quaid shows warning to user:
print("⚠ Warning: Type 'concept' has low confidence (15%)")
print("  This looks more like 'implementation' (89% confidence)")
print("  Continue anyway? [y/N]")
```

---

## Configuration

**`.quaid/config.toml`**:
```toml
[classification]
# Classification strategy: "hybrid" | "zero-shot-only" | "rule-based-only"
strategy = "hybrid"

# Type classification
[classification.type]
# How to determine type: "user" | "auto" | "hybrid"
# - "user": Require type in frontmatter
# - "auto": Auto-classify always
# - "hybrid": Use frontmatter if present, auto-classify otherwise
mode = "hybrid"

# Validate user-provided type? (warn if confidence < threshold)
validate = true
validation_threshold = 0.3

# Tag classification
[classification.tags]
# Auto-generate tags?
enabled = true

# Max tags per fragment
max_tags = 10

# Minimum confidence for topic tags
min_confidence = 0.3

# Importance classification
[classification.importance]
# How to determine: "auto" | "user" | "hybrid"
mode = "auto"

# Boost importance for admonitions?
boost_from_admonitions = true

# Boost importance for high-impact decisions?
boost_from_decisions = true

# Completeness assessment
[classification.completeness]
# Always rule-based (structural analysis)
enabled = true

# Thresholds (out of 10 points)
complete_threshold = 7
partial_threshold = 4
```

---

## Slash Command Integration

### How Slash Commands Work

```
┌─────────────────────────────────────────────────────────────┐
│              Slash Command → Classification Flow             │
│                                                             │
│  User Input:                                                │
│    /store_memory --type=implementation JWT auth guide      │
│                                                             │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Step 1: Parse Slash Command                      │     │
│  │   - Command: store_memory                         │     │
│  │   - Flags: --type=implementation                  │     │
│  │   - Content: "JWT auth guide"                     │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Step 2: Load & Render Prompt (Promptdown)        │     │
│  │   Template: .quaid/prompts/store_memory.prompt.md │     │
│  │   Variables:                                       │     │
│  │     - {user_message}: "JWT auth guide"            │     │
│  │     - {required_type}: "implementation"           │     │
│  │     - {available_types}: [...all types...]        │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Step 3: LLM Processes Prompt                     │     │
│  │   LLM Response (JSON):                            │     │
│  │   {                                               │     │
│  │     "type": "implementation",                     │     │
│  │     "title": "JWT Authentication Guide",          │     │
│  │     "content": "# JWT Auth\n\n...",              │     │
│  │     "tags": ["jwt", "auth", "security"]           │     │
│  │   }                                               │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Step 4: Validate with Classifier                 │     │
│  │   - Type confidence: 0.89 ✓                       │     │
│  │   - Enhance tags: +["python", "tokens"]           │     │
│  │   - Determine importance: "high"                  │     │
│  │   - Assess completeness: "partial"                │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Step 5: Store Fragment                           │     │
│  │   File: fragments/2025-11-09-jwt-auth-001.md     │     │
│  │   Metadata: fragment.jsonl entry                  │     │
│  │   Index: Tantivy full-text index                 │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Store Memory Prompt Template

**`.quaid/prompts/store_memory.prompt.md`**:
```markdown
# Store Memory Fragment

## Developer Message

You are storing a knowledge fragment in Quaid's memory system. 

CRITICAL REQUIREMENTS:
1. You MUST provide a 'type' field chosen from this exact list: {available_types}
2. If user specified --type={required_type}, you MUST use that type
3. Return ONLY valid JSON with this structure:

{
  "type": "one of: {available_types}",
  "title": "Clear, descriptive title",
  "content": "Full markdown content",
  "summary": "One sentence summary",
  "tags": ["tag1", "tag2", "tag3"]
}

## Conversation

**User:**
{user_message}

{#if required_type}
REQUIRED TYPE: {required_type}
You MUST use this type in your response.
{/if}

**Assistant:**
I'll create a structured fragment with the appropriate type and metadata.
```

### Available Slash Commands

```bash
# Store memory (primary command)
/store_memory [--type=TYPE] <content or conversation>

# Store with explicit type
/store_memory --type=decision We decided to use JWT over sessions

# Store without type (LLM infers)
/store_memory Here's how to implement JWT validation in Python

# Store rule (special type)
/store_rule Never use synchronous database calls in API routes

# Equivalent to:
/store_memory --type=pattern Never use synchronous database calls in API routes
```

## Recommendation: LLM-Required Approach

**Best Configuration**:
```toml
[classification.type]
# Type MUST come from LLM via slash command prompts
# LLM chooses from predefined list or user specifies via --type flag
mode = "llm_required"
validate = true  # Validate LLM's choice with zero-shot classifier

[classification.tags]
enabled = true  # Always auto-generate (LLM suggests, classifier enhances)

[classification.importance]
mode = "auto"  # Auto with structural boosts

[classification.completeness]
enabled = true  # Always rule-based
```

**Why**:
1. **LLM Control** - Type comes from LLM processing slash command
2. **Validation** - Zero-shot classifier validates LLM's choice
3. **Flexibility** - User can force type via `--type` flag in slash command
4. **Consistency** - Same importance/completeness logic across all fragments
5. **No Frontmatter** - Frontmatter is LLM output, not user input

---

## Resource Requirements

**Models**:
- **BART-MNLI**: 1.6GB (~400MB ONNX-optimized)
- **spaCy en_core_web_sm**: 50MB

**Total**: ~450MB models, 1.5GB RAM

**Performance**:
- Type classification: ~100ms
- Tag extraction: ~150ms (zero-shot + NER)
- Importance: ~100ms
- Completeness: <10ms (rule-based)
- **Total per fragment**: ~350-400ms

---

## Conclusion

**LLM-required approach with validation gives you the best of all worlds**:

✅ **Type**: LLM provides (via slash command), zero-shot classifier validates
✅ **Tags**: LLM suggests, classifier enhances with NER + code analysis
✅ **Importance**: Classifier determines with structural boosts
✅ **Completeness**: Fast rule-based structural analysis

**Control Flow**:
```
User → Slash Command → Promptdown Template → LLM Response → Classifier Validation → Storage
```

**Key Insight**: 
- Frontmatter is the **OUTPUT** of LLM processing, not user input
- User controls type via **slash command flags** (`--type=`), not frontmatter
- Classifier **validates** LLM's choice, warns if confidence is low

**Result**: Accurate, consistent classification with LLM control and validation safety nets.

---

**Previous**: [20-Enhanced-Markdown-Structure.md](20-Enhanced-Markdown-Structure.md)

**Version**: 1.0  
**Last Updated**: 2025-11-09
