# 22 - Slash Commands Design

**CLI Commands that Trigger LLM Prompts for Fragment Management**

---

## Executive Summary

Quaid uses **slash commands** as the primary interface for storing and managing fragments. Each slash command:

1. **Parses** user input and flags
2. **Loads** appropriate Promptdown template
3. **Renders** prompt with template values
4. **Calls** LLM (local via Ollama or API)
5. **Validates** LLM response with classifiers
6. **Stores** fragment with metadata

**Key Principle**: Users interact via commands, LLM handles structure/classification.

---

## Command Structure

### Basic Syntax

```bash
/command [--flag=value] <content>
```

**Examples**:
```bash
/store_memory JWT authentication implementation
/store_memory --type=decision We chose JWT over sessions
/store_rule Never use blocking I/O in async functions
/recall authentication patterns
/search --tag=jwt --type=implementation
```

---

## Core Commands

### 1. `/store_memory` - Store Knowledge Fragment

**Purpose**: Store any knowledge, code, decision, or reference

**Syntax**:
```bash
/store_memory [--type=TYPE] [--importance=LEVEL] <content>
```

**Flags**:
- `--type=TYPE` - Force specific type (concept|implementation|decision|reference|pattern|troubleshooting|api-doc)
- `--importance=LEVEL` - Force importance (high|medium|low)
- `--tags=tag1,tag2` - Add additional tags

**Examples**:
```bash
# Implicit type (LLM infers)
/store_memory Here's how to validate JWT tokens in Python

# Explicit type
/store_memory --type=implementation JWT token validation with PyJWT

# With multiple flags
/store_memory --type=decision --importance=high We will use Redis for session storage

# Long-form content (multi-line)
/store_memory --type=troubleshooting
User: JWT tokens keep expiring too quickly
Solution: Increased token TTL from 15min to 1hr in config
```

**LLM Prompt** (`.quaid/prompts/store_memory.prompt.md`):
```markdown
# Store Memory Fragment

## Developer Message

You are storing a knowledge fragment. Return ONLY valid JSON:

{
  "type": "concept|implementation|decision|reference|pattern|troubleshooting|api-doc",
  "title": "Clear, descriptive title (max 60 chars)",
  "content": "Full markdown content with code blocks, examples, etc.",
  "summary": "One sentence summary (max 120 chars)",
  "tags": ["tag1", "tag2", "tag3"]
}

{#if required_type}
REQUIRED TYPE: {required_type}
{/if}

{#if required_importance}
REQUIRED IMPORTANCE: {required_importance}
{/if}

## Conversation

**User:**
{user_message}

**Assistant:**
I'll create a structured fragment with appropriate metadata.
```

**Flow**:
```
User: /store_memory --type=implementation JWT validation
       ↓
Quaid: Parse command, extract flags
       ↓
Quaid: Load store_memory.prompt.md
       ↓
Quaid: Render prompt with {user_message: "JWT validation", required_type: "implementation"}
       ↓
LLM:   Returns {"type": "implementation", "title": "JWT Token Validation", ...}
       ↓
Quaid: Validate with HybridClassifier
       ↓
Quaid: Store fragment + index in Tantivy
       ↓
Quaid: "✓ Stored: JWT Token Validation (fragments/2025-11-09-jwt-001.md)"
```

---

### 2. `/store_rule` - Store Project Rule/Guideline

**Purpose**: Store coding rules, best practices, constraints

**Syntax**:
```bash
/store_rule <rule description>
```

**Examples**:
```bash
/store_rule Never use synchronous database calls in API routes
/store_rule All API responses must include request_id field
/store_rule Use pytest fixtures for database setup in tests
```

**Behavior**:
- Automatically sets `type=pattern`
- Automatically sets `importance=high`
- Adds `rule` tag
- Stores in special `rules/` category for easy retrieval

**LLM Prompt** (`.quaid/prompts/store_rule.prompt.md`):
```markdown
# Store Rule/Guideline

## Developer Message

You are storing a project rule or coding guideline. This is a MUST-FOLLOW constraint.

Return ONLY valid JSON:
{
  "type": "pattern",
  "title": "Rule: <concise rule statement>",
  "content": "# Rule\n\n{rule_text}\n\n## Rationale\n\n{why this rule exists}\n\n## Examples\n\n{good/bad examples}",
  "importance": "high",
  "tags": ["rule", ...other relevant tags...]
}

## Conversation

**User:**
Rule: {rule_text}

**Assistant:**
I'll create a structured rule fragment with rationale and examples.
```

---

### 3. `/recall` - Search and Retrieve Fragments

**Purpose**: Find relevant fragments using natural language or filters

**Syntax**:
```bash
/recall [--type=TYPE] [--tag=TAG] [--importance=LEVEL] <query>
```

**Examples**:
```bash
# Natural language query
/recall JWT authentication patterns

# Filtered search
/recall --type=implementation --tag=jwt authentication

# Multiple filters
/recall --type=decision --importance=high authentication choices

# Recall all rules
/recall --tag=rule
```

**Flow**:
```
User: /recall JWT authentication
       ↓
Quaid: Parse query
       ↓
Quaid: Search with Tantivy (full-text)
       ↓
Quaid: Filter by type/tags if specified
       ↓
Quaid: Rerank with FlashRank
       ↓
Quaid: Apply structural scoring
       ↓
Quaid: Return top 5-10 results
       ↓
Quaid: Display results:
       "Found 8 fragments:
        1. [implementation] JWT Token Validation (85% match)
        2. [decision] JWT vs Sessions Decision (78% match)
        3. [pattern] JWT Best Practices Rule (75% match)
        ..."
```

---

### 4. `/update_memory` - Update Existing Fragment

**Purpose**: Update or append to existing fragment

**Syntax**:
```bash
/update_memory <fragment_id> [--append] <new content>
```

**Examples**:
```bash
# Replace content
/update_memory jwt-001 Updated implementation to use RS256 instead of HS256

# Append to existing
/update_memory jwt-001 --append Added note about key rotation every 90 days
```

---

### 5. `/delete_memory` - Delete Fragment

**Purpose**: Remove fragment from memory

**Syntax**:
```bash
/delete_memory <fragment_id>
```

**Example**:
```bash
/delete_memory jwt-001
```

---

### 6. `/list_memories` - List All Fragments

**Purpose**: Browse all stored fragments

**Syntax**:
```bash
/list_memories [--type=TYPE] [--tag=TAG] [--sort=FIELD]
```

**Examples**:
```bash
# List all
/list_memories

# Filter by type
/list_memories --type=implementation

# Sort by importance
/list_memories --sort=importance
```

---

### 7. `/analyze` - Analyze Project/Codebase

**Purpose**: Analyze current codebase and suggest fragments to create

**Syntax**:
```bash
/analyze [--path=PATH]
```

**Examples**:
```bash
# Analyze current directory
/analyze

# Analyze specific path
/analyze --path=src/auth
```

**Flow**:
```
User: /analyze --path=src/auth
       ↓
Quaid: Scan directory for code files
       ↓
Quaid: Use markdown-query to extract structure
       ↓
Quaid: Identify patterns, implementations, potential decisions
       ↓
Quaid: Suggest fragments to create:
       "Found 3 potential fragments:
        1. [implementation] JWT Token Generation (auth/jwt.py)
        2. [pattern] Error Handling Pattern (auth/errors.py)
        3. [decision] Auth Strategy Choice (auth/README.md)
        
        Store all? [y/N]"
```

---

## Command Processing Pipeline

### Universal Command Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Slash Command Processing Pipeline               │
│                                                             │
│  User Input (CLI/Chat):                                     │
│    /store_memory --type=implementation JWT guide           │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Step 1: Command Parser                           │     │
│  │   - Identify command: store_memory                │     │
│  │   - Extract flags: {type: "implementation"}       │     │
│  │   - Extract content: "JWT guide"                  │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Step 2: Prompt Manager                           │     │
│  │   - Load: .quaid/prompts/{command}.prompt.md      │     │
│  │   - Apply template values:                        │     │
│  │     {user_message, required_type, ...}            │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Step 3: LLM Processor                            │     │
│  │   - Call LLM (Ollama/API)                         │     │
│  │   - Parse JSON response                           │     │
│  │   - Validate schema                               │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Step 4: Classification & Validation              │     │
│  │   - Validate type with zero-shot classifier       │     │
│  │   - Enhance tags with NER                         │     │
│  │   - Determine importance & completeness           │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Step 5: Storage & Indexing                       │     │
│  │   - Generate fragment ID                          │     │
│  │   - Write markdown file                           │     │
│  │   - Update fragment.jsonl                         │     │
│  │   - Index in Tantivy                              │     │
│  │   - Update knowledge graph                        │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  Response to User:                                          │
│    ✓ Stored: JWT Token Validation                          │
│    📄 fragments/2025-11-09-jwt-001.md                      │
│    🏷  Tags: jwt, authentication, security, python          │
│    📊 Type: implementation (89% confidence)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Command Handler (Python)

```python
from typing import Dict, Any, Optional
from pathlib import Path
import re

class SlashCommandHandler:
    """
    Handle slash commands and route to appropriate processors
    """
    
    def __init__(self, config: dict):
        self.prompt_manager = PromptManager(Path(".quaid/prompts"))
        self.llm_processor = LLMProcessor(config)
        self.classifier = HybridClassifier()
        self.storage = FragmentStorage(Path(".quaid"))
    
    def handle(self, command_str: str) -> Dict[str, Any]:
        """
        Parse and execute slash command
        """
        # Parse command
        cmd = self.parse_command(command_str)
        
        # Route to handler
        if cmd['command'] == 'store_memory':
            return self.handle_store_memory(cmd)
        elif cmd['command'] == 'store_rule':
            return self.handle_store_rule(cmd)
        elif cmd['command'] == 'recall':
            return self.handle_recall(cmd)
        # ... other commands
        else:
            raise ValueError(f"Unknown command: {cmd['command']}")
    
    def parse_command(self, command_str: str) -> Dict[str, Any]:
        """
        Parse slash command into structured dict
        
        Example:
        "/store_memory --type=implementation JWT guide"
        → {
            "command": "store_memory",
            "flags": {"type": "implementation"},
            "content": "JWT guide"
        }
        """
        # Remove leading slash
        command_str = command_str.lstrip('/')
        
        # Split into parts
        parts = command_str.split()
        
        # First part is command
        command = parts[0]
        
        # Extract flags (--key=value)
        flags = {}
        content_parts = []
        
        for part in parts[1:]:
            if part.startswith('--'):
                # Parse flag
                flag_match = re.match(r'--(\w+)=(.+)', part)
                if flag_match:
                    key, value = flag_match.groups()
                    flags[key] = value
            else:
                content_parts.append(part)
        
        return {
            "command": command,
            "flags": flags,
            "content": " ".join(content_parts)
        }
    
    def handle_store_memory(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle /store_memory command
        """
        # Get flags
        required_type = cmd['flags'].get('type')
        required_importance = cmd['flags'].get('importance')
        user_tags = cmd['flags'].get('tags', '').split(',') if cmd['flags'].get('tags') else []
        
        # Load and render prompt
        messages = self.prompt_manager.get_messages(
            "store_memory",
            template_values={
                "user_message": cmd['content'],
                "required_type": required_type,
                "required_importance": required_importance,
                "available_types": FRAGMENT_TYPES
            }
        )
        
        # Call LLM
        llm_response = self.llm_processor.generate(messages)
        
        # Parse JSON response
        try:
            fragment_data = json.loads(llm_response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            fragment_data = self.extract_json(llm_response)
        
        # Validate and enhance with classifier
        classification = self.classifier.classify(
            fragment_data,
            user_provided_type=required_type
        )
        
        # Merge classification results
        fragment_data.update(classification)
        
        # Add user tags
        fragment_data['tags'] = list(set(fragment_data['tags'] + user_tags))
        
        # Store fragment
        fragment_id = self.storage.store_fragment(fragment_data)
        
        return {
            "success": True,
            "fragment_id": fragment_id,
            "fragment_data": fragment_data
        }
    
    def handle_store_rule(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle /store_rule command (shortcut for pattern type)
        """
        # Force type and importance
        cmd['flags']['type'] = 'pattern'
        cmd['flags']['importance'] = 'high'
        cmd['flags']['tags'] = 'rule'
        
        # Use store_memory handler
        return self.handle_store_memory(cmd)
    
    def handle_recall(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle /recall command
        """
        # Extract filters
        filters = {
            'type': cmd['flags'].get('type'),
            'tag': cmd['flags'].get('tag'),
            'importance': cmd['flags'].get('importance')
        }
        
        # Search
        search_engine = SearchEngine(self.storage)
        results = search_engine.search(
            query=cmd['content'],
            filters=filters,
            top_k=10
        )
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
```

---

## Configuration

**`.quaid/config.toml`**:
```toml
[commands]
# Enable slash commands
enabled = true

# Command prefix (default: "/")
prefix = "/"

# LLM for command processing
[commands.llm]
# Provider: "ollama" | "openai" | "anthropic"
provider = "ollama"

# Model for command processing
model = "qwen2.5-coder:7b"

# Temperature for command processing
temperature = 0.3

# Validation
[commands.validation]
# Validate LLM responses with classifier?
enabled = true

# Warn if type confidence below threshold
type_confidence_threshold = 0.3
```

---

## Conclusion

**Slash commands provide a natural CLI interface** that:

✅ **Triggers LLM prompts** using Promptdown templates
✅ **Controls classification** via command flags (--type, --importance)
✅ **Validates responses** with zero-shot classifier
✅ **Stores structured fragments** with rich metadata
✅ **No frontmatter editing** - frontmatter is LLM output

**The Flow**:
```
User Command → Parse → Prompt Template → LLM → Validate → Store
```

This design makes Quaid feel like a **conversational knowledge base** while maintaining strict structure and validation.

---

**Previous**: [21-Hybrid-Classification.md](21-Hybrid-Classification.md)

**Version**: 1.0  
**Last Updated**: 2025-11-09
