# Quaid - Refined Vision & Implementation Plan

## Executive Summary

Quaid is an AI-powered memory management tool designed for developers who work with AI coding assistants. It provides a comprehensive, git-native, text-based system for capturing, organizing, and retrieving project knowledge. The system leverages Python and the powerful markdown-query (mq) library with optional AI enhancement for classification, search, and intelligence.

**Core Value Proposition**: A "second brain" for codebases that remembers decisions, tracks patterns, links concepts, and makes knowledge instantly accessible to both developers and AI assistants.

## 1. Vision & Problem Statement

### The Problem
Modern development faces critical knowledge management challenges:
- **Context Loss**: Decisions buried in PRs, Slack, or people's heads
- **Scattered Information**: Code comments, Git history, documentation, and chat are siloed
- **AI Assistant Limitations**: No persistent memory, can't remember project-specific patterns
- **Knowledge Silos**: Different worktrees and branches isolate information

### The Solution
Quaid provides a unified, text-first, git-native system that:
- Remembers decisions and why they were made
- Tracks implementation patterns and best practices
- Links related concepts across projects
- Makes all knowledge instantly accessible to developers and AI assistants

## 2. Architecture Overview

### Core Components
```
Quaid Core Components:
├── Core Application (Python 3.8+)
│   ├── CLI interface (using click/typer)
│   ├── Configuration management
│   ├── File I/O operations
│   └── Command orchestration
│
├── Query Engine (markdown-query Python bindings)
│   ├── Markdown parsing and traversal
│   ├── Pattern matching with selectors (.h1, .code, .link, etc.)
│   ├── Content extraction and transformation
│   ├── HTML processing capabilities
│   └── Structured data output
│
├── Data Layer (Python)
│   ├── JSON/JSONL file operations
│   ├── TOML configuration parsing
│   ├── Index management
│   └── Data validation
│
└── AI Integration (Optional)
    ├── HTTP calls to AI providers (OpenAI, Anthropic, etc.)
    ├── Embedding generation and management
    ├── Classification and tagging
    └── Semantic search capabilities
```

### Key Architectural Principles
1. **Text-First**: All data stored as human-readable text (Markdown, JSONL, TOML)
2. **Git-Native**: Everything version-controlled, no binary databases
3. **Python-Powered**: Leverage Python's rich ecosystem and markdown-query
4. **No Servers**: Pure CLI, no runtime daemons or protocols
5. **AI-Augmented**: AI enhances but doesn't replace structured data
6. **Configuration-Driven**: Behavior adapts based on configuration
7. **Extensible**: Plugin architecture for custom processors

## 3. Directory Structure

```
project-root/
├── .quaid/
│   ├── config.toml              # Project-specific configuration
│   │
│   ├── memory/
│   │   ├── fragments/           # Individual memory files
│   │   │   ├── 20250108-auth-concept.md
│   │   │   ├── 20250108-jwt-implementation.md
│   │   │   └── 20250109-api-decision.md
│   │   │
│   │   ├── indexes/             # Index files (JSONL format)
│   │   │   ├── fragments.jsonl  # Fragment metadata index
│   │   │   ├── tags.jsonl       # Tag taxonomy and counts
│   │   │   └── graph.jsonl      # Knowledge graph relationships
│   │   │
│   │   └── embeddings/          # RAG data (if AI enabled)
│   │       ├── vectors.db       # Vector embeddings
│   │       └── metadata.json
│   │
│   ├── processors/              # Custom Python processors
│   │   ├── queries/             # Saved query processors
│   │   └── workflows/           # Custom automation scripts
│   │
│   └── docs/                    # Documentation and templates
│       ├── prd.txt
│       └── research/
│
└── ... (other project files)
```

### Fragment Format
Each memory fragment is a Markdown file with YAML frontmatter:

```markdown
---
id: "20250108-auth-001"
type: concept
tags: [authentication, jwt, security]
created: 2025-01-08T12:00:00Z
updated: 2025-01-08T12:00:00Z
worktree: main
related: [20250108-jwt-impl-002, 20250107-security-decision-003]
confidence: 0.95
---

# JWT Authentication Concept
...
```

## 4. CLI API Specification

### quaid init [project-name]
Initialize Quaid for a project
- Creates `.quaid/` directory structure
- Generates `config.toml` 
- Creates index files in `memory/indexes/`
- Detects AI tools and generates slash commands

### quaid store [content|file|clipboard]
Capture and store information with optional AI classification
- Creates a fragment with YAML frontmatter
- Auto-classifies if AI configured using markdown-query processing
- Updates indexes automatically

### quaid search <query>
Multi-stage search with pattern matching using markdown-query and optional AI ranking
- Pattern matching via Python markdown-query library
- Extract content using selectors (.h1, .code, .link, etc.)
- Optional semantic search via AI
- Merged and ranked results

### quaid get <id>
Retrieve specific fragment by ID

### quaid list
List fragments with filtering options (--type, --tags, --worktree)

### quaid classify [file|content]
AI-powered classification and tagging using Python HTTP clients

### quaid sync
Sync memories with git

### quaid backup/restore/export/import
Advanced data management features

### quaid query <selector>
Direct markdown-query operations on fragments
- Example: `quaid query ".h1 | to_text()"` to extract all headings
- Example: `quaid query ".code | to_text()"` to extract all code blocks
- Example: `quaid query "select(or(.h1, .code))"` to get both headings and code

## 5. Configuration System

### Global Configuration (`~/.quaid/config.toml`)
```toml
[quaid]
version = "1.0.0"
auto_update = true
log_level = "info"

[storage]
backend = "nushell"
index_type = "jsonl"  # or "polars" for large datasets
fragment_format = "markdown"

[ai]
provider = "openai"  # or anthropic, etc.
default_model = "gpt-4o"
temperature = 0.7

[ai.embedding]
model = "text-embedding-3-small"
dimensions = 1536

[ai.reranker]
enabled = true
model = "jina-reranker-v2-base-multilingual"

[tools]
nushell_path = "~/.quaid/tools/nushell/bin/nu"
mq_path = "~/.quaid/tools/mq/bin/mq"
```

## 6. Multi-Worktree Support

Memory isolation strategy:
- Main worktree: Stable, production memories
- Feature worktrees: Experimental, context-specific
- Integration: Selective merge with conflict resolution

Worktree-specific queries supported via `--worktree` flag.

## 7. Knowledge Graph

Relationship tracking between fragments:
- Implements: Code implements concepts
- References: Fragment references another
- Depends-on: Fragment depends on another
- Auto-generated and manual relationships

## 8. AI Features (Optional)

When AI provider configured:
- Zero-shot classification of fragments
- Semantic search and ranking
- Auto-tagging based on content
- Context-aware reranking

## 9. Installation & Distribution

Distributed as a Rust binary with cargo-dist:
- Single binary distribution
- Auto-download of bundled tools (NuShell, mq)
- Package manager installation (Cargo, NPM, Homebrew)
- First-run auto-setup

## 10. Implementation Roadmap

### Phase 0: Foundation (Weeks 1-2)
- [ ] Create Python project structure with pyproject.toml/pip
- [ ] Set up development environment and dependencies
- [ ] Implement configuration management system (TOML files using toml library)
- [ ] Install and integrate markdown-query Python bindings (pip install markdown-query)
- [ ] Implement `quaid init` command with directory structure creation
- [ ] Set up basic storage system (Markdown fragments with YAML frontmatter)
- [ ] Implement index management using JSONL format
- [ ] Create basic CLI framework (using click or typer)
- [ ] Implement first-run auto-setup with dependency checks

**Deliverables**:
- Working `quaid init` command
- Python package with markdown-query integration
- Basic project structure creation
- Configuration file management

### Phase 1: Core Memory Operations (Weeks 3-5)
- [ ] Implement `quaid store` command with multiple input methods
  - [ ] Manual content input
  - [ ] File input (`--file` flag)
  - [ ] Clipboard input (`--clipboard` flag)
  - [ ] Type specification (`--type` flag)
  - [ ] Tag specification (`--tags` flag)
- [ ] Implement fragment ID generation (timestamp-based)
- [ ] Implement YAML frontmatter generation using pyyaml
- [ ] Implement `quaid get <id>` command
- [ ] Implement `quaid list` command with filtering
  - [ ] Filter by type (`--type` flag)
  - [ ] Filter by tags (`--tags` flag)
  - [ ] Filter by worktree (`--worktree` flag)
  - [ ] Limit results (`--limit` flag)
- [ ] Implement index auto-update (JSONL append)
- [ ] Implement multi-worktree detection and metadata
- [ ] Implement `quaid search` with basic pattern matching using markdown-query
- [ ] Create Python classes for fragment management
- [ ] Add comprehensive error handling

**Deliverables**:
- Complete CRUD operations for fragments
- Pattern-based search functionality using markdown-query
- Multi-worktree support
- Index management system

### Phase 2: Query Enhancement & Search (Weeks 6-7)
- [ ] Implement advanced markdown-query operations
  - [ ] Selector-based content extraction using mq selectors (.h1, .code, .link, etc.)
  - [ ] Text transformation operations (to_text(), upcase(), etc.)
  - [ ] Complex query composition
- [ ] Implement `quaid query <selector>` command
  - [ ] Direct access to markdown-query functionality
  - [ ] Interactive query mode
  - [ ] Query history and saved queries
- [ ] Implement multi-stage search pipeline
  - [ ] Pattern matching stage with markdown-query
  - [ ] Result merging and deduplication
  - [ ] Basic ranking by relevance
- [ ] Implement advanced search filters
  - [ ] Date ranges (`--since`, `--until`)
  - [ ] Content type filtering
  - [ ] Complexity filtering
- [ ] Implement structured output formats (table, JSON, CSV)
- [ ] Add highlighting to search results
- [ ] Implement `quaid related --to <fragment-id>` command

**Deliverables**:
- Advanced markdown-query integration
- Direct query command access
- Multi-stage search pipeline
- Advanced filtering and ranking
- Structured output options

### Phase 3: AI Integration (Weeks 8-10)
- [ ] Implement AI provider abstraction layer
  - [ ] OpenAI support using openai Python library
  - [ ] Anthropic support using anthropic Python library
  - [ ] Local model support (Ollama, LiteLLM, etc.)
- [ ] Create Python HTTP clients for AI API calls
  - [ ] Async support using aiohttp
  - [ ] Retry logic with backoff
  - [ ] Error handling and circuit breakers
  - [ ] Rate limiting
- [ ] Implement zero-shot classification
  - [ ] Classification prompt templates
  - [ ] Confidence scoring
  - [ ] Batch processing
- [ ] Implement auto-tagging system using AI and markdown-query
- [ ] Add AI configuration to TOML files
- [ ] Implement semantic search (optional AI ranking)
- [ ] Create reranking functionality
- [ ] Add cost tracking and optimization
- [ ] Implement embedding management for RAG

**Deliverables**:
- Configurable AI provider system
- Classification and auto-tagging
- Semantic search capabilities
- Cost-aware operations
- RAG system implementation

### Phase 4: Knowledge Graph & Advanced Features (Weeks 11-12)
- [ ] Implement knowledge graph with JSONL storage
  - [ ] Node management
  - [ ] Edge detection using markdown-query pattern matching
  - [ ] Relationship types configuration
  - [ ] Graph traversal algorithms
- [ ] Implement graph query operations
- [ ] Implement backup and sync features
  - [ ] Auto-commit functionality
  - [ ] Backup archiving
  - [ ] Restore capability
- [ ] Implement import/export functionality
  - [ ] Obsidian vault import using markdown-query for processing
  - [ ] Notion export
  - [ ] Generic JSON format support
  - [ ] HTML import using markdown-query's HTML processing capabilities
- [ ] Advanced analytics and statistics
- [ ] Performance monitoring

**Deliverables**:
- Knowledge graph system with relationship detection
- Advanced data management
- Import/export capabilities with format conversion
- Analytics and monitoring

### Phase 5: Integration & Polish (Weeks 13-14)
- [ ] Generate slash commands for AI tools
  - [ ] Cursor integration (`/.cursor/commands/`)
  - [ ] Claude integration (`/.claude/commands/`)
  - [ ] Cline integration (`/.cline/commands/`)
  - [ ] Other tool integrations
- [ ] Implement performance optimizations
  - [ ] Caching strategies using functools/cache
  - [ ] Lazy loading for large datasets
  - [ ] Index partitioning for large datasets
  - [ ] Async processing for I/O operations
- [ ] Add comprehensive error handling and user feedback
- [ ] Create comprehensive documentation
- [ ] Implement analytics and usage tracking
- [ ] Add comprehensive testing (unit, integration, end-to-end)
- [ ] Prepare package for distribution (PyPI, pip install)
- [ ] Create comprehensive test suite
- [ ] Performance benchmarking

**Deliverables**:
- AI tool integrations
- Performance optimizations
- Complete documentation
- Comprehensive test coverage
- Production-ready PyPI package

## 11. Critical Success Factors

1. **Text-first approach**: All data remains human-readable
2. **Git-native**: Seamless integration with version control
3. **Tool synergy**: Clear separation of concerns between NuShell and mq
4. **Configuration-driven**: AI features optional and configurable
5. **Zero infrastructure**: Pure CLI, no servers or daemons
6. **Developer-friendly**: Intuitive commands and clear documentation

## 12. Differentiation

- **Not a note-taking app**: Specifically designed for codebase knowledge
- **Not a search engine**: Git-native with structured metadata
- **Not AI-first**: AI augments, doesn't replace structured data
- **Not server-based**: Pure CLI with no runtime dependencies