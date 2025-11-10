# Quaid Workspace Design

**Modular architecture using uv workspaces for maintainable development**

---

## Workspace Overview

```
quaid/                                    # Workspace Root
├── pyproject.toml                        # Main application config
├── uv.lock                              # Shared lockfile for all packages
├── README.md
├── packages/                             # Library packages
│   ├── quaid-core/                       # Core data structures and storage
│   ├── quaid-search/                     # Search engine integration
│   ├── quaid-graph/                      # NetworkX graph operations
│   ├── quaid-nlp/                        # Natural language processing
│   ├── quaid-mcp/                        # MCP server framework
│   ├── quaid-cli/                        # Rich CLI interface
│   ├── quaid-cleanup/                    # Data cleanup and retention
│   └── quaid-config/                     # Configuration management
├── src/                                  # Main application
│   └── quaid/
│       ├── __init__.py
│       ├── main.py                       # Application entry point
│       └── app.py                        # Main orchestration
└── tests/                                # Integration tests
```

---

## Library Breakdown

### 1. quaid-core (Core Library)
**Purpose**: Fundamental data structures, storage layer, and basic operations

**Responsibilities**:
- Fragment data models and validation
- File-based storage operations
- Git-native storage abstraction
- Basic metadata management
- Configuration loading and validation

**Dependencies**: `pydantic`, `pydantic-settings`, `pathlib`

**Key Components**:
```python
# Fragment data model using Pydantic
class Fragment(BaseModel):
    id: str
    title: str
    content: str
    type: FragmentType
    created: datetime
    metadata: Dict[str, Any]

# Storage interface
class StorageBackend:
    def save_fragment(self, fragment: Fragment) -> bool
    def load_fragment(self, fragment_id: str) -> Optional[Fragment]
    def list_fragments(self, filters: Dict) -> List[Fragment]
```

### 2. quaid-search (Search Engine)
**Purpose**: Search functionality using Tantivy and metadata filtering

**Responsibilities**:
- Tantivy index management
- Full-text search operations
- Metadata-based filtering with Polars
- Search result ranking and scoring
- Index maintenance and optimization

**Dependencies**: `quaid-core`, `tantivy>=0.20`, `polars>=0.20`

**Key Components**:
```python
# Search engine interface
class SearchEngine:
    def index_fragment(self, fragment: Fragment) -> None
    def search(self, query: str, filters: Dict) -> List[SearchResult]
    def rebuild_index(self) -> None
    def optimize_index(self) -> None

# Search result model
@dataclass
class SearchResult:
    fragment_id: str
    title: str
    snippet: str
    score: float
    metadata: Dict[str, Any]
```

### 3. quaid-graph (Graph Operations)
**Purpose**: NetworkX-based knowledge graph operations

**Responsibilities**:
- NetworkX graph management
- Relationship detection and storage
- Graph algorithms (centrality, paths, communities)
- Graph visualization and export
- JSONL persistence with atomic operations

**Dependencies**: `quaid-core`, `networkx>=3.2`, `python-louvain>=0.16`

**Key Components**:
```python
# Knowledge graph manager
class KnowledgeGraph:
    def add_relationship(self, rel: Relationship) -> bool
    def find_related_concepts(self, fragment_id: str) -> List[Dict]
    def calculate_centrality(self) -> Dict[str, float]
    def detect_communities(self) -> Dict[str, int]
    def export_visualization(self, format: str) -> bytes

# Relationship model
@dataclass
class Relationship:
    from_id: str
    to_id: str
    relationship: str
    confidence: float
    metadata: Dict[str, Any]
```

### 4. quaid-nlp (Natural Language Processing)
**Purpose**: Text processing, entity extraction, and semantic analysis

**Responsibilities**:
- spaCy NLP operations
- Entity recognition and extraction
- Query intent analysis
- Text similarity scoring
- Keyphrase extraction

**Dependencies**: `quaid-core`, `spacy>=3.7`, `markdown-query>=0.1`

**Key Components**:
```python
# NLP processor
class NLPProcessor:
    def extract_entities(self, text: str) -> Dict[str, List[str]]
    def analyze_query_intent(self, query: str) -> QueryIntent
    def calculate_similarity(self, text1: str, text2: str) -> float
    def extract_keyphrases(self, text: str) -> List[str]

# Intent analysis model
@dataclass
class QueryIntent:
    search_code: bool
    search_concepts: bool
    search_decisions: bool
    entities: List[str]
    key_terms: List[str]
```

### 5. quaid-mcp (MCP Server)
**Purpose**: Model Context Protocol server for AI agent integration

**Responsibilities**:
- FastMCP server implementation
- Tool registration and discovery
- Agent communication handling
- Request/response processing
- Error handling and logging

**Dependencies**: `quaid-core`, `quaid-search`, `quaid-graph`, `fastmcp`

**Key Components**:
```python
# MCP server implementation
class QuaidMCPServer:
    def register_tools(self) -> None
    def handle_search_request(self, params: Dict) -> Dict
    def handle_graph_request(self, params: Dict) -> Dict
    def handle_cleanup_request(self, params: Dict) -> Dict

# Tool interface
class MCPTool:
    def name(self) -> str
    def description(self) -> str
    def execute(self, params: Dict) -> Dict
```

### 6. quaid-cli (Rich CLI Interface)
**Purpose**: Beautiful command-line interface using Typer and Rich

**Responsibilities**:
- Typer CLI command structure
- Rich formatting and styling
- Progress indicators and status displays
- Interactive prompts and confirmations
- Error handling with rich formatting

**Dependencies**: `quaid-core`, `quaid-search`, `quaid-graph`, `typer`, `rich>=13.7`

**Key Components**:
```python
# CLI application
class QuaidCLI:
    def init_command(self, path: str, force: bool) -> None
    def search_command(self, query: str, filters: Dict) -> None
    def add_command(self, title: str, content: str) -> None
    def status_command(self, verbose: bool) -> None

# Rich printer utilities
class RichPrinter:
    def success(self, message: str) -> None
    def error(self, message: str) -> None
    def table(self, data: List[Dict]) -> Table
    def progress(self, operation: str, total: int) -> Progress
```

### 7. quaid-cleanup (Data Cleanup)
**Purpose**: Automated data cleanup and retention management

**Responsibilities**:
- Scheduled cleanup operations using `schedule`
- Retention policy enforcement
- Git repository maintenance
- Storage optimization
- Cleanup logging and monitoring

**Dependencies**: `quaid-core`, `schedule>=1.2`, `GitPython>=3.1`

**Key Components**:
```python
# Cleanup manager
class CleanupManager:
    def start_scheduler(self) -> None
    def run_cleanup(self, cleanup_type: str) -> CleanupResult
    def enforce_retention_policies(self) -> CleanupResult
    def maintain_git_repository(self) -> CleanupResult

# Retention policy model
@dataclass
class RetentionPolicy:
    fragment_type: str
    retention_days: int
    min_count: int
    protected_tags: List[str]
```

### 8. quaid-config (Configuration Management)
**Purpose**: Configuration loading, validation, and management with Pydantic Settings

**Responsibilities**:
- TOML configuration file parsing with Pydantic Settings
- Environment variable interpolation and overrides
- Configuration validation with Pydantic models
- Default value management and validation
- Configuration schema definitions and type safety
- Runtime configuration reloading

**Dependencies**: `pydantic`, `pydantic-settings`, `toml`

**Key Components**:
```python
# Base settings using Pydantic Settings
class BaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid"
    )

# Configuration models
class CleanupConfig(BaseSettings):
    enabled: bool = True
    retention_days: int = 365
    cleanup_schedule: str = "daily"
    default_retention_days: int = 365
    max_fragments_total: int = 10000

    model_config = SettingsConfigDict(env_prefix="QUAID_CLEANUP")

class SearchConfig(BaseSettings):
    index_path: str = ".quaid/memory/indexes/tantivy"
    max_results: int = 10
    enable_semantic_search: bool = False
    tantivy_cache_size_mb: int = 100

    model_config = SettingsConfigDict(env_prefix="QUAID_SEARCH")

class MainSettings(BaseSettings):
    # Core settings
    storage_path: str = ".quaid/memory"
    log_level: str = "INFO"

    # Sub-configurations
    cleanup: CleanupConfig = CleanupConfig()
    search: SearchConfig = SearchConfig()

    # Configuration file path
    config_file: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="QUAID",
        env_file=".env"
    )

# Configuration manager
class ConfigManager:
    def __init__(self, config_file: Optional[str] = None):
        self.settings = MainSettings(config_file=config_file)

    def load_from_file(self, path: str) -> MainSettings:
        """Load configuration from TOML file"""
        with open(path) as f:
            config_data = toml.load(f)
        return MainSettings(**config_data)

    def get_settings(self) -> MainSettings:
        """Get current settings"""
        return self.settings

    def reload(self) -> MainSettings:
        """Reload configuration from file and environment"""
        if self.settings.config_file:
            self.settings = self.load_from_file(self.settings.config_file)
        else:
            self.settings = MainSettings()
        return self.settings
```

# Environment Variable Examples
# QUAID_CLEANUP__RETENTION_DAYS=90
# QUAID_SEARCH__MAX_RESULTS=20
# QUAID_STORAGE_PATH=/path/to/storage
# QUAID_LOG_LEVEL=DEBUG
```

---

## Workspace Configuration

### Root pyproject.toml
```toml
[project]
name = "quaid"
version = "0.1.0"
description = "AI-powered knowledge management system"
requires-python = ">=3.11"
dependencies = [
    "quaid-core",
    "quaid-search",
    "quaid-graph",
    "quaid-nlp",
    "quaid-mcp",
    "quaid-cli",
    "quaid-cleanup",
    "quaid-config"
]

[tool.uv.workspace]
members = [
    "packages/*"
]

[tool.uv.sources]
quaid-core = { workspace = true }
quaid-search = { workspace = true }
quaid-graph = { workspace = true }
quaid-nlp = { workspace = true }
quaid-mcp = { workspace = true }
quaid-cli = { workspace = true }
quaid-cleanup = { workspace = true }
quaid-config = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project.scripts]
quaid = "quaid.main:main"
```

### Package Dependencies Flow
```
quaid-cli
├── quaid-core (required)
├── quaid-search (required)
├── quaid-graph (required)
└── quaid-config (required)

quaid-mcp
├── quaid-core (required)
├── quaid-search (required)
├── quaid-graph (required)
├── quaid-cleanup (required)
└── quaid-config (required)

quaid-search
├── quaid-core (required)
└── quaid-config (required)

quaid-graph
├── quaid-core (required)
└── quaid-config (required)

quaid-nlp
├── quaid-core (required)
└── quaid-config (required)

quaid-cleanup
├── quaid-core (required)
└── quaid-config (required)

quaid-config
└── (no internal dependencies)

quaid-core
└── (no internal dependencies)
```

---

## Development Workflow

### Creating New Packages
```bash
# Create a new package in the workspace
cd packages
uv init quaid-new-feature --lib

# Add to workspace members in root pyproject.toml
# Add to tool.uv.sources in root pyproject.toml
# Add dependency to packages that need it
uv sync
```

### Running Commands
```bash
# Run from workspace root (affects main application)
uv run quaid --help

# Run in specific package
uv run --package quaid-cli python -m quaid_cli

# Run tests for specific package
uv run --package quaid-core pytest

# Sync entire workspace
uv sync

# Add dependency to specific package
cd packages/quaid-search
uv add tantivy>=0.20
```

### Development Benefits

1. **Isolation**: Each library has clear boundaries and responsibilities
2. **Testing**: Individual packages can be tested in isolation
3. **Development**: Teams can work on different packages simultaneously
4. **Dependencies**: Clear dependency graph prevents circular dependencies
5. **Reusability**: Libraries can be reused independently if needed
6. **Maintenance**: Easier to maintain and update individual components
7. **Performance**: Only load what's needed for specific operations

### Package Interaction Patterns

```python
# Main application orchestrates packages
from quaid_core import Fragment, StorageBackend
from quaid_search import SearchEngine
from quaid_graph import KnowledgeGraph
from quaid_config import ConfigManager

class QuaidApplication:
    def __init__(self):
        self.config = ConfigManager().load_config()
        self.storage = StorageBackend(self.config.storage_path)
        self.search = SearchEngine(self.storage, self.config.search)
        self.graph = KnowledgeGraph(self.storage, self.config.graph)

    def add_fragment(self, title: str, content: str) -> str:
        # Core package creates fragment
        fragment = Fragment.create(title, content)

        # Storage saves it
        self.storage.save_fragment(fragment)

        # Search indexes it
        self.search.index_fragment(fragment)

        # Graph processes relationships
        self.graph.process_fragment(fragment)

        return fragment.id
```

This workspace design provides a clean, maintainable architecture that can grow with the complexity of the Quaid system while keeping each component focused and testable.