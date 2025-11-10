# 09 - Config

**Unified Configuration System**

---

## 6. Configuration Unification

### Problem

Multiple configuration examples exist across documents with unclear priority and structure.

### Solution: Single Unified Configuration

**Location**: `~/.quaid/config.toml` (global) + `<project>/.quaid/config.toml` (project-specific)

#### Global Configuration (`~/.quaid/config.toml`)

```toml
[quaid]
version = "1.0.0"
auto_update = true
log_level = "info"

[storage]
backend = "nushell"
index_type = "json"  # or "polars" for large datasets
fragment_format = "markdown"

[ai]
provider = "openai"
default_model = "gpt-4o"
temperature = 0.7

[ai.embedding]
model = "text-embedding-3-small"
dimensions = 1536

[ai.reranker]
enabled = true
model = "jina-reranker-v2-base-multilingual"

[ai.rag]
chunk_size = 512
chunk_overlap = 50
top_k = 5
min_relevance = 0.7

[ai.classification]
model = "gpt-4o"
temperature = 0.3
categories = ["concept", "implementation", "decision", "documentation", "reference"]

[tools]
nushell_path = "~/.quaid/tools/nushell/bin/nu"
mq_path = "~/.quaid/tools/mq/bin/mq"
aichat_path = "~/.quaid/tools/aichat/bin/aichat"

[tools.nushell]
table_mode = "rounded"
use_ansi_coloring = true
edit_mode = "vi"
use_dataframes = true

[tools.mq]
enable_modules = true
default_format = "markdown"
include_csv = true
include_json = true
include_yaml = true

[tools.aichat]
stream = true
save = true
highlight = true

[tools.aichat.document_loaders]
pdf = 'pdftotext $1 -'
docx = 'pandoc --to plain $1'
html = 'pandoc --to plain $1'

[worktree]
auto_detect = true
default_scope = "current"

[worktree.integration]
auto_merge_types = ["concept", "decision", "documentation"]
review_required_types = ["implementation"]
conflict_resolution = "prompt"

[search]
default_limit = 10
highlight_matches = true
show_relevance_scores = true

[graph]
max_depth = 3
relationship_types = ["implements", "references", "depends-on", "related-to"]

[slash_commands]
enabled = true
auto_generate = true
target_tools = ["cursor", "claude", "windsurf", "github-copilot"]
```

#### Project Configuration (`<project>/.quaid/config.toml`)

Project-specific overrides:

```toml
[quaid]
project_name = "my-awesome-project"

[ai]
# Override for project-specific model preferences
default_model = "claude-3-opus"

[ai.rag]
# Project has large context needs
chunk_size = 1024

[search]
# Project-specific defaults
default_tags = ["api", "backend"]

[worktree]
# Project uses feature branch workflow
default_scope = "current"
```

#### Configuration Precedence

1. Project config (`<project>/.quaid/config.toml`)
2. Global config (`~/.quaid/config.toml`)
3. Built-in defaults

#### Configuration Management Commands

```bash
# View current configuration
quaid config show

# Edit global config
quaid config edit

# Edit project config
quaid config edit --project

# Set specific value
quaid config set ai.default_model "gpt-4o"

# Get specific value
quaid config get ai.rag.chunk_size

# Validate configuration
quaid config validate
```

---


---

**Previous**: [08-Slash-Commands.md](08-Slash-Commands.md) | **Next**: [10-Install.md](10-Install.md)
