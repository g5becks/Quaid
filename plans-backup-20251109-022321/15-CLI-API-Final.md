# 15 - CLI API Final Specification

**Definitive CLI commands, configuration, and implementation details**

---

## Core Principles

1. **Slash commands match CLI commands** - `/quaid-store` → `quaid store`
2. **Configuration-driven behavior** - If AI is configured, use it; if not, skip
3. **Single API calls** - Classify + tag in one request with structured output
4. **Lazy loading** - Use JSONL with polars lazy dataframes for large datasets
5. **Environment interpolation** - Support `#{VAR}` syntax in config files

---

## Complete CLI Reference

### quaid init <project-name>

**Purpose**: Initialize Quaid for a project

**What it does**:
1. Creates `.quaid/` directory structure
2. Generates `config.toml` with project name
3. Creates `memory/indexes/` directory (not single index file)
4. Initializes empty JSONL indexes (fragments.jsonl, tags.jsonl)
5. Creates `memory/fragments/` directory
6. Creates `context/` directory with `current.md`
7. Detects AI tools (Cursor, Claude, Windsurf, Cline, etc.)
8. Generates tool-specific slash command files:
   - `.cursor/commands/quaid-*.md`
   - `.claude/commands/quaid/*.md`
   - `.windsurf/workflows/quaid-*.md`
   - `.cline/commands/quaid-*.md`

**Example**:
```bash
$ cd ~/my-project
$ quaid init my-awesome-project

✓ Created .quaid/ directory structure
✓ Initialized indexes: memory/indexes/
✓ Generated config: .quaid/config.toml
✓ Detected AI tools: Cursor, Cline
✓ Generated slash commands:
  - .cursor/commands/quaid-store.md
  - .cursor/commands/quaid-recall.md
  - .cline/commands/quaid-store.md
  - .cline/commands/quaid-recall.md

Next steps:
  1. Edit config: quaid config edit
  2. Configure AI provider, model, and API key env var
  3. Define tags (optional): edit .quaid/config.toml [tags] section
  4. Export API key: export OPENAI_API_KEY=sk-...
```

---

### quaid store <type> <content>

**Purpose**: Store long-term knowledge fragment

**Signature**:
```bash
quaid store <type> <content> [--tags TAG,TAG] [--no-classify]
quaid store <type> --file <path> [--tags TAG,TAG] [--no-classify]
cat file.md | quaid store <type> [--tags TAG,TAG] [--no-classify]
```

**Arguments**:
- `<type>` (required): One of: `rule`, `concept`, `implementation`, `decision`, `documentation`, `reference`
- `<content>`: Content to store (exclusive with `--file`)
- `--file <path>`: File to store (exclusive with content)
- `--tags <tags>`: Override auto-tagging with specific tags
- `--no-classify`: Skip AI classification even if configured

**Behavior**:
- **If AI configured**: Single API call to classify + extract tags (unless `--no-classify`)
- **If AI not configured**: Use provided type and tags only
- **Always**: Append to `memory/indexes/fragments.jsonl` with polars

**Classification API Call** (single request):
```nu
# NuShell HTTP - structured output request
let prompt = $"Analyze this content and return JSON:
{
  \"type\": \"rule|concept|implementation|decision|documentation|reference\",
  \"tags\": [\"tag1\", \"tag2\", \"tag3\"],
  \"summary\": \"one sentence summary\"
}

Content: ($content)"

let response = (http post $api_url --headers $headers {
    model: $model
    messages: [{role: "user", content: $prompt}]
    response_format: {type: "json_object"}  # Structured output
} | from json)

# Parse JSON response
let classification = ($response.choices.0.message.content | from json)
```

**Examples**:
```bash
# Store a rule
quaid store rule "Never use .ts extension for config files"

# Store from file
quaid store documentation --file ./docs/architecture.md

# Store with specific tags (skip AI tagging)
quaid store decision --tags database,postgres "Use PostgreSQL for main DB"

# Store without classification (manual only)
quaid store concept --no-classify "MVC pattern separates concerns"

# From stdin
git log -1 --pretty=%B | quaid store reference
```

**Slash command equivalent**: `/quaid-store <type> <content>`

---

### quaid recall <query>

**Purpose**: Search long-term knowledge

**Signature**:
```bash
quaid recall <query> [--type TYPE] [--tags TAG,TAG] [--skip-rerank]
```

**Arguments**:
- `<query>` (required): Search query
- `--type <type>`: Filter by type
- `--tags <tags>`: Filter by tags (comma-separated)
- `--skip-rerank`: Skip AI reranking even if configured

**Behavior**:
- **Always**: Search with mq + filter with NuShell polars
- **If AI configured**: Rerank results by relevance (unless `--skip-rerank`)
- **If AI not configured**: Return results sorted by date

**Search pipeline**:
```nu
# 1. mq content search
let mq_results = (mq '.h1, .h2, .p' .quaid/memory/fragments/*.md 
  | lines 
  | where ($it | str contains -i $query))

# 2. NuShell polars filter
let index_results = (polars open .quaid/memory/indexes/fragments.jsonl
  | polars filter (
      (polars col tags | polars is-in $search_tags) or
      (polars col type) == $search_type
  )
  | polars into-nu)

# 3. Merge
let combined = ($mq_results | append $index_results | uniq-by id)

# 4. AI rerank (if configured)
if $config.ai.rerank_model? and not $skip_rerank {
    let ranked = (nu scripts/rerank.nu $query $combined)
} else {
    $combined | sort-by created --reverse
}
```

**Examples**:
```bash
# Basic search (with AI reranking if configured)
quaid recall "authentication"

# Filter by type
quaid recall "database" --type decision

# Filter by tags
quaid recall "api" --tags rest,graphql

# Skip AI reranking
quaid recall "auth" --skip-rerank
```

**Slash command equivalent**: `/quaid-recall <query>`

---

### quaid context append <note>

**Purpose**: Add to current session context

**Signature**:
```bash
quaid context append <note>
cat file.md | quaid context append
```

**Behavior**:
- Appends timestamped note to `.quaid/context/current.md`
- **If AI configured + context size > threshold**: Auto-summarize and archive
- **If AI not configured**: Just append, manual archive

**Auto-summarization** (when context exceeds configured threshold):
```nu
# Check size
let current_size = (open .quaid/context/current.md | str length)

if $current_size > $config.context.max_size {
    # Summarize with AI
    let summary = (cat .quaid/context/current.md | nu scripts/summarize.nu)
    
    # Archive
    let session_id = (date now | format date "%Y-%m-%d-%H%M")
    $summary | save $".quaid/context/($session_id).md"
    
    # Clear current
    "" | save .quaid/context/current.md
}
```

**Examples**:
```bash
quaid context append "Implemented JWT middleware in src/auth/"
quaid context append "Bug: token expiry not checking timezone"
```

**Slash command equivalent**: `/quaid-context <note>`

---

### quaid context load [--recent N]

**Purpose**: Load recent session context

**Signature**:
```bash
quaid context load [--recent N] [--full]
```

**Arguments**:
- `--recent <N>`: Number of sessions to load (default: 3)
- `--full`: Load full sessions, not summaries

**Examples**:
```bash
# Load last 3 session summaries
quaid context load

# Load last 5 sessions
quaid context load --recent 5
```

**Slash command equivalent**: `/quaid-context-load [--recent N]`

---

### quaid tags list [--sort-by count|name]

**Purpose**: List all defined tags

**Signature**:
```bash
quaid tags list [--sort-by count|name]
quaid tags search <pattern>
```

**Tags are pre-defined in config**:
```toml
[tags.authentication]
description = "Authentication and authorization mechanisms"
examples = [
  "JWT token validation in middleware",
  "OAuth2 flow implementation",
  "Session management with Redis"
]

[tags.database]
description = "Database operations and schema"
examples = [
  "PostgreSQL migration scripts",
  "ORM model definitions",
  "Database connection pooling"
]
```

**Classification uses these tags**:
```nu
# Use Cohere classify API with predefined tags
let tag_examples = ($config.tags | transpose name data | each { |tag|
    {
        label: $tag.name
        examples: $tag.data.examples
    }
})

# Classify with examples
curl --request POST \
  --url https://api.cohere.com/v1/classify \
  --header "Authorization: bearer #{COHERE_API_KEY}" \
  --data {
    model: "embed-english-v3.0"
    inputs: [$content]
    examples: $tag_examples
  }
```

---

### quaid config edit [project-name]

**Purpose**: Edit configuration file

**Signature**:
```bash
quaid config edit [project-name]
```

**Behavior**:
- If `project-name`: Opens `.quaid/config.toml` in that project
- If no arg in project dir: Opens `.quaid/config.toml`
- If no arg outside project: Opens `~/.quaid/config.toml` (global)

**Config precedence**:
1. Project config (`.quaid/config.toml`)
2. Global config (`~/.quaid/config.toml`)
3. Built-in defaults

---

## Configuration File Structure

### Complete config.toml

```toml
# .quaid/config.toml
# Edit with: quaid config edit

[project]
name = "my-awesome-project"
initialized = "2025-11-08T12:00:00Z"

# Optional .env file (merged into environment)
dot_env = ".env"  # or "/absolute/path/to/.env"

# ============================================================================
# AI Configuration
# ============================================================================

[ai]
# Provider: openai, anthropic, cohere, openrouter
provider = "openai"
model = "gpt-4"

# API key from environment (supports interpolation)
api_key = "#{OPENAI_API_KEY}"

# API endpoint (supports interpolation)
api_url = "https://api.openai.com/v1/chat/completions"

max_tokens = 4000
temperature = 0.7

# Classification & Tagging (single API call)
[ai.classify]
enabled = true
model = "gpt-4"  # Can use different model
api_key = "#{OPENAI_API_KEY}"

# Reranking
[ai.rerank]
enabled = true
model = "gpt-4"
api_key = "#{OPENAI_API_KEY}"

# Context summarization
[ai.summarize]
enabled = true
model = "gpt-4"
api_key = "#{OPENAI_API_KEY}"
max_words = 200

# Alternative: Use Cohere for classification with examples
# [ai.classify]
# enabled = true
# provider = "cohere"
# model = "embed-english-v3.0"
# api_key = "#{COHERE_API_KEY}"
# api_url = "https://api.cohere.com/v1/classify"
# use_tag_examples = true  # Use examples from [tags.*]

# ============================================================================
# Storage Configuration
# ============================================================================

[storage]
fragment_dir = ".quaid/memory/fragments"
indexes_dir = ".quaid/memory/indexes"  # Note: plural!

# Auto-behaviors (only if AI configured)
auto_classify = true
auto_tag = true
auto_summarize = true  # Auto-summarize context when threshold hit

# ============================================================================
# Context Configuration
# ============================================================================

[context]
# Auto-summarize when current.md exceeds this size (bytes)
max_size = 50000  # ~50KB

# Number of sessions to keep in history
max_sessions = 10

# ============================================================================
# Tag Definitions (for classification)
# ============================================================================

[tags.rule]
description = "Project conventions, standards, and rules"
examples = [
  "Never use .ts extension for configuration files",
  "Always use async/await instead of promises",
  "Component names must be PascalCase"
]

[tags.authentication]
description = "Authentication and authorization mechanisms"
examples = [
  "JWT token validation in middleware",
  "OAuth2 flow implementation",
  "Session management with Redis"
]

[tags.database]
description = "Database operations, schema, and migrations"
examples = [
  "PostgreSQL migration scripts",
  "ORM model definitions",
  "Database connection pooling setup"
]

[tags.api]
description = "API endpoints, REST, GraphQL"
examples = [
  "REST endpoint implementation",
  "GraphQL resolver functions",
  "API rate limiting configuration"
]

# Add more tags as needed...

# ============================================================================
# Tool Paths
# ============================================================================

[tools]
nushell_path = "~/.quaid/tools/nu"
mq_path = "~/.quaid/tools/mq"
```

### Environment Variable Interpolation

**Syntax**: `#{VAR_NAME}`

**Example**:
```toml
[ai]
api_key = "#{OPENAI_API_KEY}"
api_url = "#{CUSTOM_API_ENDPOINT}"

[database]
connection_string = "postgresql://#{DB_USER}:#{DB_PASS}@localhost/mydb"
```

**NuShell implementation**:
```nu
# Load config
let raw_config = (open .quaid/config.toml)

# Load environment
let env_vars = (
    if ($raw_config.project.dot_env?) {
        # Load .env file
        let env_file = (open ($raw_config.project.dot_env) | lines)
        let parsed_env = ($env_file | each { |line|
            if ($line | str starts-with "#") or ($line | str trim | is-empty) {
                null
            } else {
                let parts = ($line | split row "=")
                {key: ($parts.0 | str trim), value: ($parts.1 | str trim)}
            }
        } | compact)
        
        # Merge with system env (system env takes precedence)
        $parsed_env | reduce -f $env { |item, acc|
            $acc | insert $item.key (
                if ($env | get -i $item.key) != null {
                    $env | get $item.key
                } else {
                    $item.value
                }
            )
        }
    } else {
        $env
    }
)

# Interpolate recursively
def interpolate [value: any, env_vars: record] {
    match ($value | describe) {
        "string" => {
            $value | str replace -a -r '#\{([A-Z_][A-Z0-9_]*)\}' { |match|
                let var_name = ($match.captures.0)
                $env_vars | get $var_name
            }
        }
        "record" => {
            $value | transpose k v | each { |pair|
                {($pair.k): (interpolate $pair.v $env_vars)}
            } | reduce -f {} { |it, acc| $acc | merge $it }
        }
        "list" => {
            $value | each { |item| interpolate $item $env_vars }
        }
        _ => $value
    }
}

let config = (interpolate $raw_config $env_vars)
```

---

## Directory Structure (Final)

```
project/
└── .quaid/
    ├── config.toml                    # Configuration with interpolation
    │
    ├── memory/
    │   ├── indexes/                   # JSONL indexes (lazy loading)
    │   │   ├── fragments.jsonl        # Fragment index
    │   │   └── tags.jsonl             # Tag index
    │   │
    │   └── fragments/                 # Markdown fragments
    │       ├── 2025-11-08-auth-001.md
    │       └── ...
    │
    ├── context/                       # Session context
    │   ├── current.md                 # Active session
    │   ├── 2025-11-08-session-1.md   # Archived summaries
    │   └── ...
    │
    └── scripts/                       # NuShell scripts
        ├── classify-and-tag.nu        # Single API call!
        ├── summarize.nu
        ├── rerank.nu
        └── lib/
            └── http-ai.nu

# AI tool integration (auto-generated by quaid init)
.cursor/
└── commands/
    ├── quaid-store.md
    ├── quaid-recall.md
    └── quaid-context.md

.cline/
└── commands/
    ├── quaid-store.md
    ├── quaid-recall.md
    └── quaid-context.md
```

---

## Lazy Loading with Polars

**Why indexes/ directory (plural)**:
- Large projects may have thousands of fragments
- Single JSONL file can grow to 100MB+
- Polars lazy dataframes only load what's needed

**Implementation**:
```nu
# Lazy loading - no memory overhead
let fragments = (polars open .quaid/memory/indexes/fragments.jsonl)

# Filter without loading entire file
let auth_fragments = (
    $fragments
    | polars filter (polars col tags | polars is-in ["authentication"])
    | polars select id path created
    | polars collect  # Only collect filtered results
)
```

**Performance**:
- Loading 100MB JSONL eagerly: ~5 seconds, ~200MB RAM
- Lazy loading + filter: ~50ms, ~5MB RAM

---

## Single API Call for Classification + Tags

**Old (wasteful)**:
```nu
# Two API calls!
let type = (echo $content | nu scripts/classify.nu)
let tags = (echo $content | nu scripts/extract-tags.nu)
```

**New (efficient)**:
```nu
# Single API call with structured output
let result = (nu scripts/classify-and-tag.nu $content)

# Returns: {type: "decision", tags: ["auth", "jwt"], summary: "..."}
```

**scripts/classify-and-tag.nu**:
```nu
use lib/http-ai.nu

def main [content: string] {
    let config = (open .quaid/config.toml | interpolate $env)
    
    if not $config.ai.classify.enabled {
        return null
    }
    
    # Load tag definitions for examples
    let tag_examples = (
        $config.tags 
        | transpose name data
        | each { |tag| 
            $"- ($tag.name): ($tag.data.description)\n  Examples: ($tag.data.examples | str join ', ')"
        }
        | str join "\n"
    )
    
    let prompt = $"Analyze this content and return ONLY valid JSON in this exact format:
{
  \"type\": \"rule|concept|implementation|decision|documentation|reference\",
  \"tags\": [\"tag1\", \"tag2\", \"tag3\"],
  \"summary\": \"one sentence summary\"
}

Use these predefined tags when possible:
($tag_examples)

Content to analyze:
($content)

JSON:"
    
    let response = (
        http post $config.ai.classify.api_url 
        --headers [
            "Authorization" $"Bearer ($config.ai.classify.api_key)"
            "Content-Type" "application/json"
        ]
        {
            model: $config.ai.classify.model
            messages: [{role: "user", content: $prompt}]
            response_format: {type: "json_object"}
            temperature: 0.3
        }
        | from json
    )
    
    let result = ($response.choices.0.message.content | from json)
    
    # Validate structure
    if ($result.type? == null) or ($result.tags? == null) {
        error make {msg: "AI returned invalid format"}
    }
    
    $result
}
```

---

## Slash Command Files (Auto-generated)

**Example: .cursor/commands/quaid-store.md**:
```markdown
---
name: /quaid-store
id: quaid-store
category: Quaid
description: Store knowledge fragment with auto-classification
---

**Purpose**: Store information as a queryable memory fragment.

**Usage**:
```
/quaid-store <type> <content>
```

**Types**: rule, concept, implementation, decision, documentation, reference

**Steps**:
1. Accept type and content from user
2. Run: `quaid store <type> "<content>"`
3. System automatically classifies and tags (if AI configured)
4. Confirm storage with fragment ID

**Examples**:
- `/quaid-store rule "Never use .ts extension for config files"`
- `/quaid-store decision "Use PostgreSQL for primary database"`
- `/quaid-store concept "MVC separates concerns into Model-View-Controller"`

**Reference**:
- Skip classification: `quaid store <type> --no-classify "<content>"`
- Add specific tags: `quaid store <type> --tags "tag1,tag2" "<content>"`
- Store from file: `quaid store <type> --file path/to/file.md`
```

**Example: .cursor/commands/quaid-recall.md**:
```markdown
---
name: /quaid-recall
id: quaid-recall
category: Quaid
description: Search knowledge base with AI reranking
---

**Purpose**: Retrieve relevant knowledge fragments.

**Usage**:
```
/quaid-recall <query>
```

**Steps**:
1. Accept search query from user
2. Run: `quaid recall "<query>"`
3. System searches content + metadata
4. Results auto-reranked by relevance (if AI configured)
5. Display top results

**Examples**:
- `/quaid-recall "how do we handle authentication"`
- `/quaid-recall "database connection pooling"`
- Filter by type: `quaid recall "api endpoints" --type implementation`
- Filter by tags: `quaid recall "security" --tags auth,encryption`

**Reference**:
- Skip AI reranking: `quaid recall --skip-rerank "<query>"`
```

---

## Summary of Key Decisions

| Question | Decision | Reason |
|----------|----------|--------|
| Index format? | **indexes/ directory with JSONL files** | Polars lazy loading for large datasets |
| Classify + tag? | **Single API call with structured output** | Reduce API calls, faster |
| AI reranking? | **Always if configured, --skip-rerank to disable** | Smart default |
| Context summarization? | **Auto-summarize when threshold hit (if AI configured)** | Automatic memory management |
| Tag definition? | **Pre-defined in config with examples** | Enable example-based classification (Cohere) |
| Config management? | **Edit only, no get/set commands** | Keep it simple |
| Env interpolation? | **#{VAR} syntax with .env support** | Flexible, secure |
| Slash commands? | **Auto-generated during init** | Matches CLI exactly |

---

**Previous**: [14-Core-Features-Refined.md](14-Core-Features-Refined.md)

**Version**: 1.0  
**Last Updated**: 2025-11-08
