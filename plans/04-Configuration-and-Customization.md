# 04 - Configuration and Customization

**Comprehensive guide to configuring and customizing Quaid for your specific needs**

---

## Overview

Quaid uses a hierarchical, TOML-based configuration system that allows fine-grained control over every aspect of the system. Configuration can be applied at global, user, and project levels, with sensible defaults that work out of the box.

This guide covers all configuration options, customization patterns, and advanced scenarios for tailoring Quaid to your specific workflow and requirements.

---

## Configuration Architecture

### Configuration Precedence

Quaid uses a layered configuration system where settings are merged in the following order (later settings override earlier ones):

1. **System Defaults** (`/etc/quaid/config.toml`)
2. **Global User Config** (`~/.quaid/config.toml`)
3. **Project Config** (`<project>/.quaid/config.toml`)
3. **Environment Variables**
5. **Command-Line Flags**

### Configuration Files

#### Project Configuration (`.quaid/config.toml`)

Project-specific settings that are committed to version control:

```toml
# .quaid/config.toml
# This file should be committed to your repository

[project]
name = "my-awesome-project"
description = "A web application with JWT authentication"
initialized = "2025-11-09T12:00:00Z"

# Optional .env file for environment variables
dot_env = ".env"
```

#### Global Configuration (`~/.quaid/config.toml`)

User-specific settings that apply across all projects:

```toml
# ~/.quaid/config.toml
# Personal preferences and tool configurations

[global]
default_editor = "vim"
default_shell = "bash"
auto_update = true
telemetry = false

[ai]
# Default AI provider for all projects
provider = "openai"
model = "gpt-4"
api_key = "#{OPENAI_API_KEY}"
```

#### Environment Variables

Environment variables take precedence over configuration files and support interpolation:

```bash
# Override configuration with environment variables
export QUAID_AI_PROVIDER="anthropic"
export QUAID_AI_MODEL="claude-3-opus-20240229"
export QUAID_SEARCH_DEFAULT_LIMIT="20"
export OPENAI_API_KEY="your-api-key-here"
```

---

## Core Configuration Sections

### Project Configuration

```toml
[project]
# Basic project information
name = "my-project"
description = "Project description"
version = "1.0.0"
initialized = "2025-11-09T12:00:00Z"

# Optional environment file (relative to project root)
dot_env = ".env"

# Project-specific metadata
maintainer = "your-name@example.com"
repository = "https://github.com/user/repo"
homepage = "https://project.example.com"
```

### Storage Configuration

```toml
[storage]
# Directory settings
fragment_dir = ".quaid/memory/fragments"
indexes_dir = ".quaid/memory/indexes"
context_dir = ".quaid/context"
cache_dir = ".quaid/cache"

# Index settings
index_format = "jsonl"  # jsonl, parquet
compression = true  # Compress index files

# Auto-behaviors
auto_index = true  # Automatically update indexes
auto_backup = false  # Auto-backup before major changes
backup_retention_days = 30

# Performance settings
memory_limit = "1GB"
max_fragment_size = "10MB"
```

### Search Configuration

```toml
[search]
# Search defaults
default_limit = 10
highlight_matches = true
show_scores = true
show_snippets = true

# Search scope
default_scope = "all"  # all, current_worktree, specific_worktree
include_archived = false

# Search behavior
fuzzy_search = true
case_sensitive = false
partial_match = true

# Performance
cache_results = true
cache_ttl = 3600  # seconds
max_cache_size = "100MB"

# Advanced search
enable_semantic_search = false
enable_cross_encoder = false
semantic_threshold = 0.7
```

### AI Configuration

```toml
[ai]
# Enable/disable AI features
enabled = true
mode = "hybrid"  # local, api, hybrid

# API configuration (when mode is "api" or "hybrid")
provider = "openai"  # openai, anthropic, cohere, azure
model = "gpt-4"
api_key = "#{OPENAI_API_KEY}"
api_url = "https://api.openai.com/v1/chat/completions"
temperature = 0.7
max_tokens = 4000

# Local AI configuration (when mode is "local" or "hybrid")
[ai.local]
classification_mode = "rule-based"  # rule-based, zero-shot, llm
llm_model = "phi-2"  # phi-2, tinyllama, stablelm-zephyr
enable_semantic_search = true
embedding_model = "all-MiniLM-L6-v2"
enable_cross_encoder = false
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Performance settings
[ai.local.performance]
max_workers = 4
use_gpu = false
low_memory_mode = false
batch_size = 32

# Classification settings
[ai.classification]
enabled = true
model = "gpt-4"
temperature = 0.3
confidence_threshold = 0.3
validate_user_choices = true

# Reranking settings
[ai.rerank]
enabled = true
model = "gpt-4"
top_k = 20
min_score = 0.5

# Context summarization
[ai.summarize]
enabled = true
model = "gpt-4"
max_words = 200
trigger_threshold = 15  # messages
```

### Worktree Configuration

```toml
[worktree]
# Worktree detection and management
auto_detect = true
default_scope = "current"  # current, all, specific
isolation = "partial"  # full, partial, none

# Integration settings
[worktree.integration]
auto_merge_types = ["concept", "decision", "documentation"]
review_required_types = ["implementation"]
conflict_resolution = "prompt"  # prompt, keep-both, keep-newer

# Worktree-specific memory
[worktree.memory]
share_by_default = false
inherit_global = true
sync_on_switch = true
```

### Tools and External Programs

```toml
[tools]
# External tool configurations
editor = "${EDITOR:-vim}"
pager = "less"
shell = "${SHELL:-bash}"
git = "git"

# Tool-specific settings
[tools.nushell]
enabled = false
path = "nu"
table_mode = "rounded"
use_ansi_coloring = true
edit_mode = "vi"
use_dataframes = true

[tools.mq]
enabled = true
path = "mq"
enable_modules = true
default_format = "markdown"
include_csv = true
include_json = true
include_yaml = true

[tools.aichat]
enabled = false
path = "aichat"
stream = true
save = true
highlight = true
```

### Logging Configuration

```toml
[logging]
# Logging levels: error, warn, info, debug, trace
level = "info"
format = "pretty"  # pretty, json, compact

# Output settings
console = true
file = true
file_path = ".quaid/logs/quaid.log"
max_file_size = "10MB"
max_files = 5

# Component-specific logging
[logging.components]
tantivy = "info"
polars = "warn"
spacy = "warn"
ai = "info"
```

---

## Fragment Type Configuration

### Custom Fragment Types

Define custom fragment types for your specific domain:

```toml
[fragment_types]
# Standard types (built-in)
concept = {description = "Conceptual explanations and theories"}
implementation = {description = "Code implementations and examples"}
decision = {description = "Architectural decisions and ADRs"}
reference = {description = "External references and links"}
pattern = {description = "Reusable patterns and templates"}
troubleshooting = {description = "Problem-solving guides"}
api_doc = {description = "API documentation"}

# Custom types
[fragment_types.design_doc]
description = "Design documents and specifications"
icon = "📋"
color = "blue"
importance_default = "high"

[fragment_types.meeting_notes]
description = "Meeting notes and outcomes"
icon = "📝"
color = "green"
importance_default = "medium"

[fragment_types.research]
description = "Research findings and analysis"
icon = "🔬"
color = "purple"
importance_default = "high"

[fragment_types.bug_report]
description = "Bug reports and fixes"
icon = "🐛"
color = "red"
importance_default = "high"
```

### Tag Configuration

Define your project's tag taxonomy:

```toml
# Tag definitions with examples for classification
[tags.authentication]
description = "Authentication and authorization mechanisms"
examples = [
    "JWT token validation in middleware",
    "OAuth2 flow implementation",
    "Session management with Redis",
    "Multi-factor authentication setup"
]
color = "red"

[tags.database]
description = "Database operations, schema, and migrations"
examples = [
    "PostgreSQL migration scripts",
    "ORM model definitions",
    "Database connection pooling",
    "Query optimization techniques"
]
color = "blue"

[tags.api]
description = "API endpoints, REST, GraphQL"
examples = [
    "REST endpoint implementation",
    "GraphQL resolver functions",
    "API rate limiting configuration",
    "OpenAPI specification"
]
color = "green"

[tags.security]
description = "Security practices and vulnerabilities"
examples = [
    "XSS prevention strategies",
    "CSRF token implementation",
    "Input validation and sanitization",
    "Security audit findings"
]
color = "orange"

[tags.performance]
description = "Performance optimization and monitoring"
examples = [
    "Database query optimization",
    "Caching strategies",
    "Load balancing configuration",
    "Performance profiling results"
]
color = "purple"

[tags.testing]
description = "Testing strategies and test implementations"
examples = [
    "Unit test examples",
    "Integration test setup",
    "Test data management",
    "Testing best practices"
]
color = "yellow"

# Custom project-specific tags
[tags.payment]
description = "Payment processing and financial operations"
examples = [
    "Stripe webhook handling",
    "Payment gateway integration",
    "Refund processing logic",
    "Financial audit trails"
]
color = "green"

[tags.mobile]
description = "Mobile application development"
examples = [
    "React Native components",
    "iOS/Android specific implementations",
    "Mobile API integration",
    "Push notification handling"
]
color = "indigo"
```

---

## Advanced Configuration

### Custom Classifiers

Configure custom classification rules and models:

```toml
[classification]
# Classification strategy
strategy = "hybrid"  # hybrid, ai-only, rule-based-only

# Type classification
[classification.type]
mode = "hybrid"  # user, auto, hybrid
validate = true
validation_threshold = 0.3
conflict_resolution = "prompt"  # prompt, ai, user

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
boost_from_user_flags = true

# Completeness assessment
[classification.completeness]
enabled = true
rules = "default"  # default, custom
thresholds.complete = 7
thresholds.partial = 4
```

### Custom Rules

Define custom classification rules:

```toml
[custom_rules]
# Content-based rules
[custom_rules.type_detection]
code_heavy = {threshold = 3, type = "implementation", field = "code_block_count"}
decision_markers = {patterns = ["> **Decision**", "**Rationale**"], type = "decision"}
reference_heavy = {threshold = 5, type = "reference", field = "link_count"}
tutorial_like = {patterns = ["## Step", "### Example", "## How to"], type = "implementation"}

# Importance rules
[custom_rules.importance_boosting]
high_importance_words = ["critical", "urgent", "security", "production"]
high_importance_admonitions = ["!!! warning", "!!! danger", "!!! important"]
high_importance_tags = ["security", "production", "deployment"]

# Tag extraction rules
[custom_rules.tag_extraction]
file_extensions = {pattern = r"\.(py|js|ts|jsx|tsx)$", tags = ["code"]}
framework_names = {patterns = ["React", "Vue", "Angular", "Django", "Flask"], tags = ["framework"]}
cloud_providers = {patterns = ["AWS", "Azure", "GCP", "Google Cloud"], tags = ["cloud"]}
```

### Custom Templates

Configure custom templates for different fragment types:

```toml
[templates]
# Template directory
directory = ".quaid/templates"
default_type = "concept"

# Type-specific templates
[templates.concept]
file = "concept.md"
required_sections = ["Overview", "Examples", "Related Concepts"]
optional_sections = ["Implementation", "References"]

[templates.implementation]
file = "implementation.md"
required_sections = ["Overview", "Implementation", "Examples"]
optional_sections = ["Testing", "Troubleshooting", "References"]

[templates.decision]
file = "decision.md"
required_sections = ["Decision", "Rationale", "Alternatives", "Consequences"]
optional_sections = ["Implementation", "References"]
```

### Custom Prompts

Configure custom AI prompts for different operations:

```toml
[prompts]
directory = ".quaid/prompts"
default_language = "en"

# Classification prompts
[prompts.classify.type]
template = "classify_type.prompt.md"
temperature = 0.3
max_tokens = 100

[prompts.classify.tags]
template = "classify_tags.prompt.md"
temperature = 0.5
max_tokens = 200

[prompts.summarize]
template = "summarize.prompt.md"
temperature = 0.7
max_tokens = 300

# Custom prompts
[prompts.custom.code_review]
template = "code_review.prompt.md"
enabled = true
auto_trigger = false
```

---

## Environment-Specific Configuration

### Development Environment

```toml
# config/development.toml
[environment]
name = "development"
debug = true
logging.level = "debug"

[ai]
enabled = true
mode = "local"
temperature = 0.8

[search]
default_limit = 20
show_scores = true

[storage]
auto_backup = true
backup_retention_days = 7
```

### Production Environment

```toml
# config/production.toml
[environment]
name = "production"
debug = false
logging.level = "warn"

[ai]
enabled = false  # Disable AI in production for privacy
mode = "rule-based"

[search]
default_limit = 10
cache_ttl = 7200

[storage]
compression = true
auto_backup = true
backup_retention_days = 90
```

### Testing Environment

```toml
# config/testing.toml
[environment]
name = "testing"
debug = true

[storage]
fragment_dir = ".quaid/test/fragments"
indexes_dir = ".quaid/test/indexes"

[search]
cache_results = false

[ai]
enabled = false
```

### Loading Environment-Specific Config

```bash
# Set environment variable
export QUAID_ENV="development"

# Or use command line flag
quaid --config config/development.toml doctor
```

---

## Configuration Management Commands

### Basic Configuration Commands

```bash
# View current configuration
quaid config show
quaid config show --all  # Include defaults
quaid config show --project  # Project config only
quaid config show --global   # Global config only

# Edit configuration
quaid config edit  # Opens appropriate config in default editor
quaid config edit --project
quaid config edit --global

# Set configuration values
quaid config set search.default_limit 20
quaid config set ai.enabled true
quaid config set tags.database.color blue

# Get configuration values
quaid config get search.default_limit
quaid config get ai.provider

# List all configuration keys
quaid config list
quaid config list --section ai
quaid config list --section search
```

### Advanced Configuration Management

```bash
# Validate configuration
quaid config validate
quaid config validate --strict

# Reset configuration
quaid config reset  # Reset to defaults
quaid config reset --section ai
quaid config reset search.default_limit

# Import/Export configuration
quaid config export > backup.toml
quaid config import backup.toml
quaid config import --merge additional.toml

# Compare configurations
quaid config diff
quaid config diff --global
quaid config diff --file other-config.toml

# Template generation
quaid config init --template minimal
quaid config init --template full
quaid config init --template ai-heavy
```

### Configuration Templates

```bash
# Initialize with template
quaid config init --template basic
quaid config init --template advanced
quaid config init --template local-ai
quaid config init --template cloud-ai

# Create custom template
quaid config template create my-template
quaid config template edit my-template
quaid config template use my-template
```

---

## Environment Variables

### Supported Environment Variables

```bash
# Core configuration
QUAID_CONFIG_FILE="/path/to/config.toml"
QUAID_ENV="development"
QUAID_DEBUG="true"
QUAID_LOG_LEVEL="debug"

# AI configuration
QUAID_AI_PROVIDER="anthropic"
QUAID_AI_MODEL="claude-3-opus-20240229"
QUAID_AI_API_KEY="your-api-key"
QUAID_AI_API_URL="https://api.anthropic.com"

# Search configuration
QUAID_SEARCH_DEFAULT_LIMIT="20"
QUAID_SEARCH_CACHE_SIZE="200MB"

# Storage configuration
QUAID_STORAGE_FRAGMENT_DIR="/custom/path/fragments"
QUAID_STORAGE_INDEXES_DIR="/custom/path/indexes"

# External tools
QUAID_EDITOR="code"
QUAID_PAGER="less"
QUAID_SHELL="zsh"

# Paths
QUAID_HOME="/custom/quaid/home"
QUAID_CACHE_DIR="/custom/cache/dir"
```

### Environment Variable Interpolation

Configuration files support environment variable interpolation:

```toml
[ai]
api_key = "#{OPENAI_API_KEY}"
api_url = "#{CUSTOM_API_ENDPOINT}/v1/chat"

[storage]
cache_dir = "#{HOME}/.quaid/cache"

[database]
connection_string = "postgresql://#{DB_USER}:#{DB_PASS}@localhost/mydb"
```

### .env File Support

Quaid automatically loads `.env` files from:

1. Project root (if `dot_env` is configured)
2. `.quaid/` directory
3. User home directory

```bash
# .env file example
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
QUAID_AI_MODEL=claude-3-opus-20240229
QUAID_DEBUG=true
```

---

## Customization Examples

### Example 1: Security-Focused Configuration

```toml
# For security audits and compliance
[project]
name = "security-audit-tool"
description = "Security auditing and compliance tracking"

[tags]
[tags.security]
description = "Security vulnerabilities and fixes"
examples = ["XSS vulnerabilities", "SQL injection prevention"]
color = "red"
importance_default = "high"

[tags.compliance]
description = "Compliance requirements and checks"
examples = ["GDPR compliance", "SOC2 controls"]
color = "orange"
importance_default = "high"

[ai]
enabled = true
mode = "local"  # Keep security data local
classification_mode = "rule-based"

[search]
default_scope = "all"
include_archived = true  # Keep historical security data

[storage]
auto_backup = true
backup_retention_days = 2555  # 7 years for compliance
compression = true
```

### Example 2: Machine Learning Project Configuration

```toml
# For ML/AI development projects
[project]
name = "ml-model-development"
description = "Machine learning model development pipeline"

[tags]
[tags.model]
description = "ML model implementations"
examples = ["Neural network architecture", "Model training code"]
color = "blue"

[tags.data]
description = "Data processing and pipelines"
examples = ["Data preprocessing", "Feature engineering"]
color = "green"

[tags.experiment]
description = "ML experiments and results"
examples = ["Hyperparameter tuning", "Model comparison"]
color = "purple"

[fragment_types]
[fragment_types.experiment]
description = "ML experiment documentation"
icon = "🧪"
color = "purple"
importance_default = "medium"

[fragment_types.dataset]
description = "Dataset documentation and processing"
icon = "📊"
color = "green"
importance_default = "high"

[ai]
enabled = true
mode = "api"  # Use powerful models for ML insights
provider = "openai"
model = "gpt-4"
temperature = 0.3

[search]
enable_semantic_search = true
semantic_threshold = 0.8
```

### Example 3: Open Source Project Configuration

```toml
# For open source projects with many contributors
[project]
name = "awesome-library"
description = "A popular open source library"
maintainer = "maintainer@example.com"
repository = "https://github.com/user/awesome-library"

[tags]
[tags.contributor-guide]
description = "Contributor guidelines and documentation"
examples = ["Pull request guidelines", "Code review process"]
color = "blue"

[tags.release]
description = "Release notes and version information"
examples = ["Version 2.0 release notes", "Breaking changes"]
color = "green"

[tags.community]
description = "Community discussions and decisions"
examples = ["RFC discussions", "Community feedback"]
color = "orange"

[worktree]
auto_detect = true
default_scope = "all"
isolation = "partial"

[worktree.integration]
auto_merge_types = ["concept", "decision", "documentation", "pattern"]
review_required_types = ["implementation"]
conflict_resolution = "prompt"

[ai]
enabled = false  # Keep community project accessible
mode = "rule-based"

[search]
default_limit = 15
include_archived = false
```

---

## Configuration Best Practices

### 1. Version Control Strategy

```bash
# Commit project configuration (safe for team sharing)
git add .quaid/config.toml
git commit -m "Add Quaid configuration"

# Ignore global configuration (personal preferences)
echo ".quaid/config.toml" >> .gitignore
echo ".quaid/cache/" >> .gitignore
echo ".quaid/logs/" >> .gitignore
```

### 2. Environment Separation

```bash
# Use different configs for different environments
quaid config edit --environment development
quaid config edit --environment production
quaid config edit --environment testing

# Load with environment variable
export QUAID_ENV="production"
quaid recall "production settings"
```

### 3. Security Considerations

```bash
# Use environment variables for sensitive data
quaid config set ai.api_key "#{OPENAI_API_KEY}"
export OPENAI_API_KEY="your-secure-key"

# Never commit API keys to version control
echo ".env" >> .gitignore
echo "**.key" >> .gitignore
```

### 4. Performance Optimization

```toml
# For large projects
[storage]
memory_limit = "2GB"
max_fragment_size = "50MB"
compression = true

[search]
cache_results = true
cache_ttl = 3600
max_cache_size = "500MB"

[logging]
level = "warn"  # Reduce log overhead
```

### 5. Team Collaboration

```toml
# Define team standards in project config
[project]
name = "team-project"
team_standards = true

[tags]
# Define required tags for consistency
[tags.required]
description = "Required tags for all fragments"
tags = ["component", "status"]

[fragment_types]
# Standardize fragment types
custom_types = false  # Use only built-in types
```

---

## Troubleshooting Configuration

### Common Issues

#### 1. Configuration Not Loading

```bash
# Check configuration file locations
quaid config show --paths

# Validate configuration syntax
quaid config validate --verbose

# Check for interpolation errors
quaid config show --debug
```

#### 2. Environment Variables Not Working

```bash
# Check environment variable expansion
quaid config show --interpolated

# Verify variable names
env | grep QUAID_
```

#### 3. AI Configuration Issues

```bash
# Test AI configuration
quaid ai test --provider openai
quaid ai test --local

# Check model availability
quaid models list
quaid models status
```

### Debug Mode

```bash
# Enable debug logging
export QUAID_DEBUG=1
quaid config show

# Run with verbose output
quaid --verbose config validate

# Check configuration loading
quaid config show --trace
```

---

## Next Steps

After configuring Quaid for your needs:

1. **Test Your Configuration**: [05-Core-Features.md](05-Core-Features.md)
2. **Explore Search Capabilities**: [06-Search-and-Intelligence.md](06-Search-and-Intelligence.md)
3. **Learn CLI Commands**: [07-CLI-and-API-Reference.md](07-CLI-and-API-Reference.md)
4. **Discover Advanced Features**: [08-Advanced-Features.md](08-Advanced-Features.md)

---

**Previous**: [03-Installation-and-Setup.md](03-Installation-and-Setup.md) | **Next**: [05-Core-Features.md](05-Core-Features.md)