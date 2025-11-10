# 07 - MCP Server and Tool Reference

**Complete reference guide to Quaid's MCP server, tools, and agent integration capabilities**

---

## Overview

Quaid implements a Model Context Protocol (MCP) server that provides AI agents with efficient access to memory management capabilities through code-based tool composition. This approach enables agents to discover tools progressively, process data context-efficiently, and build reusable skills while maintaining privacy and state persistence.

### Tool Categories

1. **Core Memory Tools**: Fragment creation, retrieval, and management
2. **Search Tools**: Full-text, structural, and semantic search capabilities
3. **Context Tools**: Session management and working memory operations
4. **Agent Tools**: Multi-agent coordination and performance tracking
5. **Analysis Tools**: ML-powered insights and pattern recognition
6. **Integration Tools**: External system connections and automation

### MCP Server Architecture

```
servers/
├── quaid/
│   ├── fragments/
│   │   ├── create.py
│   │   ├── search.py
│   │   ├── update.py
│   │   └── delete.py
│   ├── search/
│   │   ├── fulltext.py
│   │   ├── semantic.py
│   │   └── temporal.py
│   ├── context/
│   │   ├── session.py
│   │   ├── workspace.py
│   │   └── hot_context.py
│   ├── agents/
│   │   ├── coordination.py
│   │   ├── performance.py
│   │   └── skills.py
│   └── analysis/
│       ├── importance.py
│       ├── patterns.py
│       └── relationships.py
└── __init__.py
```

### Code Execution Benefits

The MCP server approach provides several key advantages over traditional CLI/slash commands:

1. **Progressive Tool Discovery**: Agents load only the tools they need
2. **Context-Efficient Processing**: Filter and transform data in code before returning results
3. **Privacy-Preserving Operations**: Sensitive data can flow through the execution environment without entering model context
4. **State Persistence**: Maintain state across operations using the filesystem
5. **Skill Development**: Agents can create and reuse higher-level capabilities

---

## Core Commands

### `quaid init`

Initialize a new Quaid project or reconfigure an existing one.

#### Syntax
```bash
quaid init [project-name] [options]
```

#### Options
- `--ai-enabled`: Enable AI features during initialization
- `--local-models`: Download local AI models
- `--worktree-support`: Enable multi-worktree support
- `--template <template>`: Use specific initialization template
- `--config <file>`: Use custom configuration file
- `--force`: Reinitialize existing project

#### Examples
```bash
# Initialize basic project
quaid init my-project

# Initialize with AI features
quaid init my-project --ai-enabled --local-models

# Initialize with specific template
quaid init my-project --template security-focused

# Reinitialize existing project
quaid init --force
```

### `quaid store`

Store knowledge fragments with automatic classification.

#### Syntax
```bash
quaid store [type] [content] [options]
quaid store --file <path> [options]
quaid store --clipboard [options]
```

#### Arguments
- `type` (optional): Fragment type (concept, implementation, decision, reference, pattern, troubleshooting, api-doc)
- `content`: Fragment content (when not using --file or --clipboard)

#### Options
- `--file <path>`: Store content from file
- `--clipboard`: Store content from clipboard
- `--tags <tags>`: Specify tags (comma-separated)
- `--importance <level>`: Set importance (high, medium, low)
- `--type <type>`: Force specific fragment type
- `--no-classify`: Skip AI classification
- `--title <title>`: Set fragment title
- `--author <author>`: Set fragment author
- `--interactive`: Interactive fragment creation

#### Examples
```bash
# Store simple concept
quaid store concept "JWT provides stateless authentication"

# Store implementation with tags
quaid store implementation "JWT validation code" --tags python,jwt,security

# Store from file
quaid store --file ./docs/api-design.md --tags api,documentation

# Store with specific type and importance
quaid store decision --importance high "Use JWT over sessions"

# Interactive creation
quaid store --interactive

# Store without classification
quaid store note "Quick reminder" --no-classify
```

### `quaid get`

Retrieve and display specific fragments.

#### Syntax
```bash
quaid get <fragment-id> [options]
```

#### Arguments
- `fragment-id`: ID of fragment to retrieve

#### Options
- `--format <format>`: Output format (pretty, json, markdown, raw)
- `--content-only`: Show only content, no metadata
- `--sections <sections>`: Show specific sections (comma-separated)
- `--output <file>`: Save to file instead of displaying

#### Examples
```bash
# Get fragment with pretty formatting
quaid get jwt-001

# Get as JSON
quaid get jwt-001 --format json

# Get only content
quaid get jwt-001 --content-only

# Get specific sections
quaid get jwt-001 --sections overview,implementation

# Save to file
quaid get jwt-001 --output jwt-doc.md
```

### `quaid list`

List fragments with filtering and sorting options.

#### Syntax
```bash
quaid list [options]
```

#### Options
- `--type <type>`: Filter by fragment type
- `--tags <tags>`: Filter by tags (comma-separated)
- `--importance <level>`: Filter by importance
- `--author <author>`: Filter by author
- `--created-since <date>`: Filter by creation date
- `--updated-within <days>`: Filter by update recency
- `--worktree <worktree>`: Filter by worktree
- `--limit <number>`: Limit number of results
- `--sort <field>`: Sort by field (created, updated, title, importance)
- `--order <order>`: Sort order (asc, desc)
- `--format <format>`: Output format (table, json, list)
- `--detailed`: Show detailed information
- `--group-by <field>`: Group results by field

#### Examples
```bash
# List all fragments
quaid list

# List by type
quaid list --type implementation

# List by tags
quaid list --tags authentication,jwt

# List recent high-importance items
quaid list --importance high --updated-within 7

# Detailed list with limit
quaid list --detailed --limit 20

# Group by type
quaid list --group-by type

# JSON output
quaid list --format json
```

### `quaid update`

Update existing fragments.

#### Syntax
```bash
quaid update <fragment-id> [options]
```

#### Arguments
- `fragment-id`: ID of fragment to update

#### Options
- `--content <content>`: Update fragment content
- `--title <title>`: Update fragment title
- `--tags <tags>`: Update fragment tags
- `--importance <level>`: Update importance
- `--type <type>`: Update fragment type
- `--append`: Append content instead of replacing
- `--section <section>`: Update specific section
- `--metadata <key=value>`: Update metadata key-value pair

#### Examples
```bash
# Update content
quaid update jwt-001 --content "Updated JWT documentation"

# Update metadata
quaid update jwt-001 --title "JWT Authentication Guide" --importance high

# Append content
quaid update jwt-001 --append "Additional implementation notes"

# Update specific section
quaid update jwt-001 --section implementation --content "New implementation details"

# Update custom metadata
quaid update jwt-001 --metadata status=reviewed
```

### `quaid delete`

Delete fragments with confirmation and archive options.

#### Syntax
```bash
quaid delete <fragment-id> [options]
```

#### Arguments
- `fragment-id`: ID of fragment to delete

#### Options
- `--force`: Delete without confirmation
- `--archive`: Move to archive instead of deleting
- `--reason <reason>`: Reason for deletion (logged)

#### Examples
```bash
# Delete with confirmation
quaid delete jwt-001

# Force delete
quaid delete jwt-001 --force

# Archive instead of delete
quaid delete jwt-001 --archive

# Delete with reason
quaid delete jwt-001 --reason "Outdated information"
```

---

## Search Commands

### `quaid recall` / `quaid search`

Search for fragments using multiple search strategies.

#### Syntax
```bash
quaid recall <query> [options]
quaid search <query> [options]
```

#### Arguments
- `query`: Search query or natural language question

#### Options
- `--type <type>`: Filter by fragment type
- `--tags <tags>`: Filter by tags (comma-separated)
- `--importance <level>`: Filter by importance
- `--worktree <worktree>`: Search specific worktree (current, all, or name)
- `--limit <number>`: Limit number of results
- `--semantic`: Enable semantic search
- `--rerank`: Enable result reranking
- `--no-rerank`: Disable reranking
- `--field <field>`: Search specific field (title, content, code, decisions)
- `--format <format>`: Output format (table, json, list)
- `--highlight`: Highlight search matches
- `--show-scores`: Show relevance scores
- `--explain`: Show search explanation

#### Examples
```bash
# Basic search
quaid recall "JWT authentication"

# Filtered search
quaid recall "authentication" --type implementation --tags python

# Semantic search
quaid recall "how to implement user authentication" --semantic

# Search specific worktree
quaid recall "database" --worktree feature/api

# Limited results with scores
quaid recall "security" --limit 5 --show-scores

# Search with explanation
quaid recall "token validation" --explain
```

### `quaid query`

Direct markdown-query operations on fragments.

#### Syntax
```bash
quaid query <selector> [options]
```

#### Arguments
- `selector`: Markdown-query selector expression

#### Options
- `--fragments <ids>`: Query specific fragments (comma-separated)
- `--output <format>`: Output format (text, json, count)
- `--limit <number>`: Limit results

#### Examples
```bash
# Extract all code blocks
quaid query ".code | to_text()"

# Get headings
quaid query "select(.h1, .h2) | to_text()"

# Find decision blocks
quaid query "select(.blockquote) | to_text()"

# Count code blocks
quaid query ".code | count"

# Query specific fragments
quaid query ".h1 | to_text()" --fragments jwt-001,jwt-002
```

### `quaid related`

Find fragments related to a specific fragment.

#### Syntax
```bash
quaid related <fragment-id> [options]
```

#### Arguments
- `fragment-id`: ID of fragment to find relations for

#### Options
- `--depth <number>`: Relationship depth (default: 2)
- `--type <type>`: Filter by relationship type (implements, references, related-to, depends-on)
- `--limit <number>`: Limit number of results
- `--include-self`: Include the original fragment in results

#### Examples
```bash
# Find related fragments
quaid related jwt-001

# Find implementations of a concept
quaid related jwt-001 --type implements --depth 3

# Find references
quaid related jwt-001 --type references
```

---

## Context Commands

### `quaid context`

Manage session context and working memory.

#### Syntax
```bash
quaid context <subcommand> [options]
```

#### Subcommands

##### `context status`
Show current session context.

```bash
quaid context status [--format <format>]
```

##### `context add-goal`
Add a new goal to current context.

```bash
quaid context add-goal <goal> [--priority <priority>] [--parent <parent-id>]
```

##### `context complete-goal`
Mark a goal as complete.

```bash
quaid context complete-goal <goal-id-or-description>
```

##### `context list-goals`
List all active goals.

```bash
quaid context list-goals [--status <status>]
```

##### `context log-decision`
Log a decision to context.

```bash
quaid context log-decision <decision> [rationale]
```

##### `context add-constraint`
Add a constraint to current context.

```bash
quaid context add-constraint <constraint>
```

##### `context summarize`
Generate session summary.

```bash
quaid context summarize [--max-words <number>]
```

##### `context save`
Save current context state.

```bash
quaid context save [--label <label>]
```

##### `context restore`
Restore previous context.

```bash
quaid context restore [--session-id <id>] [--label <label>]
```

##### `context clear`
Clear all context.

```bash
quaid context clear [--keep-goals]
```

#### Examples
```bash
# Show current context
quaid context status

# Add goals
quaid context add-goal "Implement JWT authentication"
quaid context add-goal "Add error handling" --priority high

# Complete goals
quaid context complete-goal "Implement JWT authentication"

# Log decisions
quaid context log-decision "Use JWT for authentication" "Stateless scaling"

# Add constraints
quaid context add-constraint "All API endpoints must be authenticated"

# Save and restore
quaid context save --label "jwt-work"
quaid context restore --label "jwt-work"

# Generate summary
quaid context summarize --max-words 200
```

### `quaid remember`

Quick way to add important information to context.

#### Syntax
```bash
quaid remember <information>
```

#### Examples
```bash
# Remember important detail
quaid remember "JWT tokens expire after 15 minutes"

# Remember file location
quaid remember "Auth middleware is in src/middleware/auth.py"

# Remember command
quaid remember "Run tests with: pytest tests/test_auth.py"
```

---

## Configuration Commands

### `quaid config`

Manage configuration settings.

#### Syntax
```bash
quaid config <subcommand> [options]
```

#### Subcommands

##### `config show`
Display current configuration.

```bash
quaid config show [--all] [--section <section>] [--format <format>]
```

##### `config get`
Get specific configuration value.

```bash
quaid config get <key>
```

##### `config set`
Set configuration value.

```bash
quaid config set <key> <value>
```

##### `config edit`
Edit configuration in editor.

```bash
quaid config edit [--project] [--global]
```

##### `config validate`
Validate configuration.

```bash
quaid config validate [--strict]
```

##### `config reset`
Reset configuration to defaults.

```bash
quaid config reset [--section <section>] [--key <key>]
```

##### `config import`
Import configuration from file.

```bash
quaid config import <file> [--merge]
```

##### `config export`
Export configuration to file.

```bash
quaid config export [--output <file>]
```

#### Examples
```bash
# Show all configuration
quaid config show --all

# Get specific value
quaid config get search.default_limit

# Set configuration
quaid config set ai.enabled true
quaid config set search.enable_semantic_search true

# Edit configuration
quaid config edit --project

# Validate configuration
quaid config validate --strict

# Reset section
quaid config reset --section ai

# Export configuration
quaid config export --output backup.toml
```

---

## Integration Commands

### `quaid integrate`

Integrate with external tools and services.

#### Syntax
```bash
quaid integrate <tool> [options]
```

#### Supported Tools

##### Cursor
Generate Cursor slash commands.

```bash
quaid integrate cursor [--force]
```

##### Claude
Generate Claude slash commands.

```bash
quaid integrate claude [--force]
```

##### GitHub Copilot
Generate GitHub Copilot integration.

```bash
quaid integrate github-copilot
```

##### VS Code
Generate VS Code extension configuration.

```bash
quaid integrate vscode
```

##### Neovim
Generate Neovim plugin configuration.

```bash
quaid integrate neovim
```

#### Options
- `--force`: Overwrite existing integration files
- `--path <path>`: Custom installation path
- `--template <template>`: Use custom template

#### Examples
```bash
# Generate all integrations
quaid integrate all

# Generate specific tool integration
quaid integrate cursor --force

# Generate with custom path
quaid integrate claude --path ~/.claude-custom
```

### `quaid slash`

Generate slash commands for AI assistants.

#### Syntax
```bash
quaid slash generate [options]
quaid slash validate [options]
```

#### Options
- `--tools <tools>`: Specific tools to generate for
- `--template <template>`: Use custom template
- `--validate-only`: Only validate existing commands

#### Examples
```bash
# Generate all slash commands
quaid slash generate

# Generate for specific tools
quaid slash generate --tools cursor,claude

# Validate existing commands
quaid slash validate
```

---

## Advanced Commands

### `quaid index`

Manage search indexes.

#### Syntax
```bash
quaid index <subcommand> [options]
```

#### Subcommands

##### `index build`
Build or rebuild search indexes.

```bash
quaid index build [--force] [--optimize]
```

##### `index optimize`
Optimize existing indexes.

```bash
quaid index optimize
```

##### `index stats`
Show index statistics.

```bash
quaid index stats [--detailed]
```

##### `index validate`
Validate index integrity.

```bash
quaid index validate [--fix]
```

#### Examples
```bash
# Rebuild indexes
quaid index build --force

# Optimize indexes
quaid index optimize

# Show detailed statistics
quaid index stats --detailed
```

### `quaid graph`

Manage knowledge graph relationships.

#### Syntax
```bash
quaid graph <subcommand> [options]
```

#### Subcommands

##### `graph show`
Show relationships for a fragment.

```bash
quaid graph show <fragment-id> [--depth <depth>]
```

##### `graph related`
Find related fragments.

```bash
quaid graph related <fragment-id> [--limit <number>]
```

##### `graph visualize`
Generate graph visualization.

```bash
quaid graph visualize [--output <file>] [--format <format>]
```

##### `graph stats`
Show graph statistics.

```bash
quaid graph stats [--detailed]
```

#### Examples
```bash
# Show fragment relationships
quaid graph show jwt-001 --depth 2

# Find related fragments
quaid graph related jwt-001 --limit 10

# Generate visualization
quaid graph visualize --output graph.png

# Show detailed statistics
quaid graph stats --detailed
```

### `quaid backup`

Backup and restore Quaid data.

#### Syntax
```bash
quaid backup <subcommand> [options]
```

#### Subcommands

##### `backup create`
Create backup.

```bash
quaid backup create [--description <text>]
```

##### `backup restore`
Restore from backup.

```bash
quaid backup restore <backup-id> [--config-only] [--fragments <ids>]
```

##### `backup list`
List available backups.

```bash
quaid backup list [--detailed]
```

##### `backup delete`
Delete backup.

```bash
quaid backup delete <backup-id>
```

#### Examples
```bash
# Create backup with description
quaid backup create --description "Before major refactoring"

# List backups
quaid backup list --detailed

# Restore from backup
quaid backup restore backup-2025-11-09-001

# Restore specific fragments
quaid backup restore backup-2025-11-09-001 --fragments jwt-001,jwt-002
```

### `quaid worktree`

Manage multi-worktree operations.

#### Syntax
```bash
quaid worktree <subcommand> [options]
```

#### Subcommands

##### `worktree status`
Show worktree information.

```bash
quaid worktree status
```

##### `worktree switch`
Switch to different worktree context.

```bash
quaid worktree switch <worktree-name>
```

##### `worktree integrate`
Integrate memories between worktrees.

```bash
quaid worktree integrate --from <source> --to <target> [--dry-run]
```

#### Examples
```bash
# Show worktree status
quaid worktree status

# Switch worktree
quaid worktree switch feature/authentication

# Integrate worktrees
quaid worktree integrate --from feature/auth --to main
```

---

## Slash Commands

Quaid automatically generates slash commands for AI coding assistants. These mirror the CLI functionality but are optimized for conversational AI interaction.

### `/quaid-store`

Store knowledge fragments with AI assistance.

#### Usage
```
/quaid-store <type> <content> [options]
```

#### Types
- `concept`, `implementation`, `decision`, `reference`, `pattern`, `troubleshooting`, `api-doc`

#### Options
- `--tags <tags>`: Specify tags
- `--importance <level>`: Set importance
- `--no-classify`: Skip AI classification

#### Examples
```
/quaid-store concept JWT provides stateless authentication
/quaid-store implementation Here's how to validate JWT tokens in Python
/quaid-store decision --importance high We chose JWT over sessions
/quaid-store pattern Always validate JWT signature before processing
```

### `/quaid-recall`

Search and retrieve knowledge fragments.

#### Usage
```
/quaid-recall <query> [options]
```

#### Options
- `--type <type>`: Filter by fragment type
- `--tags <tags>`: Filter by tags
- `--limit <number>`: Limit results

#### Examples
```
/quaid-recall JWT authentication patterns
/quaid-recall how to implement token validation
/quaid-recall database connection pooling --type implementation
/quaid-recall security best practices --limit 5
```

### `/quaid-context`

Manage session context and working memory.

#### Subcommands
- `/quaid-context status`: Show current context
- `/quaid-context add-goal <goal>`: Add a goal
- `/quaid-context complete-goal <goal>`: Complete a goal
- `/quaid-context remember <information>`: Remember important information
- `/quaid-context summarize`: Generate session summary

#### Examples
```
/quaid-context status
/quaid-context add-goal Implement JWT authentication
/quaid-context remember JWT tokens expire after 15 minutes
/quaid-context complete-goal Add error handling
/quaid-context summarize
```

### `/quaid-related`

Find fragments related to current conversation.

#### Usage
```
/quaid-related [options]
```

#### Options
- `--depth <number>`: Relationship depth
- `--limit <number>`: Limit results

#### Examples
```
/quaid-related
/quaid-related --depth 3 --limit 10
```

---

## Command Reference Summary

### Quick Reference Card

```bash
# Core Operations
quaid init [project]              # Initialize project
quaid store [type] [content]       # Store fragment
quaid get <id>                    # Get fragment
quaid list [options]              # List fragments
quaid recall <query>              # Search fragments
quaid update <id> [options]       # Update fragment
quaid delete <id>                 # Delete fragment

# Context Management
quaid context status              # Show context
quaid context add-goal <goal>     # Add goal
quaid remember <info>             # Remember info
quaid context summarize           # Generate summary

# Configuration
quaid config show                 # Show config
quaid config set <key> <value>    # Set config
quaid config edit                 # Edit config

# Advanced
quaid index build                 # Rebuild indexes
quaid graph show <id>             # Show relationships
quaid backup create               # Create backup
quaid integrate <tool>            # Generate integrations
```

### Common Workflows

#### Daily Knowledge Management
```bash
# Start work
quaid context restore

# Search for relevant info
quaid recall "authentication patterns"

# Store new insights
quaid store implementation "Added JWT validation to middleware"

# Update goals
quaid context complete-goal "Implement auth"

# Save context
quaid context save
```

#### Project Setup
```bash
# Initialize project
quaid init my-project --ai-enabled

# Configure preferences
quaid config set search.enable_semantic_search true
quaid config set ai.mode local

# Generate integrations
quaid integrate all

# Store initial knowledge
quaid store decision "Use JWT for API authentication"
```

#### Research and Documentation
```bash
# Search existing knowledge
quaid recall "database design patterns"

# Store research findings
quaid store concept --research "CQRS pattern for microservices"

# Find related information
quaid related cqrs-001

# Create documentation
quaid store api-doc --file ./docs/api-spec.md
```

---

## Global Options

All Quaid commands support these global options:

- `--help, -h`: Show help information
- `--version, -v`: Show version information
- `--verbose`: Enable verbose output
- `--quiet`: Suppress non-error output
- `--config <file>`: Use specific configuration file
- `--debug`: Enable debug mode
- `--no-color`: Disable colored output

### Environment Variables

- `QUAID_CONFIG_FILE`: Path to configuration file
- `QUAID_DEBUG`: Enable debug mode
- `QUAID_LOG_LEVEL`: Set logging level
- `QUAID_CACHE_DIR`: Set cache directory
- `QUAID_HOME`: Set Quaid home directory

---

## Next Steps

After mastering the CLI and API:

1. **Explore Advanced Features**: [08-Advanced-Features.md](08-Advanced-Features.md)
2. **Return to Main Guide**: [01-Vision-and-Problem.md](01-Vision-and-Problem.md)

---

**Previous**: [06-Search-and-Intelligence.md](06-Search-and-Intelligence.md) | **Next**: [08-Advanced-Features.md](08-Advanced-Features.md)