# Quaid

AI-powered knowledge management system built with Python, FastMCP, and modern tooling.

---

## Features

- **MCP Server**: FastMCP-based server for AI agent communication
- **Knowledge Graph**: NetworkX-powered relationship analysis and visualization
- **Search Engine**: Tantivy + Polars for fast full-text and metadata search
- **Rich CLI**: Beautiful command-line interface using Typer and Rich
- **Automated Cleanup**: Scheduled data retention and repository maintenance
- **Modular Architecture**: Clean separation with uv workspaces

## Structure

- `packages/`: Workspace with 8 focused libraries
- `plans/`: Comprehensive documentation and architecture plans
- `dependencies.txt`: Complete dependency breakdown by package

## Getting Started

```bash
# Install dependencies
uv sync

# Run CLI
uv run quaid --help
```

## Development

This project uses uv workspaces for modular development. See `workspace-design.md` for architecture details.

### Workspace Packages

- `quaid-config`: Configuration management with Pydantic Settings
- `quaid-core`: Core data structures and storage layer
- `quaid-search`: Full-text search with Tantivy and Polars
- `quaid-graph`: NetworkX knowledge graph operations
- `quaid-nlp`: spaCy natural language processing
- `quaid-mcp`: FastMCP server framework
- `quaid-cli`: Rich CLI interface with Typer and Rich
- `quaid-cleanup`: Automated data cleanup and retention

---

## Status

**Phase**: Foundation Setup ✅
**Next**: Package Implementation (starting with quaid-config and quaid-core)

See `plans/` directory for complete documentation and implementation plans.