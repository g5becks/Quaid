# 12 - Rich CLI Integration

**Beautiful and user-friendly command-line interface using Rich and Typer**

---

## Executive Summary

Rich integration transforms Quaid's CLI from a basic command-line tool into a visually appealing and user-friendly interface. By leveraging Rich's styling capabilities with Typer's CLI structure, we create an excellent developer experience with colored output, formatted tables, progress bars, and interactive elements while maintaining the simplicity of the command-line interface.

**Design Philosophy**: Provide rich visual feedback without overwhelming users, making the CLI both powerful and approachable for daily use.

---

## Rich CLI Architecture

### Visual Design System

```
┌─────────────────────────────────────────────────────────────┐
│                    Rich CLI Components                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Color Scheme & Branding                 │   │
│  │                                                     │   │
│  │  • Primary: Blue (#4A90E2) - Main actions           │   │
│  │  • Success: Green (#52C41A) - Successful operations │   │
│  │  • Warning: Orange (#FA8C16) - Warnings            │   │
│  │  • Error: Red (#F5222D) - Errors and failures      │   │
│  │  • Info: Cyan (#13C2C2) - Information              │   │
│  │  • Muted: Gray (#8C8C8C) - Secondary text          │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Rich UI Components                     │   │
│  │                                                     │   │
│  │  • Styled Tables - Data presentation                │   │
│  │  • Progress Bars - Long-running operations          │   │
│  │  • Panels & Boxes - Grouped content                │   │
│  │  • Syntax Highlighting - Code display               │   │
│  │  • Tree Views - Hierarchical data                   │   │
│  │  • Markdown Rendering - Rich text display           │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Console Management                      │   │
│  │                                                     │   │
│  │  • stdout vs stderr separation                      │   │
│  │  • Theme management                                 │   │
│  │  • Error handling with rich formatting              │   │
│  │  • Logging integration                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Rich Console Setup

```python
import typer
from rich.console import Console
from rich.theme import Theme
from rich.style import Style
from rich.logging import RichHandler
import logging
from typing import Optional

# Custom theme for consistent styling
custom_theme = Theme({
    "primary": Style(color="#4A90E2", bold=True),
    "success": Style(color="#52C41A", bold=True),
    "warning": Style(color="#FA8C16", bold=True),
    "error": Style(color="#F5222D", bold=True),
    "info": Style(color="#13C2C2"),
    "muted": Style(color="#8C8C8C", dim=True),
    "title": Style(color="#4A90E2", bold=True, underline=True),
    "subtitle": Style(color="#4A90E2", bold=True),
    "code": Style(color="#E6A23C", italic=True),
    "path": Style(color="#9254DE", italic=True),
    "highlight": Style(color="#F5222D", bold=True, reverse=True),
    "border": Style(color="#D9D9D9"),
})

# Main console for output
console = Console(theme=custom_theme)

# Error console for stderr
err_console = Console(theme=custom_theme, stderr=True)

# Configure logging with Rich
def setup_logging(verbose: bool = False) -> None:
    """Setup Rich logging handler"""
    logging.basicConfig(
        level="DEBUG" if verbose else "INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)]
    )

def get_console(error: bool = False) -> Console:
    """Get appropriate console"""
    return err_console if error else console
```

### 2. Rich Print Helpers

```python
from rich import print as rprint
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align
from rich.rule import Rule
from typing import Any, List, Optional

class RichPrinter:
    """Helper class for consistent rich printing"""

    @staticmethod
    def success(message: str, **kwargs) -> None:
        """Print success message"""
        console.print(f"✅ [success]{message}[/success]", **kwargs)

    @staticmethod
    def error(message: str, **kwargs) -> None:
        """Print error message"""
        err_console.print(f"❌ [error]{message}[/error]", **kwargs)

    @staticmethod
    def warning(message: str, **kwargs) -> None:
        """Print warning message"""
        console.print(f"⚠️  [warning]{message}[/warning]", **kwargs)

    @staticmethod
    def info(message: str, **kwargs) -> None:
        """Print info message"""
        console.print(f"ℹ️  [info]{message}[/info]", **kwargs)

    @staticmethod
    def title(text: str, subtitle: Optional[str] = None) -> None:
        """Print title with optional subtitle"""
        console.print(f"[title]{text}[/title]")
        if subtitle:
            console.print(f"[subtitle]{subtitle}[/subtitle]")
        console.print()

    @staticmethod
    def panel(content: Any, title: str = "", style: str = "primary") -> None:
        """Print content in a panel"""
        console.print(Panel(content, title=title, border_style=style))

    @staticmethod
    def rule(text: str = "", style: str = "primary") -> None:
        """Print horizontal rule"""
        console.print(Rule(text, style=style))

    @staticmethod
    def divider() -> None:
        """Print simple divider"""
        console.print(Rule(style="border"))

    @staticmethod
    def code_block(code: str, language: str = "python", title: Optional[str] = None) -> None:
        """Print syntax-highlighted code block"""
        from rich.syntax import Syntax

        syntax = Syntax(code, language, theme="monokai", line_numbers=True)

        if title:
            console.print(Panel(syntax, title=title, border_style="code"))
        else:
            console.print(syntax)

    @staticmethod
    def path_display(path: str, label: str = "Path") -> None:
        """Display file path"""
        console.print(f"{label}: [path]{path}[/path]")

    @staticmethod
    def key_value_list(items: List[tuple], title: str = "") -> None:
        """Display key-value pairs"""
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=0)
        table.add_column("Key", style="primary")
        table.add_column("Value")

        for key, value in items:
            table.add_row(f"{key}:", str(value))

        if title:
            console.print(Panel(table, title=title, border_style="primary"))
        else:
            console.print(table)

    @staticmethod
    def status_items(items: List[tuple], title: str = "") -> None:
        """Display status items with icons"""
        status_icons = {
            True: "✅",
            False: "❌",
            None: "➖",
            "warning": "⚠️",
            "info": "ℹ️"
        }

        content = []
        for status, text in items:
            if isinstance(status, bool):
                icon = status_icons[status]
                style = "success" if status else "error"
            else:
                icon = status_icons.get(status, "•")
                style = status if status in ["warning", "info"] else "muted"

            content.append(f"{icon} [{style}]{text}[/{style}]")

        if content:
            if title:
                RichPrinter.panel("\n".join(content), title=title)
            else:
                for line in content:
                    console.print(line)

    @staticmethod
    def command_suggestion(command: str, description: str = "") -> None:
        """Suggest a command to the user"""
        if description:
            console.print(f"💡 [info]Suggestion:[/info] [code]{command}[/code] - {description}")
        else:
            console.print(f"💡 [info]Suggestion:[/info] [code]{command}[/code]")
```

### 3. Rich Tables for Data Display

```python
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.tree import Tree
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from typing import List, Dict, Any, Optional
import time

class RichTables:
    """Helper class for creating rich tables"""

    @staticmethod
    def fragments_table(fragments: List[Dict], title: str = "Fragments") -> Table:
        """Create table for fragment data"""
        table = Table(title=title, show_header=True, header_style="title")

        table.add_column("ID", style="primary", no_wrap=True)
        table.add_column("Title", style="info", max_width=40)
        table.add_column("Type", style="warning")
        table.add_column("Importance", style="success")
        table.add_column("Tags", style="muted", max_width=20)
        table.add_column("Updated", style="info")

        for fragment in fragments:
            tags = ", ".join(fragment.get('tags', [])[:3])
            if len(fragment.get('tags', [])) > 3:
                tags += "..."

            table.add_row(
                fragment['id'][:15],
                fragment['title'][:37] + "..." if len(fragment['title']) > 40 else fragment['title'],
                fragment['type'],
                fragment.get('importance', 'unknown'),
                tags,
                fragment.get('updated', 'Unknown')[:10]
            )

        return table

    @staticmethod
    def search_results_table(results: List[Dict], query: str) -> Table:
        """Create table for search results"""
        table = Table(title=f"Search Results: [code]{query}[/code]", show_header=True)

        table.add_column("Score", style="success", justify="right")
        table.add_column("ID", style="primary")
        table.add_column("Title", style="info", max_width=50)
        table.add_column("Type", style="warning")
        table.add_column("Match", style="highlight", max_width=30)

        for result in results:
            match_text = result.get('snippet', 'No snippet')[:27]
            if len(result.get('snippet', '')) > 30:
                match_text += "..."

            table.add_row(
                f"{result.get('score', 0):.3f}",
                result['id'][:20],
                result['title'][:47] + "..." if len(result['title']) > 50 else result['title'],
                result['type'],
                match_text
            )

        return table

    @staticmethod
    def relationships_table(relationships: List[Dict], center_id: str) -> Table:
        """Create table for fragment relationships"""
        table = Table(title=f"Relationships for [primary]{center_id}[/primary]", show_header=True)

        table.add_column("Direction", style="info")
        table.add_column("Type", style="warning")
        table.add_column("Fragment", style="primary")
        table.add_column("Title", style="info", max_width=40)
        table.add_column("Confidence", style="success", justify="right")

        for rel in relationships:
            direction = "→ OUT" if rel.get('direction') == 'out' else "← IN"
            confidence = rel.get('confidence', 1.0)

            table.add_row(
                direction,
                rel['relationship'],
                rel['fragment_id'][:15],
                rel['title'][:37] + "..." if len(rel['title']) > 40 else rel['title'],
                f"{confidence:.2f}"
            )

        return table

    @staticmethod
    def cleanup_results_table(results: List[Dict]) -> Table:
        """Create table for cleanup results"""
        table = Table(title="Cleanup Results", show_header=True)

        table.add_column("Operation", style="primary")
        table.add_column("Status", style="success")
        table.add_column("Items Processed", style="info", justify="right")
        table.add_column("Items Deleted", style="warning", justify="right")
        table.add_column("Space Freed", style="success", justify="right")
        table.add_column("Duration", style="info", justify="right")

        for result in results:
            status = "✅ Success" if result.get('success') else "❌ Failed"
            space_freed = f"{result.get('space_freed_mb', 0):.1f}MB"
            duration = f"{result.get('duration_seconds', 0):.1f}s"

            table.add_row(
                result.get('operation', 'Unknown'),
                status,
                str(result.get('items_processed', 0)),
                str(result.get('items_deleted', 0)),
                space_freed,
                duration
            )

        return table

    @staticmethod
    def statistics_table(stats: Dict, title: str = "Statistics") -> Table:
        """Create table for statistics"""
        table = Table(title=title, show_header=False, box=None)

        table.add_column("Metric", style="primary")
        table.add_column("Value", style="info")

        for key, value in stats.items():
            # Format value based on type
            if isinstance(value, float):
                if value > 1000000:  # Large numbers in MB
                    formatted_value = f"{value/1024/1024:.1f}MB"
                elif value > 1000:  # Numbers in KB
                    formatted_value = f"{value/1024:.1f}KB"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            # Format key name
            formatted_key = key.replace('_', ' ').title()

            table.add_row(formatted_key, formatted_value)

        return table
```

### 4. Progress and Loading Indicators

```python
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    SpinnerColumn, TaskProgressColumn
)
from rich.live import Live
from rich.spinner import Spinner
from rich.status import Status
import threading
import time

class ProgressManager:
    """Manage progress indicators for long-running operations"""

    @staticmethod
    def show_progress(operation: str, total: int, update_func: callable) -> None:
        """Show progress bar for an operation"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[primary]{operation}[/primary]", total=total)

            def update_progress():
                for i in range(total):
                    time.sleep(0.1)  # Simulate work
                    progress.advance(task)
                    update_func(i, total)

            update_progress()

    @staticmethod
    def show_spinner(operation: str, func: callable, *args, **kwargs) -> Any:
        """Show spinner during operation"""
        with console.status(f"[primary]{operation}[/primary]", spinner="dots"):
            return func(*args, **kwargs)

    @staticmethod
    def show_multi_progress(tasks: List[Dict]) -> None:
        """Show multiple progress bars"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task_list = []

            for task_info in tasks:
                task = progress.add_task(
                    f"[primary]{task_info['name']}[/primary]",
                    total=task_info['total']
                )
                task_list.append((task, task_info))

            # Simulate work
            for task, task_info in task_list:
                for i in range(task_info['total']):
                    time.sleep(0.05)
                    progress.advance(task)

    @staticmethod
    def create_live_updater(update_func: callable, initial_content: str = "") -> Live:
        """Create live updating content"""
        class LiveUpdater:
            def __init__(self, update_func: callable):
                self.update_func = update_func
                self.live = Live(initial_content, console=console, refresh_per_second=4)

            def start(self):
                self.live.start()

            def stop(self):
                self.live.stop()

            def update(self, content: str):
                self.live.update(content)

        return LiveUpdater(update_func)

class LoadingAnimations:
    """Various loading animations"""

    @staticmethod
    def loading_database():
        """Database loading animation"""
        with Status("Loading database...", spinner="aesthetic"):
            time.sleep(2)

    @staticmethod
    def indexing_files():
        """File indexing animation"""
        with Status("Indexing files...", spinner="dots12"):
            time.sleep(1.5)

    @staticmethod
    def processing_search():
        """Search processing animation"""
        with Status("Processing search query...", spinner="bouncingBar"):
            time.sleep(1)

    @staticmethod
    def generating_report():
        """Report generation animation"""
        with Status("Generating report...", spinner="hamburger"):
            time.sleep(2.5)
```

### 5. Rich CLI Commands Integration

```python
import typer
from typing import Optional, List
from pathlib import Path

# Initialize Rich printer
printer = RichPrinter()
tables = RichTables()
progress = ProgressManager()

# Main CLI app
app = typer.Typer(
    name="quaid",
    help="[primary]Quaid[/primary] - [info]AI-powered knowledge management system[/info]",
    no_args_is_help=True,
    rich_markup_mode="rich"
)

@app.command()
def init(
    path: Optional[str] = typer.Argument(
        ".",
        help="[info]Path to initialize Quaid repository[/info]"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="[warning]Force initialization even if directory exists[/warning]"
    )
) -> None:
    """Initialize a new Quaid knowledge repository"""

    printer.title("🚀 Quaid Repository Initialization")

    try:
        # Check if directory exists
        repo_path = Path(path)
        if repo_path.exists() and any(repo_path.iterdir()) and not force:
            printer.error(f"Directory [path]{path}[/path] is not empty. Use --force to override.")
            raise typer.Exit(1)

        printer.info(f"Initializing repository at: [path]{path}[/path]")

        # Show progress
        with console.status("[primary]Creating repository structure...[/primary]", spinner="dots"):
            # Simulate initialization steps
            time.sleep(0.5)
            printer.success("✓ Created .quaid directory")
            time.sleep(0.5)
            printer.success("✓ Created memory structure")
            time.sleep(0.5)
            printer.success("✓ Initialized indexes")
            time.sleep(0.5)
            printer.success("✓ Created configuration")

        # Show completion message
        printer.divider()
        printer.success("🎉 Repository initialized successfully!")

        # Show next steps
        printer.panel(
            "[info]Next steps:[/info]\n"
            "• [code]quaid add 'My first fragment'[/code] - Add your first fragment\n"
            "• [code]quaid search 'query'[/code] - Search your knowledge\n"
            "• [code]quaid tui[/code] - Launch the terminal interface\n"
            title="🎯 Getting Started"
        )

    except Exception as e:
        printer.error(f"Failed to initialize repository: {e}")
        raise typer.Exit(1)

@app.command()
def search(
    query: str = typer.Argument(..., help="[info]Search query[/info]"),
    limit: int = typer.Option(10, "--limit", "-l", help="[info]Maximum results to return[/info]"),
    type_filter: Optional[str] = typer.Option(None, "--type", "-t", help="[info]Filter by fragment type[/info]"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="[info]Show detailed results[/info]")
) -> None:
    """Search through your knowledge fragments"""

    printer.title(f"🔍 Searching for: [code]{query}[/code]")

    # Show search progress
    with console.status("[primary]Searching knowledge base...[/primary]", spinner="bouncingBar"):
        # Simulate search
        time.sleep(1)

        # Mock results
        results = [
            {
                "id": "jwt-auth-001",
                "title": "JWT Authentication Strategy",
                "type": "concept",
                "score": 0.92,
                "snippet": "JWT tokens provide stateless authentication..."
            },
            {
                "id": "session-mgmt-002",
                "title": "Session Management Implementation",
                "type": "implementation",
                "score": 0.87,
                "snippet": "Server-side sessions with Redis storage..."
            }
        ]

    if not results:
        printer.warning("No results found")
        printer.command_suggestion(
            f"quaid add '{query}'",
            "Create a fragment about this topic"
        )
        return

    # Display results
    console.print(tables.search_results_table(results, query))

    # Show summary
    printer.divider()
    printer.info(f"Found [primary]{len(results)}[/primary] results")

    if verbose:
        printer.panel(
            f"[info]Search metrics:[/info]\n"
            f"Query: [code]{query}[/code]\n"
            f"Limit: {limit}\n"
            f"Type filter: {type_filter or 'none'}\n"
            f"Search time: 0.23s",
            title="📊 Search Details"
        )

@app.command()
def add(
    title: str = typer.Argument(..., help="[info]Fragment title[/info]"),
    content: Optional[str] = typer.Option(None, "--content", "-c", help="[info]Fragment content[/info]"),
    type: str = typer.Option("concept", "--type", "-t", help="[info]Fragment type[/info]"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", help="[info]Tags for the fragment[/info]"),
    importance: str = typer.Option("medium", "--importance", "-i", help="[info]Importance level[/info]")
) -> None:
    """Add a new knowledge fragment"""

    printer.title(f"📝 Adding Fragment: [info]{title}[/info]")

    # Collect content if not provided
    if not content:
        content = typer.prompt("Enter fragment content", confirmation_prompt=False)

    # Show fragment details
    printer.panel(
        f"[primary]Title:[/primary] {title}\n"
        f"[primary]Type:[/primary] {type}\n"
        f"[primary]Importance:[/primary] {importance}\n"
        f"[primary]Tags:[/primary] {', '.join(tags) if tags else 'none'}\n"
        f"[primary]Content:[/primary] {content[:100]}{'...' if len(content) > 100 else ''}",
        title="📋 Fragment Details"
    )

    # Confirm addition
    if typer.confirm("Add this fragment?"):
        with console.status("[primary]Adding fragment to repository...[/primary]", spinner="dots"):
            time.sleep(1)

        printer.success("✅ Fragment added successfully!")
        printer.info(f"Fragment ID: [code]fragment-123[/code]")

        printer.command_suggestion(
            "quaid search 'your query'",
            "Search for your fragment"
        )
    else:
        printer.warning("Fragment addition cancelled")

@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="[info]Show detailed status[/info]")
) -> None:
    """Show repository status and statistics"""

    printer.title("📊 Repository Status")

    # Mock statistics
    stats = {
        "total_fragments": 127,
        "total_size_mb": 45.2,
        "last_updated": "2025-01-08",
        "search_index_size": 12.3,
        "graph_nodes": 127,
        "graph_edges": 234
    }

    # Display statistics table
    console.print(tables.statistics_table(stats, "Repository Statistics"))

    # Show health indicators
    health_items = [
        (True, "Repository is healthy"),
        (True, "Indexes are up to date"),
        (False, "Git has uncommitted changes"),
        ("warning", "Cleanup scheduled in 2 days")
    ]

    printer.status_items(health_items, "Health Indicators")

    if verbose:
        printer.divider()
        printer.panel(
            "[info]System Information:[/info]\n"
            f"Python: 3.11.0\n"
            f"Quaid: 1.0.0\n"
            f"Git: git version 2.39.0\n"
            f"OS: Darwin 23.0.0",
            title="💻 System Details"
        )

@app.command()
def cleanup(
    dry_run: bool = typer.Option(True, "--dry-run", help="[info]Show what would be deleted without actually deleting[/info]"),
    force: bool = typer.Option(False, "--force", "-f", help="[warning]Skip confirmation prompts[/warning]")
) -> None:
    """Clean up old fragments and optimize repository"""

    if dry_run:
        printer.title("🧹 Cleanup Dry Run")
        printer.warning("This is a dry run - no files will be deleted")
    else:
        printer.title("🧹 Repository Cleanup")
        printer.warning("This will permanently delete old fragments")

    printer.divider()

    # Mock cleanup analysis
    fragments_to_delete = 23
    space_to_free = 15.7

    # Show what will be cleaned up
    cleanup_items = [
        (True, f"{fragments_to_delete} old fragments"),
        (True, f"{space_to_free:.1f}MB of storage space"),
        (True, "Optimized search indexes"),
        (True, "Compressed git history")
    ]

    printer.status_items(cleanup_items, "Cleanup Plan")

    if not dry_run and not force:
        if not typer.confirm("Proceed with cleanup?"):
            printer.warning("Cleanup cancelled")
            return

    # Run cleanup
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        # Simulate cleanup steps
        tasks = [
            {"name": "Analyzing fragments", "total": 100},
            {"name": "Deleting old fragments", "total": fragments_to_delete},
            {"name": "Optimizing indexes", "total": 100},
            {"name": "Git maintenance", "total": 100}
        ]

        for task_info in tasks:
            task = progress.add_task(f"[primary]{task_info['name']}[/primary]", total=task_info['total'])
            for i in range(task_info['total']):
                time.sleep(0.02)
                progress.advance(task)

    # Show results
    printer.divider()
    if dry_run:
        printer.success(f"Would delete [primary]{fragments_to_delete}[/primary] fragments")
        printer.success(f"Would free [primary]{space_to_free:.1f}MB[/primary] of space")
        printer.info("Run without --dry-run to actually clean up")
    else:
        printer.success(f"Deleted [primary]{fragments_to_delete}[/primary] fragments")
        printer.success(f"Freed [primary]{space_to_free:.1f}MB[/primary] of space")
        printer.success("✅ Repository cleanup completed")

# Error handler for rich error display
def handle_errors(error: Exception) -> None:
    """Handle errors with rich formatting"""
    printer.error(f"Command failed: {error}")

    # Show helpful suggestions based on error type
    if "Permission denied" in str(error):
        printer.command_suggestion("chmod +x .quaid", "Check directory permissions")
    elif "not found" in str(error):
        printer.command_suggestion("quaid init", "Initialize a Quaid repository")
    else:
        printer.command_suggestion("quaid --help", "Get help with commands")

if __name__ == "__main__":
    try:
        setup_logging()
        app()
    except KeyboardInterrupt:
        printer.warning("\nOperation cancelled by user")
        raise typer.Exit(1)
    except Exception as e:
        handle_errors(e)
        raise typer.Exit(1)
```

---

## Usage Examples

### Basic Command Execution

```bash
# Show help with rich formatting
$ quaid --help

╭─────────────────────────────────────────────────────────────╮
│                 Quaid - AI-powered knowledge management system     │
╰─────────────────────────────────────────────────────────────╯

🚀  Usage: quaid [OPTIONS] COMMAND [ARGS]...

📚  Commands:
  init    Initialize a new Quaid knowledge repository
  search  Search through your knowledge fragments
  add     Add a new knowledge fragment
  status  Show repository status and statistics
  cleanup Clean up old fragments and optimize repository

💡  For help with a specific command: quaid COMMAND --help
```

### Rich Search Results

```bash
$ quaid search "authentication"

╭─────────────────────────────────────────────────────────────╮
│                 Search Results: authentication               │
├────┬─────────────────┬─────────────────────────┬─────────────┤
│Score│      ID         │          Title          │    Type     │
├────┼─────────────────┼─────────────────────────┼─────────────┤
│0.92│ jwt-auth-001    │JWT Authentication     │concept      │
│    │                 │Strategy                │             │
│0.87│ session-mgmt-002│Session Management      │implementation│
│    │                 │Implementation          │             │
└────┴─────────────────┴─────────────────────────┴─────────────┘

─────────────────────────────────────────────────────────────
ℹ️  Found 2 results
```

### Rich Status Display

```bash
$ quaid status --verbose

╭─────────────────────────────────────────────────────────────╮
│                    Repository Status                        │
├─────────────────────────────────────────────────────────────┤
│Metric              │Value                                  │
├─────────────────────────────────────────────────────────────┤
│Total Fragments    │127                                    │
│Total Size          │45.2MB                                 │
│Last Updated        │2025-01-08                             │
│Search Index Size   │12.3MB                                 │
│Graph Nodes         │127                                    │
│Graph Edges         │234                                    │
└─────────────────────────────────────────────────────────────┘

─────────────────────────────────────────────────────────────
Health Indicators
─────────────────────────────────────────────────────────────
✅  Repository is healthy
✅  Indexes are up to date
❌  Git has uncommitted changes
⚠️  Cleanup scheduled in 2 days
```

### Rich Progress Indicators

```bash
$ quaid cleanup

🧹 Repository Cleanup
⚠️  This will permanently delete old fragments

─────────────────────────────────────────────────────────────
Cleanup Plan
─────────────────────────────────────────────────────────────
✅  23 old fragments
✅  15.7MB of storage space
✅  Optimized search indexes
✅  Compressed git history

Analyzing fragments ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02
Deleting old fragments ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23/23 0:00:01
Optimizing indexes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03
Git maintenance ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:04

─────────────────────────────────────────────────────────────
✅  Deleted 23 fragments
✅  Freed 15.7MB of space
✅  Repository cleanup completed
```

---

## Configuration

### Rich Configuration in pyproject.toml

```toml
[tool.quaid.rich]
# Enable rich output
enabled = true

# Theme configuration
theme = "default"  # default, dark, light, custom

# Color scheme
colors = {
    primary = "#4A90E2",
    success = "#52C41A",
    warning = "#FA8C16",
    error = "#F5222D",
    info = "#13C2C2",
    muted = "#8C8C8C"
}

# Display options
show_progress_bars = true
show_status_indicators = true
show_tables = true
max_table_width = 100

# Console options
emoji = true
markdown_rendering = true
syntax_highlighting = true
```

---

## Benefits of Rich Integration

1. **Better User Experience**: Visually appealing output with colors and formatting
2. **Clear Information Hierarchy**: Titles, panels, and tables organize information
3. **Status Feedback**: Progress bars and status indicators for long operations
4. **Error Clarity**: Rich error formatting with helpful suggestions
5. **Professional Appearance**: Consistent styling and branding
6. **Accessibility**: Clear visual structure and contrast
7. **Debugging Friendly**: Verbose modes with detailed information
8. **Cross-Platform**: Consistent appearance across terminals

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.dependencies]
# ... existing dependencies ...
rich = "^13.7"                      # Rich text and beautiful formatting

# Update all group
all = [
    "quaid[dev,docs,graph,ai,cleanup]"
]
```

---

**Previous**: [11-Data-Cleanup-and-Retention.md](11-Data-Cleanup-and-Retention.md) | **Next**: [13-MCP-Server-Reference.md](13-MCP-Server-Reference.md)