# 07 - Worktrees

**Multi-Worktree Support and Workflows**

---

## 5. Worktree Support Implementation

### Problem Statement

Developers often work on multiple features simultaneously using Git worktrees. Memory collected in one worktree context may not be relevant to another, but selective integration is valuable.

### Solution Architecture

```
Memory Isolation Strategy:
├── Main worktree: Stable, production memories
├── Feature worktree: Experimental, context-specific
└── Integration: Selective merge with conflict resolution
```

### Implementation Details

#### Fragment Metadata

Add `worktree` field to fragment frontmatter:

```yaml
---
id: "20250108-new-feature-001"
worktree: "feature/authentication"
worktree_created: "feature/authentication"
worktree_updated: ["feature/authentication", "main"]
---
```

#### Worktree Detection

```bash
# Automatically detect current worktree
git rev-parse --show-toplevel
git branch --show-current
```

#### Query Commands

```bash
# Search only current worktree
quaid search "auth" --worktree current

# Search across all worktrees
quaid search "auth" --worktree all

# Search specific worktree
quaid search "auth" --worktree feature/new-api
```

#### Integration Workflow

```bash
# List worktree-specific memories
quaid list --worktree feature/authentication

# Review before integration
quaid diff --worktree feature/authentication main

# Selective integration
quaid integrate --from feature/authentication --to main
  → Prompts for each fragment:
    - Keep in feature worktree only
    - Copy to main
    - Merge with existing
    - Skip

# Auto-integration (with rules)
quaid integrate --from feature/auth --policy auto-merge
  → Uses configured rules:
    - type=concept → always integrate
    - type=implementation → review required
    - type=decision → always integrate
```

#### Configuration

```toml
[worktree]
auto_detect = true
default_scope = "current"  # or "all"

[worktree.integration]
auto_merge_types = ["concept", "decision", "documentation"]
review_required_types = ["implementation"]
conflict_resolution = "prompt"  # or "keep-both", "keep-newer"
```

---


---

**Previous**: [06-Graph.md](06-Graph.md) | **Next**: [08-Slash-Commands.md](08-Slash-Commands.md)
