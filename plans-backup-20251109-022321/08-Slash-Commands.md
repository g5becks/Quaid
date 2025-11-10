# 08 - Slash Commands

**AI Tool Integration via OpenSpec**

---

## 2. Slash Command Integration

### Priority Slash Commands

Create these commands first during `quaid init`:

#### `/quaid-store`
```markdown
---
name: /quaid-store
id: quaid-store
category: Quaid
description: Store a memory fragment with automatic classification
---

**Purpose**: Capture and store information as a queryable memory fragment.

**Steps**:
1. Ask user what to store (or accept direct content)
2. Run `quaid store --content "<text>"` or `quaid store --file <path>`
3. System auto-classifies into: concept, implementation, decision, documentation, reference
4. System auto-tags based on content analysis
5. System links to related fragments
6. Confirm storage with fragment ID

**Reference**:
- Store from clipboard: `quaid store --clipboard`
- Store with manual tags: `quaid store --tags "auth,jwt,security"`
- Store with specific type: `quaid store --type concept`
```

#### `/quaid-recall`
```markdown
---
name: /quaid-recall
id: quaid-recall
category: Quaid
description: Query memory with semantic search and context
---

**Purpose**: Retrieve relevant memories based on natural language query.

**Steps**:
1. Accept search query from user
2. Run `quaid search "<query>" --ai-rank`
3. Display top 5 results with relevance scores
4. If user wants more context, run `quaid related --to <fragment-id>`
5. Optionally inject results into conversation context

**Reference**:
- Recent memories: `quaid search --recent 7d`
- By type: `quaid search --type implementation "auth"`
- By tag: `quaid search --tags auth,security`
```

#### `/quaid-classify`
```markdown
---
name: /quaid-classify
id: quaid-classify
category: Quaid
description: Auto-classify and tag content
---

**Purpose**: Automatically categorize code, documentation, or text.

**Steps**:
1. Accept file path or content
2. Run `quaid classify --file <path>` or `quaid classify --content "<text>"`
3. Display classification results (type, tags, relationships)
4. Ask if user wants to store with these classifications
5. If yes, run `quaid store` with classification metadata

**Reference**:
- Classify multiple files: `quaid classify ./docs/*.md`
- Custom categories: `quaid classify --categories "bug,feature,refactor"`
```

#### `/quaid-graph`
```markdown
---
name: /quaid-graph
id: quaid-graph
category: Quaid
description: Visualize knowledge graph relationships
---

**Purpose**: Show connections between memory fragments.

**Steps**:
1. Accept starting fragment ID or concept
2. Run `quaid graph --from <id> --depth 2`
3. Display relationships in table format (NuShell's built-in formatting)
4. Optionally visualize with ASCII graph
5. Allow navigation to related fragments

**Reference**:
- Full graph stats: `quaid graph --stats`
- By relationship type: `quaid graph --type implements`
- Export graph: `quaid graph --export graph.json`
```

### Implementation Strategy

Generate slash commands automatically during initialization:

```bash
quaid init
  ↓
Detect AI tools: Cursor, Claude, Windsurf, etc.
  ↓
Generate tool-specific command files
  ↓
.cursor/commands/quaid-*.md
.claude/commands/quaid/*.md
.windsurf/workflows/quaid-*.md
etc.
```

---


---

**Previous**: [07-Worktrees.md](07-Worktrees.md) | **Next**: [09-Config.md](09-Config.md)
