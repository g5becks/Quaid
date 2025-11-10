# 05 - Core Features

**Comprehensive guide to Quaid's core functionality for managing development knowledge**

---

## Overview

Quaid provides a rich set of core features designed to capture, organize, and retrieve development knowledge efficiently. This guide covers the essential functionality that forms the foundation of Quaid's knowledge management system.

### Core Capabilities

1. **Fragment Management**: Create, store, and organize knowledge fragments
2. **Multi-Worktree Support**: Isolate and integrate knowledge across git worktrees
3. **Context Management**: Maintain session context and working memory
4. **Knowledge Graph**: Track relationships between concepts and implementations
5. **Version Control Integration**: Git-native storage and synchronization

---

## Fragment Management

### Understanding Fragments

Fragments are the fundamental unit of knowledge in Quaid. Each fragment is a structured markdown document containing:

- **Rich Metadata**: Type, tags, importance, relationships
- **Structured Content**: Headings, code blocks, decisions, examples
- **Auto-Extracted Information**: Entities, links, file references
- **Relationships**: Links to related fragments and implementations

### Creating Fragments

#### Method 1: Direct Storage

```bash
# Store a concept
quaid store concept "JWT provides stateless authentication for microservices"

# Store an implementation
quaid store implementation "JWT token validation using PyJWT library"

# Store with specific tags
quaid store decision --tags authentication,security "Use JWT over session-based auth"

# Store from file
quaid store implementation --file ./docs/jwt-implementation.md

# Store from clipboard
quaid store concept --clipboard

# Store with specific type
quaid store --type=pattern "Always validate JWT signature before processing claims"
```

#### Method 2: Interactive Creation

```bash
# Interactive fragment creation
quaid create

# Output:
# ? Fragment type: concept
# ? Title: JWT Authentication Strategy
# ? Tags: authentication, jwt, security
# ? Importance: high
# Enter fragment content (Ctrl-D to finish):
# # JWT Authentication Strategy
#
# > **Decision**: Use JWT for stateless authentication
# > **Date**: 2025-11-09
# > **Status**: Approved
#
# ## Rationale
# Our microservices architecture requires stateless authentication...
#
# ^D
# ✓ Created fragment: 2025-11-09-jwt-strategy-001.md
```

#### Method 3: AI-Assisted Creation

```bash
# Let AI help structure your content
quaid store --ai-assisted implementation "Implement JWT validation in Express.js"

# AI will:
# 1. Parse your input
# 2. Suggest appropriate structure
# 3. Extract key concepts
# 4. Generate code examples
# 5. Create comprehensive fragment
```

### Fragment Types

Quaid supports several built-in fragment types, each with specific purposes:

#### Concept Fragments
```markdown
---
id: "2025-11-09-jwt-concept-001"
type: concept
title: "JWT Authentication"
tags: [authentication, jwt, security]
importance: high
---

# JWT Authentication

## Overview
JSON Web Tokens (JWT) are an open standard for securely transmitting information between parties as JSON objects.

## Key Characteristics
- **Stateless**: No server-side session storage required
- **Self-Contained**: Contains all necessary information
- **Digitally Signed**: Can be verified and trusted
- **Compact**: Small in size compared to alternatives

## Use Cases
- Authentication in distributed systems
- Information exchange between services
- Secure API access tokens
```

#### Implementation Fragments
```markdown
---
id: "2025-11-09-jwt-impl-001"
type: implementation
title: "JWT Token Validation Implementation"
tags: [authentication, jwt, python, security]
code_languages: [python]
has_code: true
---

# JWT Token Validation Implementation

## Code Example

```python
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app

def validate_jwt(token: str) -> dict:
    """Validate JWT token and return payload"""
    try:
        payload = jwt.decode(
            token,
            current_app.config['JWT_SECRET'],
            algorithms=['HS256']
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")

def jwt_required(f):
    """Decorator to require JWT authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')

        if auth_header:
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                raise ValueError("Bearer token malformed")

        if not token:
            raise ValueError("Token is missing")

        try:
            payload = validate_jwt(token)
            request.current_user = payload
        except ValueError as e:
            return jsonify({"error": str(e)}), 401

        return f(*args, **kwargs)

    return decorated_function
```

## Integration Points
- Flask application setup
- Secret key management
- Error handling middleware
```

#### Decision Fragments
```markdown
---
id: "2025-11-09-auth-decision-001"
type: decision
title: "Authentication Strategy Decision"
tags: [authentication, decision, architecture]
has_decision: true
importance: high
---

# Authentication Strategy Decision

> **Decision**: Use JWT tokens for API authentication instead of session cookies
> **Date**: 2025-11-09
> **Status**: Approved
> **Decision Maker**: Architecture Team
> **Impact**: High

## Problem Statement
Our microservices architecture requires an authentication mechanism that:
- Scales horizontally without session affinity
- Supports mobile applications
- Enables independent service validation
- Maintains security best practices

## Considered Alternatives

1. **Session Cookies**
   - Pros: Simple implementation, automatic security
   - Cons: Requires session storage, doesn't scale well

2. **OAuth 2.0**
   - Pros: Industry standard, robust security
   - Cons: Complex implementation, overkill for internal APIs

3. **API Keys**
   - Pros: Simple, stateless
   - Cons: No user context, harder to revoke

## Decision Rationale

JWT was selected because it:
- ✅ Provides stateless authentication
- ✅ Supports horizontal scaling
- ✅ Works well with mobile apps
- ✅ Includes expiration and refresh mechanisms
- ✅ Can carry user context/payload

## Implementation Requirements
- Use RS256 algorithm for production
- Implement token refresh mechanism
- Store public keys securely
- Set appropriate token TTL (15 minutes access, 7 days refresh)

## Consequences
- **Positive**: Scalable authentication, mobile-friendly
- **Negative**: Token revocation complexity, larger request size
- **Neutral**: Requires key management infrastructure
```

### Managing Fragments

#### Listing Fragments

```bash
# List all fragments
quaid list

# Output:
# ╭─────────────┬──────────────────────┬─────────┬─────────────┬──────────────╮
# │     ID      │        Title         │  Type   │   Tags      │   Created    │
# ├─────────────┼──────────────────────┼─────────┼─────────────┼──────────────┤
# │ jwt-001     │ JWT Authentication   │ concept │ auth,jwt    │ 2 hours ago  │
# │ jwt-002     │ JWT Implementation  │ impl   │ auth,jwt,py │ 2 hours ago  │
# │ auth-003    │ Auth Decision       │ decision│ auth,arch   │ 1 hour ago   │
# ╰─────────────┴──────────────────────┴─────────┴─────────────┴──────────────╯

# List with filtering
quaid list --type concept
quaid list --tags authentication,jwt
quaid list --importance high
quaid list --created-since 2025-11-08
quaid list --search JWT

# Detailed listing
quaid list --detailed
```

#### Viewing Fragments

```bash
# View specific fragment
quaid get jwt-001

# View with syntax highlighting
quaid get jwt-001 --format pretty

# View as JSON (metadata only)
quaid get jwt-001 --format json

# View only content
quaid get jwt-001 --content-only

# View specific sections
quaid get jwt-001 --sections overview,implementation
```

#### Updating Fragments

```bash
# Update fragment content
quaid update jwt-001 --content "Updated content here"

# Update metadata
quaid update jwt-001 --title "New Title"
quaid update jwt-001 --tags authentication,jwt,updated
quaid update jwt-001 --importance high

# Append content
quaid update jwt-001 --append "Additional information"

# Update specific section
quaid update jwt-001 --section implementation --content "New implementation"
```

#### Deleting Fragments

```bash
# Delete fragment (with confirmation)
quaid delete jwt-001

# Force delete without confirmation
quaid delete jwt-001 --force

# Soft delete (move to archive)
quaid delete jwt-001 --archive
```

---

## Multi-Worktree Support

### Understanding Worktrees

Git worktrees allow you to check out multiple branches simultaneously. Quaid provides worktree-aware memory management to:

- **Isolate Context**: Keep memories specific to each worktree
- **Share Knowledge**: Selectively merge important information
- **Maintain Context**: Track which worktree created each fragment

### Worktree Detection

Quaid automatically detects the current worktree:

```bash
# Show current worktree information
quaid worktree status

# Output:
# Current worktree: feature/authentication
# Base repository: /Users/user/projects/my-app
# Branch: feature/authentication
# Isolation: partial
# Memories: 12 fragments
```

### Worktree-Scoped Operations

```bash
# Search in current worktree only
quaid recall "authentication" --worktree current

# Search across all worktrees
quaid recall "authentication" --worktree all

# Search in specific worktree
quaid recall "authentication" --worktree feature/api

# List memories by worktree
quaid list --group-by worktree
```

### Worktree Integration

#### Manual Integration

```bash
# List worktree-specific memories
quaid list --worktree feature/authentication

# Review differences between worktrees
quaid diff --worktree feature/authentication main

# Integrate memories from feature worktree
quaid integrate --from feature/authentication --to main

# Output:
# Found 5 fragments to integrate:
# 1. [concept] JWT vs Session Comparison (keep, copy, merge, skip)
# 2. [implementation] Auth Middleware Code (keep, copy, merge, skip)
# 3. [decision] Use RS256 Algorithm (keep, copy, merge, skip)
# 4. [pattern] Error Handling Pattern (keep, copy, merge, skip)
# 5. [reference] JWT Specification Link (keep, copy, merge, skip)
```

#### Automatic Integration Rules

Configure automatic integration based on fragment types and importance:

```toml
# .quaid/config.toml
[worktree.integration]
# Automatically integrate these types
auto_merge_types = ["concept", "decision", "documentation", "pattern"]

# Require review for these types
review_required_types = ["implementation", "reference"]

# Conflict resolution strategy
conflict_resolution = "prompt"  # prompt, keep-both, keep-newer, keep-older

# Integration rules based on importance
[worktree.integration.rules]
high_importance = "prompt"  # Always ask for high importance
medium_importance = "auto"  # Auto-merge medium importance
low_importance = "skip"    # Skip low importance
```

### Worktree Configuration

```toml
# .quaid/config.toml
[worktree]
# Worktree detection and management
auto_detect = true
default_scope = "current"  # current, all, specific
isolation = "partial"  # full, partial, none

# Memory sharing
[worktree.memory]
share_by_default = false
inherit_global = true
sync_on_switch = true
share_types = ["concept", "decision", "pattern"]
keep_private = ["implementation", "troubleshooting"]

# Integration settings
[worktree.integration]
auto_merge = false
merge_on_branch_switch = true
conflict_resolution = "prompt"
```

---

## Context Management

### Understanding Context

Context management maintains the working memory and session state for:

- **Active Goals**: Current objectives and tasks
- **Working State**: Files being edited, functions being implemented
- **Key Decisions**: Recent decisions and rationale
- **Session History**: Recent interactions and their context

### Session Context

#### Viewing Current Context

```bash
# Show current session context
quaid context status

# Output:
# Session: conv-2025-11-09-001
# Started: 2 hours ago
# Messages: 15

# Active Goals:
# ✅ Implement JWT authentication
# 🔄 Add error handling for token validation
# ⏳ Write tests for auth middleware

# Current Files:
# - src/auth/jwt.py (editing function: validate_token)
# - src/middleware/auth.py (recently modified)

# Key Decisions:
# - Use RS256 algorithm for JWT signatures
# - Token TTL: 15 minutes access, 7 days refresh

# Pending Actions:
# - Test token refresh flow
# - Update API documentation
# - Add logging to auth events
```

#### Managing Goals

```bash
# Add a new goal
quaid context add-goal "Implement token refresh mechanism"
quaid context add-goal "Add rate limiting to auth endpoints" --priority high

# List all goals
quaid context list-goals

# Update goal status
quaid context update-goal jwt-auth --status complete
quaid context update-goal token-refresh --status in-progress

# Mark goal as complete
quaid context complete-goal "Add rate limiting to auth endpoints"

# Remove goal
quaid context remove-goal "Write tests for auth middleware"
```

#### Managing Decisions

```bash
# Log a decision
quaid context log-decision "Use Redis for token blacklist" "Fast lookup, built-in TTL"
quaid context log-decision "Set token TTL to 15 minutes" "Balance security and UX"

# View recent decisions
quaid context list-decisions

# Search decisions
quaid context search-decisions "Redis"
```

#### Managing Constraints

```bash
# Add constraints
quaid context add-constraint "All API endpoints must be rate-limited"
quaid context add-constraint "Never log sensitive user data"
quaid context add-constraint "Token validation must be synchronous"

# View constraints
quaid context list-constraints

# Remove constraint
quaid context remove-constraint "All API endpoints must be rate-limited"
```

### Session Management

#### Session History

```bash
# Show recent session history
quaid context history

# Search session history
quaid context search-history "authentication"

# Clear session history
quaid context clear-history
```

#### Session Persistence

```bash
# Save current session
quaid context save

# Restore previous session
quaid context restore --session-id conv-2025-11-08-003

# List available sessions
quaid context list-sessions

# Output:
# Available sessions:
# conv-2025-11-09-001  "JWT Authentication"     (active, 2h ago, 15 messages)
# conv-2025-11-08-003  "API Refactoring"       (1d ago, 23 messages)
# conv-2025-11-07-002  "Database Migration"    (2d ago, 18 messages)
```

#### Context Summarization

```bash
# Generate session summary
quaid context summarize

# Auto-summarize when context grows
quaid context auto-summarize --threshold 20

# Archive current session as fragment
quaid context archive --title "JWT Authentication Session"
```

### Context Commands

```bash
# Context management commands
quaid context status          # Show current context
quaid context summary         # Generate summary
quaid context clear           # Clear all context
quaid context save            # Save current state
quaid context restore         # Restore previous state

# Goal management
quaid context add-goal <goal>
quaid context complete-goal <goal-id>
quaid context list-goals

# Decision logging
quaid context log-decision <decision> [rationale]
quaid context list-decisions

# Constraint management
quaid context add-constraint <constraint>
quaid context list-constraints

# History and search
quaid context history
quaid context search-history <query>
```

---

## Knowledge Graph (NetworkX-Powered)

### Understanding the Knowledge Graph

The knowledge graph uses NetworkX to track and analyze relationships between fragments:

- **Implements**: Code implements concepts or decisions
- **References**: Fragments reference other information
- **Depends-on**: Fragments depend on other fragments
- **Related-to**: General relationship between concepts
- **Supersedes**: New fragments replace old ones

**NetworkX Integration**: The graph is powered by NetworkX for advanced algorithms including path finding, centrality analysis, community detection, and intelligent relationship scoring while maintaining git-native JSONL storage.

### Viewing Relationships

```bash
# Show relationships for a fragment
quaid graph show jwt-001

# Output:
# Fragment: JWT Authentication Concept (jwt-001)
#
# Relationships:
# ↓ implements (2)
#   - jwt-002 (JWT Implementation)
#   - jwt-005 (Middleware Implementation)
#
# ↑ referenced-by (3)
#   - auth-003 (Auth Decision)
#   - api-012 (API Documentation)
#   - test-007 (Authentication Tests)
#
# ↔ related-to (4)
#   - session-001 (Session Management)
#   - oauth-002 (OAuth2 Implementation)
#   - security-003 (Security Best Practices)
#   - microservice-005 (Service Communication)

# Show related fragments
quaid graph related jwt-001 --depth 2

# Show implementation chain
quaid graph implements jwt-001

# Show dependency graph
quaid graph dependencies jwt-001
```

### Managing Relationships

#### Manual Relationship Management

```bash
# Add manual relationship
quaid graph relate jwt-001 --to session-001 --type related-to
quaid graph relate jwt-002 --to jwt-001 --type implements

# Remove relationship
quaid graph unrelate jwt-001 --to session-001 --type related-to

# Update relationship type
quaid graph update jwt-001 --to jwt-002 --type implements
```

#### Automatic Relationship Detection

Quaid automatically detects relationships based on:

- **Content Analysis**: Mentions of fragment IDs or titles
- **Code References**: File paths and function names
- **Entity Co-occurrence**: Shared entities and concepts
- **Temporal Patterns**: Fragments created in sequence

```toml
# Configure automatic relationship detection
[graph]
# Enable automatic detection
auto_detect = true

# Detection methods
detect_mentions = true        # Detect @fragment-id mentions
detect_references = true      # Detect markdown links
detect_entities = true        # Detect shared entities
detect_temporal = true        # Detect temporal patterns

# Relationship confidence thresholds
[graph.thresholds]
mention_confidence = 0.9      # High confidence for mentions
entity_confidence = 0.7       # Medium confidence for entities
temporal_confidence = 0.5      # Lower confidence for temporal
```

### Graph Visualization

```bash
# Generate graph visualization
quaid graph visualize --output graph.png
quaid graph visualize --format svg --output graph.svg
quaid graph visualize --subgraph authentication --output auth-graph.png

# Interactive graph (if supported)
quaid graph visualize --interactive
```

### Graph Analytics

```bash
# Show graph statistics
quaid graph stats

# Output:
# Knowledge Graph Statistics:
# Nodes: 127 fragments
# Edges: 234 relationships
# Average degree: 3.68
# Connected components: 3
# Largest component: 118 fragments

# Find central nodes (high connectivity)
quaid graph central --top 10

# Find isolated fragments
quaid graph isolated

# Find relationship paths
quaid graph path jwt-001 session-003
```

### Advanced NetworkX Features

#### Centrality Analysis

```bash
# Calculate importance scores using multiple algorithms
quaid graph centrality jwt-001

# Output:
# Fragment: JWT Authentication Concept (jwt-001)
#
# Centrality Metrics:
# PageRank Score: 0.0842 (measures overall importance)
# Betweenness: 0.156 (connects different concepts)
# In-Degree: 0.067 (referenced by others)
# Out-Degree: 0.089 (references others)
# Combined Score: 0.092 (overall importance)

# Show top important fragments
quaid graph importance --top 10

# Output:
# Top 10 Most Important Fragments:
# 1. JWT Authentication Strategy (0.142) - concept
# 2. API Security Architecture (0.128) - concept
# 3. Session Management Implementation (0.115) - implementation
# 4. OAuth2 Provider Setup (0.098) - implementation
# 5. Database Migration Strategy (0.087) - decision
```

#### Path Finding and Confidence Scoring

```bash
# Find all paths between fragments with confidence scores
quaid graph find-paths jwt-concept-001 jwt-impl-015 --max-paths 5

# Output:
# Paths from jwt-concept-001 to jwt-impl-015:
#
# Path 1 (confidence: 0.92):
# jwt-concept-001 → [implements, 0.95] → jwt-impl-015
#
# Path 2 (confidence: 0.78):
# jwt-concept-001 → [related-to, 0.80] → auth-arch-003 → [implements, 0.85] → jwt-impl-015
#
# Path 3 (confidence: 0.65):
# jwt-concept-001 → [references, 0.70] → security-guide-007 → [related-to, 0.75] → jwt-impl-015

# Show implementation chains with multi-step analysis
quaid graph chain jwt-concept-001 --depth 3

# Output:
# Implementation Chains for JWT Authentication Concept:
#
# Chain 1 (confidence: 0.94):
# jwt-concept-001 → [implements] → jwt-impl-015
#   → Middleware Code: jwt-middleware-023
#   → Unit Tests: jwt-tests-041
#
# Chain 2 (confidence: 0.81):
# jwt-concept-001 → [related-to] → oauth2-concept-002 → [implements] → oauth2-impl-018
```

#### Community Detection

```bash
# Detect knowledge communities using Louvain algorithm
quaid graph communities

# Output:
# Knowledge Communities Detected: 8
#
# Community 1 (24 fragments) - Authentication & Security:
#   Central: jwt-concept-001, oauth2-concept-002
#   Topics: authentication, security, tokens, sessions
#
# Community 2 (18 fragments) - Database Architecture:
#   Central: db-schema-001, migration-003
#   Topics: database, migrations, schemas, queries
#
# Community 3 (15 fragments) - API Development:
#   Central: api-design-005, rest-std-007
#   Topics: API, REST, endpoints, documentation

# Show community for a specific fragment
quaid graph community jwt-impl-015

# Output:
# Fragment jwt-impl-015 belongs to Community 1: Authentication & Security
# Related fragments in same community: 23
# Cross-community connections: 4
```

#### Graph Health Analysis

```bash
# Comprehensive graph health check
quaid graph health

# Output:
# 📊 Knowledge Graph Health Report
#
# 🟢 Connectivity: Strong (1 connected component)
# 🟢 Density: Medium (0.084 - good connectivity without being overwhelming)
# 🟡 Cycles Detected: 3 (review for potential circular dependencies)
#    - Cycle 1: jwt-001 → auth-002 → session-003 → jwt-001
#    - Cycle 2: db-001 → migration-002 → db-001
#    - Cycle 3: api-001 → docs-002 → api-001
#
# 🔴 Critical Bridges: 2 (removal would disconnect graph)
#    - Bridge: jwt-001 ↔ auth-002 (critical dependency)
#    - Bridge: db-001 ↔ schema-002 (architectural link)
#
# 🟢 Isolated Nodes: 0 (all fragments are connected)
# 🟢 Average Path Length: 2.8 (good knowledge accessibility)

# Detect problematic patterns
quaid graph detect-issues

# Output:
# Issues Detected:
# ⚠️  Circular Dependencies: 3
#    - Recommended: Review cycles for potential refactoring opportunities
#
# 🔴 Single Points of Failure: 2
#    - Recommended: Create alternative paths for critical bridges
#
# 🟡 Knowledge Silos: 2 communities with low external connections
#    - Recommended: Add cross-references to improve knowledge flow
```

#### TUI Interactive Graph Navigation

```bash
# Launch TUI with graph focus
quaid tui --graph

# TUI Controls:
# [G]rph mode    - Switch to graph visualization
# [C]enter node  - Set center node for graph view
# [D]epth        - Adjust relationship depth (1-5)
# [M]ode         - Toggle: ascii | table | analytics
# [P]ath         - Find paths between nodes
# [A]nalytics    - Show graph statistics and health
# [E]xport       - Export graph visualization

# Example TUI ASCII Graph Display:
# 🔗 Knowledge Graph Explorer
# ============================================================
#
# 📊 Knowledge Graph: jwt-concept-001
# ============================================================
# 🔷 [CONCEPT] JWT Authentication Strategy
#
# 📍 Distance 1:
#   🔸 [IMPLEMENTATION] JWT Token Validation
#     └─[implements] (0.9)
#   🔸 [DECISION] Authentication Approach
#     └─[references] (0.8)
#
# 📍 Distance 2:
#     🔸 [IMPLEMENTATION] Middleware Implementation
#       └─[implements] (0.7) → [references] (0.8)
#     🔸 [REFERENCE] Security Documentation
#       └─[related-to] (0.6) → [references] (0.8)
```

---

## Version Control Integration

### Git-Native Storage

All Quaid data is stored as text files that can be version controlled:

```
.quaid/
├── config.toml              # Configuration (version controlled)
├── memory/
│   ├── fragments/           # Markdown fragments (version controlled)
│   │   ├── 2025-11-09-jwt-001.md
│   │   └── ...
│   └── indexes/              # Indexes (version controlled)
│       ├── fragments.jsonl
│       └── tantivy/
└── cache/                    # Runtime cache (gitignored)
```

### Git Operations

#### Automatic Git Integration

```bash
# Enable automatic git operations
quaid config set git.auto_commit true

# Automatic operations:
# - Commit fragment changes
# - Commit index updates
# - Create descriptive commit messages
# - Handle merge conflicts
```

#### Manual Git Operations

```bash
# Commit changes manually
quaid git commit --message "Add JWT authentication documentation"

# Show git status
quaid git status

# Sync with remote
quaid git sync

# Create branch for experimental work
quaid git branch feature/experimental-auth
```

### Merge Conflict Resolution

When conflicts occur in Quaid files:

```bash
# Detect conflicts
quaid git detect-conflicts

# Resolve conflicts interactively
quaid git resolve-conflicts

# Auto-resolve conflicts (when possible)
quaid git auto-resolve --strategy keep-both

# Mark conflicts as resolved
quaid git mark-resolved
```

### Backup and Recovery

#### Creating Backups

```bash
# Create backup
quaid backup create

# Create backup with description
quaid backup create --description "Before major refactoring"

# List backups
quaid backup list

# Output:
# Available backups:
# backup-2025-11-09-001.tar.gz  (2 hours ago, 45MB)
# backup-2025-11-08-003.tar.gz  (1 day ago, 42MB)
# backup-2025-11-07-002.tar.gz  (2 days ago, 38MB)
```

#### Restoring Backups

```bash
# Restore from backup
quaid backup restore backup-2025-11-09-001

# Restore specific fragments
quaid backup restore backup-2025-11-09-001 --fragments jwt-001,jwt-002

# Restore configuration only
quaid backup restore backup-2025-11-09-001 --config-only
```

---

## Workflows and Best Practices

### Daily Workflow

```bash
# 1. Start work - restore context
quaid context restore

# 2. Check active goals
quaid context list-goals

# 3. Search for relevant information
quaid recall "authentication implementation"

# 4. Store new insights
quaid store implementation "Added JWT validation to middleware"

# 5. Update goals as needed
quaid context complete-goal "Implement JWT validation"

# 6. Save context at end of day
quaid context save
```

### Team Collaboration

```bash
# 1. Initialize team project
quaid init team-project --team-standards

# 2. Define shared tags and types
quaid config set tags.required "component,status"

# 3. Create shared knowledge
quaid store decision --team "Use JWT for all new services"

# 4. Review and integrate
quaid integrate --from feature/auth --review

# 5. Sync with team
quaid git sync
```

### Project Onboarding

```bash
# 1. Create onboarding guide
quaid store concept --tags onboarding "Project architecture overview"

# 2. Document key decisions
quaid list --type decision --export onboarding-decisions.md

# 3. Share context with new team member
quaid context export --for-user new-member > context-for-new-member.md

# 4. Provide search access
quaid recall "setup instructions" --for-user new-member
```

---

## Next Steps

After mastering core features:

1. **Explore Search Capabilities**: [06-Search-and-Intelligence.md](06-Search-and-Intelligence.md)
2. **Learn CLI Commands**: [07-CLI-and-API-Reference.md](07-CLI-and-API-Reference.md)
3. **Discover Advanced Features**: [08-Advanced-Features.md](08-Advanced-Features.md)

---

**Previous**: [04-Configuration-and-Customization.md](04-Configuration-and-Customization.md) | **Next**: [06-Search-and-Intelligence.md](06-Search-and-Intelligence.md)