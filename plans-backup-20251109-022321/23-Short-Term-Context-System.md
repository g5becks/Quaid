# 23 - Short-Term Context System

**Ephemeral Context Management for Multi-Agent Conversations**

---

## Executive Summary

Quaid needs a **dual-memory system**:

1. **Long-term memory** (fragments) - Permanent knowledge, searchable, structured
2. **Short-term memory** (context) - Ephemeral conversation state, working memory, active goals

**Key Challenge**: Keep conversation context fresh, efficient, and accessible across:
- Multiple agents (different tools/roles)
- New sessions (restore context)
- Long conversations (compact/truncate without losing critical info)
- Context window limits (LLMs have 8k-128k token limits)

**Current Problem**: Updating markdown files works but isn't scalable, token-efficient, or user-friendly.

---

## The Problem with Current Approach

### What You're Doing Now

```markdown
# Current Conversation Context

## Active Goals
- Implement JWT authentication
- Fix token expiration bug

## Key Decisions Made
- Using RS256 instead of HS256
- Token TTL: 1 hour

## Current Focus
Working on token validation in auth/jwt.py

## Next Steps
- Test token refresh flow
- Update documentation
```

### Problems

1. **Not Token-Efficient**
   - Entire markdown file sent every time
   - Redundant formatting overhead
   - No semantic compression

2. **Doesn't Scale**
   - Gets longer over time
   - No automatic pruning/summarization
   - Hard to know what to keep vs discard

3. **Not User-Friendly**
   - Manual updates at "checkpoints"
   - No automatic context tracking
   - Hard to review/debug context state

4. **No Cross-Agent Coordination**
   - Each agent might have different context
   - No shared working memory
   - Race conditions if multiple agents update

---

## Proposed Solution: Structured Short-Term Memory

### Core Concept

**Treat short-term context as a database, not a document**

```
┌─────────────────────────────────────────────────────────────┐
│                   Short-Term Memory Store                    │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Active Context (JSONL - Last 20 messages)        │     │
│  │   - conversation_id                               │     │
│  │   - timestamp                                     │     │
│  │   - agent/role                                    │     │
│  │   - message content                               │     │
│  │   - importance score                              │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Working State (JSON - Current goals/focus)       │     │
│  │   - active_goals                                  │     │
│  │   - current_file_context                          │     │
│  │   - pending_actions                               │     │
│  │   - key_decisions                                 │     │
│  └───────────────────────────────────────────────────┘     │
│                      ↓                                      │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Context Summary (Text - Compressed history)      │     │
│  │   - session_summary (1-2 paragraphs)             │     │
│  │   - key_entities (people, files, concepts)        │     │
│  │   - critical_constraints (rules, requirements)    │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture

### 1. Three-Tier Context System

#### **Tier 1: Hot Context** (Always Included)
- **What**: Current working state
- **Size**: ~500-1000 tokens
- **Lifespan**: Current session only
- **Format**: JSON

```json
{
  "conversation_id": "conv-2025-11-09-001",
  "session_started": "2025-11-09T04:30:00Z",
  "active_goals": [
    {
      "goal": "Implement JWT authentication",
      "status": "in-progress",
      "priority": "high",
      "created": "2025-11-09T04:30:00Z"
    }
  ],
  "current_file_context": {
    "files": ["auth/jwt.py", "auth/validators.py"],
    "last_edited": "auth/jwt.py",
    "current_function": "validate_jwt_token"
  },
  "pending_actions": [
    "Test token refresh flow",
    "Update documentation"
  ],
  "key_decisions": [
    {
      "decision": "Use RS256 instead of HS256",
      "rationale": "Better security for distributed systems",
      "timestamp": "2025-11-09T04:45:00Z"
    }
  ],
  "important_constraints": [
    "Token TTL must be 1 hour",
    "Must support refresh tokens"
  ]
}
```

**Usage**: 
```python
# Load hot context (always included in prompts)
hot_context = context_manager.get_hot_context()

# Update working state
context_manager.update_goal(goal_id, status="complete")
context_manager.add_pending_action("Write tests for JWT validation")
```

---

#### **Tier 2: Recent History** (Sliding Window)
- **What**: Last N messages/events
- **Size**: ~2000-3000 tokens
- **Lifespan**: Last 10-20 interactions
- **Format**: JSONL (Polars-queryable)

```jsonl
{"msg_id": "msg-001", "timestamp": "2025-11-09T04:30:15Z", "agent": "user", "content": "Let's implement JWT auth", "importance": 0.8}
{"msg_id": "msg-002", "timestamp": "2025-11-09T04:30:30Z", "agent": "quaid", "content": "I'll help with that...", "importance": 0.6}
{"msg_id": "msg-003", "timestamp": "2025-11-09T04:31:00Z", "agent": "user", "content": "Use RS256", "importance": 0.9}
```

**Smart Window Management**:
```python
# Keep last 20 messages, but prioritize high-importance
recent_history = context_manager.get_recent_history(
    max_messages=20,
    max_tokens=3000,
    importance_threshold=0.5  # Drop low-importance messages first
)
```

**Importance Scoring** (automatic):
- User messages: 0.7-1.0 (always important)
- Agent responses: 0.4-0.8 (based on content)
- System events: 0.3-0.6
- Boost factors:
  - Contains decision: +0.2
  - References code: +0.1
  - User explicitly said "remember this": +0.3
  - Contains error/warning: +0.15

---

#### **Tier 3: Session Summary** (Compressed)
- **What**: Distilled conversation history
- **Size**: ~500-1000 tokens
- **Lifespan**: Entire session
- **Format**: Structured text

```markdown
# Session Summary (conv-2025-11-09-001)

## Objective
Implementing JWT authentication system with refresh tokens

## Progress
- ✅ Decided on RS256 algorithm
- ✅ Implemented token generation
- 🔄 Currently working on token validation
- ⏳ Pending: token refresh flow

## Key Decisions
1. RS256 over HS256 (security in distributed systems)
2. 1-hour token TTL with refresh tokens
3. Store public keys in Redis

## Critical Context
- Working files: auth/jwt.py, auth/validators.py
- Dependencies: PyJWT, cryptography
- Next: test token refresh + update docs
```

**Auto-Generation** (every 10-15 messages):
```python
# Triggered automatically when history grows
if message_count % 15 == 0:
    summary = context_manager.generate_session_summary()
    # LLM compresses history into structured summary
```

---

### 2. Context Assembly Strategy

**When constructing LLM prompts, use tiered approach**:

```python
def build_context_for_llm(query: str, max_tokens: int = 8000) -> str:
    """
    Intelligently assemble context based on token budget
    """
    budget = max_tokens
    context_parts = []
    
    # TIER 1: Always include hot context (non-negotiable)
    hot = context_manager.get_hot_context()
    context_parts.append(format_hot_context(hot))
    budget -= estimate_tokens(context_parts[-1])
    
    # TIER 2: Recent history (as much as fits)
    if budget > 2000:
        recent = context_manager.get_recent_history(
            max_tokens=min(budget - 1000, 3000),
            importance_threshold=0.5
        )
        context_parts.append(format_recent_history(recent))
        budget -= estimate_tokens(context_parts[-1])
    
    # TIER 3: Session summary (if space remains)
    if budget > 500:
        summary = context_manager.get_session_summary()
        context_parts.append(summary)
        budget -= estimate_tokens(context_parts[-1])
    
    # TIER 4: Relevant long-term memories (if space remains)
    if budget > 1000:
        # Search fragments related to current context
        fragments = search_relevant_fragments(hot, query, max_tokens=budget)
        context_parts.append(format_fragments(fragments))
    
    return "\n\n".join(context_parts)
```

**Token Budget Example** (8k context window):
```
┌─────────────────────────────────────────────────┐
│ LLM Context Window (8000 tokens)                │
│                                                 │
│ ┌─────────────────────────────────────────┐   │
│ │ System Prompt (500 tokens)              │   │
│ └─────────────────────────────────────────┘   │
│ ┌─────────────────────────────────────────┐   │
│ │ Hot Context (800 tokens) [ALWAYS]       │   │
│ └─────────────────────────────────────────┘   │
│ ┌─────────────────────────────────────────┐   │
│ │ Recent History (2500 tokens)            │   │
│ └─────────────────────────────────────────┘   │
│ ┌─────────────────────────────────────────┐   │
│ │ Session Summary (600 tokens)            │   │
│ └─────────────────────────────────────────┘   │
│ ┌─────────────────────────────────────────┐   │
│ │ Relevant Fragments (1500 tokens)        │   │
│ └─────────────────────────────────────────┘   │
│ ┌─────────────────────────────────────────┐   │
│ │ User Query + Response (2100 tokens)     │   │
│ └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

### 3. Context Compaction Strategy

**As conversations grow, compress intelligently**:

```python
class ContextManager:
    def compact_history(self):
        """
        Triggered when history exceeds thresholds
        """
        # 1. Identify messages to compress
        messages = self.get_all_messages()
        
        # 2. Group by semantic similarity (using spaCy)
        clusters = self.cluster_similar_messages(messages)
        
        # 3. Summarize each cluster
        for cluster in clusters:
            if len(cluster) > 3:
                # Replace cluster with summary
                summary = self.llm_summarize(cluster)
                self.replace_messages(cluster, summary)
        
        # 4. Extract key entities/decisions to hot context
        entities = self.extract_entities(messages)
        decisions = self.extract_decisions(messages)
        
        self.update_hot_context({
            "key_entities": entities,
            "key_decisions": decisions
        })
        
        # 5. Archive old messages to long-term storage
        # (Optionally create fragment if conversation was valuable)
        if self.should_archive_as_fragment():
            self.create_fragment_from_conversation()
```

**Compaction Triggers**:
- Every 20 messages: Update session summary
- Every 50 messages: Compact similar message clusters
- Every 100 messages: Archive to long-term + reset short-term
- Context window >80% full: Emergency compaction

---

### 4. Multi-Agent Coordination

**Shared context across agents**:

```python
# Agent A updates context
context_manager.update_goal("Implement JWT auth", status="in-progress", agent="agent-a")
context_manager.add_file_context("auth/jwt.py", current_function="validate_token")

# Agent B reads context
context = context_manager.get_hot_context()
print(context['active_goals'])  # Sees Agent A's update
print(context['current_file_context'])  # Sees what Agent A is working on

# Agent B updates
context_manager.add_decision(
    "Use Redis for public key storage",
    rationale="Fast lookup, built-in TTL",
    agent="agent-b"
)

# Agent A sees Agent B's decision
context = context_manager.get_hot_context()
print(context['key_decisions'])  # Includes Agent B's decision
```

**Locking for Concurrent Updates**:
```python
# Optimistic locking with version numbers
with context_manager.atomic_update() as ctx:
    ctx.add_goal("Write tests")
    ctx.mark_action_complete("Implement JWT validation")
    # Commits atomically or retries if version mismatch
```

---

### 5. Session Restoration

**Restore context when starting new session**:

```bash
# List recent sessions
$ quaid sessions list
conv-2025-11-09-001  "JWT authentication"     (2h ago, 45 messages)
conv-2025-11-08-003  "Fix Redis connection"   (1d ago, 23 messages)

# Restore session
$ quaid sessions restore conv-2025-11-09-001

✓ Restored session: JWT authentication
📊 Active goals: 1
📝 Last activity: Working on token validation
💡 Next steps: Test refresh flow, update docs
```

**Implementation**:
```python
class SessionManager:
    def restore_session(self, session_id: str):
        """
        Restore short-term context from previous session
        """
        # Load session data
        session_data = self.load_session(session_id)
        
        # Restore hot context
        context_manager.set_hot_context(session_data['hot_context'])
        
        # Restore recent history (last 10 messages)
        context_manager.set_recent_history(session_data['recent_history'][-10:])
        
        # Restore session summary
        context_manager.set_session_summary(session_data['summary'])
        
        # Display context to user
        self.display_session_state()
```

---

## Data Storage

### File Structure

```
.quaid/
├── context/
│   ├── sessions/
│   │   ├── conv-2025-11-09-001/
│   │   │   ├── hot_context.json         # Current working state
│   │   │   ├── recent_history.jsonl     # Last 20 messages
│   │   │   ├── summary.md               # Session summary
│   │   │   └── metadata.json            # Session metadata
│   │   └── conv-2025-11-09-002/
│   │       └── ...
│   ├── active_session.txt               # Current session ID
│   └── sessions.jsonl                   # Session index
```

**Session Metadata**:
```json
{
  "session_id": "conv-2025-11-09-001",
  "started": "2025-11-09T04:30:00Z",
  "last_activity": "2025-11-09T06:15:23Z",
  "message_count": 45,
  "title": "JWT Authentication Implementation",
  "status": "active",
  "agents": ["user", "quaid"],
  "primary_files": ["auth/jwt.py", "auth/validators.py"],
  "tags": ["authentication", "jwt", "implementation"]
}
```

---

## Smart Context Features

### 1. Automatic Entity Tracking

**Use spaCy NER to track entities mentioned in conversation**:

```python
# Automatically extract and track
entities = {
    "files": ["auth/jwt.py", "auth/validators.py", "config/settings.py"],
    "functions": ["validate_jwt_token", "generate_token", "refresh_token"],
    "technologies": ["PyJWT", "Redis", "RS256"],
    "people": ["@john", "@sarah"],
    "concepts": ["refresh tokens", "public key infrastructure"]
}

# Update hot context with entities
context_manager.update_entities(entities)
```

**Benefits**:
- LLM knows what files/functions are in scope
- Can reference entities without re-explaining
- Entity-based context retrieval

---

### 2. Goal Tracking

**Explicit goal management**:

```python
# User can explicitly set goals
/set_goal Implement JWT authentication system

# Agent can add sub-goals
context_manager.add_goal(
    "Write unit tests for token validation",
    parent_goal="Implement JWT authentication",
    priority="medium"
)

# Mark goals complete
/complete_goal Implement JWT authentication

# Goals stay in hot context until complete
```

---

### 3. Decision Log

**Track important decisions made during conversation**:

```python
# Automatically detect decision statements
if message.contains_decision_keywords():
    context_manager.add_decision(
        decision=message.extract_decision(),
        rationale=message.extract_rationale(),
        timestamp=now()
    )

# Or user can explicitly log
/log_decision Use RS256 because better for distributed systems
```

**Decisions stay in hot context** and are candidates for fragment creation.

---

### 4. Constraints Tracking

**Remember important constraints/rules for current session**:

```python
# User can set constraints
/add_constraint Token TTL must be exactly 1 hour

# LLM automatically respects constraints in hot context
# Constraints persist until session ends or user removes
```

---

## Context Commands

### User-Facing Slash Commands

```bash
# View current context
/context status

# Update goals
/set_goal <goal description>
/complete_goal <goal_id>

# Add important info to hot context
/remember <important information>

# Log decision
/log_decision <decision> [rationale]

# Add constraint
/add_constraint <constraint>

# View session summary
/context summary

# Clear context (start fresh)
/context clear

# Restore previous session
/context restore <session_id>

# Archive current context as fragment
/context archive [--title="Session Title"]
```

---

## Implementation

### Core Classes

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import polars as pl
from pathlib import Path
import json

@dataclass
class Message:
    msg_id: str
    timestamp: datetime
    agent: str
    content: str
    importance: float
    metadata: Dict[str, Any]

@dataclass
class Goal:
    goal_id: str
    description: str
    status: str  # pending | in-progress | complete
    priority: str  # high | medium | low
    created: datetime
    parent_goal_id: Optional[str] = None

@dataclass
class Decision:
    decision_id: str
    description: str
    rationale: str
    timestamp: datetime
    agent: str

class HotContext:
    """
    Current working state (always included in prompts)
    """
    def __init__(self):
        self.active_goals: List[Goal] = []
        self.pending_actions: List[str] = []
        self.key_decisions: List[Decision] = []
        self.current_file_context: Dict[str, Any] = {}
        self.important_constraints: List[str] = []
        self.key_entities: Dict[str, List[str]] = {}
    
    def to_dict(self) -> dict:
        return {
            "active_goals": [g.__dict__ for g in self.active_goals],
            "pending_actions": self.pending_actions,
            "key_decisions": [d.__dict__ for d in self.key_decisions],
            "current_file_context": self.current_file_context,
            "important_constraints": self.important_constraints,
            "key_entities": self.key_entities
        }
    
    def to_markdown(self) -> str:
        """Format hot context as markdown for LLM"""
        md = "# Current Context\n\n"
        
        if self.active_goals:
            md += "## Active Goals\n"
            for goal in self.active_goals:
                status_icon = {"pending": "⏳", "in-progress": "🔄", "complete": "✅"}
                md += f"- {status_icon[goal.status]} {goal.description}\n"
            md += "\n"
        
        if self.current_file_context:
            md += "## Current Files\n"
            for file in self.current_file_context.get('files', []):
                md += f"- {file}\n"
            md += "\n"
        
        if self.key_decisions:
            md += "## Key Decisions\n"
            for dec in self.key_decisions[-3:]:  # Last 3 decisions
                md += f"- {dec.description}\n"
                md += f"  *Rationale: {dec.rationale}*\n"
            md += "\n"
        
        if self.important_constraints:
            md += "## Important Constraints\n"
            for constraint in self.important_constraints:
                md += f"- {constraint}\n"
            md += "\n"
        
        return md

class ContextManager:
    """
    Manages short-term conversation context
    """
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.hot_context = HotContext()
        self.recent_history: List[Message] = []
        self.session_summary: str = ""
        
        # Load existing context if available
        self.load()
    
    def add_message(self, agent: str, content: str, importance: float = 0.5):
        """Add message to recent history"""
        msg = Message(
            msg_id=f"msg-{len(self.recent_history):04d}",
            timestamp=datetime.now(),
            agent=agent,
            content=content,
            importance=importance,
            metadata={}
        )
        self.recent_history.append(msg)
        
        # Auto-compact if needed
        if len(self.recent_history) > 20:
            self._compact_history()
        
        # Auto-generate summary every 15 messages
        if len(self.recent_history) % 15 == 0:
            self._update_session_summary()
        
        self.save()
    
    def get_hot_context(self) -> HotContext:
        """Get current hot context"""
        return self.hot_context
    
    def update_goal(self, goal_id: str, **kwargs):
        """Update existing goal"""
        for goal in self.hot_context.active_goals:
            if goal.goal_id == goal_id:
                for key, value in kwargs.items():
                    setattr(goal, key, value)
                break
        self.save()
    
    def add_goal(self, description: str, priority: str = "medium", parent_goal_id: Optional[str] = None):
        """Add new goal to hot context"""
        goal = Goal(
            goal_id=f"goal-{len(self.hot_context.active_goals):03d}",
            description=description,
            status="pending",
            priority=priority,
            created=datetime.now(),
            parent_goal_id=parent_goal_id
        )
        self.hot_context.active_goals.append(goal)
        self.save()
    
    def add_decision(self, description: str, rationale: str, agent: str = "quaid"):
        """Log important decision"""
        decision = Decision(
            decision_id=f"dec-{len(self.hot_context.key_decisions):03d}",
            description=description,
            rationale=rationale,
            timestamp=datetime.now(),
            agent=agent
        )
        self.hot_context.key_decisions.append(decision)
        self.save()
    
    def get_recent_history(self, max_messages: int = 20, max_tokens: int = 3000, importance_threshold: float = 0.5) -> List[Message]:
        """Get recent messages with smart filtering"""
        # Filter by importance
        filtered = [m for m in self.recent_history if m.importance >= importance_threshold]
        
        # Sort by timestamp (most recent first)
        filtered.sort(key=lambda m: m.timestamp, reverse=True)
        
        # Take top N
        selected = filtered[:max_messages]
        
        # TODO: Token-based filtering (estimate tokens, drop if over budget)
        
        return selected
    
    def build_context_for_llm(self, query: str, max_tokens: int = 8000) -> str:
        """Assemble context for LLM prompt"""
        context_parts = []
        budget = max_tokens
        
        # System prompt (reserved)
        budget -= 500
        
        # Hot context (always included)
        hot_md = self.hot_context.to_markdown()
        context_parts.append(hot_md)
        budget -= len(hot_md) // 4  # Rough token estimate
        
        # Recent history
        if budget > 2000:
            recent = self.get_recent_history(max_tokens=min(budget - 1000, 3000))
            history_md = self._format_history(recent)
            context_parts.append(history_md)
            budget -= len(history_md) // 4
        
        # Session summary
        if budget > 500 and self.session_summary:
            context_parts.append(f"## Session Summary\n\n{self.session_summary}")
            budget -= len(self.session_summary) // 4
        
        return "\n\n".join(context_parts)
    
    def _compact_history(self):
        """Compress old messages"""
        # Keep last 10 messages
        self.recent_history = self.recent_history[-10:]
    
    def _update_session_summary(self):
        """Generate session summary using LLM"""
        # TODO: Call LLM to summarize conversation
        pass
    
    def _format_history(self, messages: List[Message]) -> str:
        """Format message history as markdown"""
        md = "## Recent Conversation\n\n"
        for msg in reversed(messages):  # Chronological order
            md += f"**{msg.agent}**: {msg.content}\n\n"
        return md
    
    def save(self):
        """Persist context to disk"""
        # Save hot context
        with open(self.session_dir / "hot_context.json", "w") as f:
            json.dump(self.hot_context.to_dict(), f, indent=2, default=str)
        
        # Save recent history
        df = pl.DataFrame([
            {
                "msg_id": m.msg_id,
                "timestamp": m.timestamp.isoformat(),
                "agent": m.agent,
                "content": m.content,
                "importance": m.importance
            }
            for m in self.recent_history
        ])
        df.write_ndjson(self.session_dir / "recent_history.jsonl")
        
        # Save summary
        if self.session_summary:
            with open(self.session_dir / "summary.md", "w") as f:
                f.write(self.session_summary)
    
    def load(self):
        """Load context from disk"""
        hot_path = self.session_dir / "hot_context.json"
        if hot_path.exists():
            with open(hot_path) as f:
                data = json.load(f)
                # TODO: Deserialize into HotContext
        
        history_path = self.session_dir / "recent_history.jsonl"
        if history_path.exists():
            df = pl.read_ndjson(history_path)
            # TODO: Deserialize into Message objects
        
        summary_path = self.session_dir / "summary.md"
        if summary_path.exists():
            with open(summary_path) as f:
                self.session_summary = f.read()
```

---

## Benefits of This Approach

✅ **Token-Efficient**
- Structured data (JSON/JSONL) > verbose markdown
- Tiered approach (only include what fits)
- Smart importance-based filtering

✅ **Scalable**
- Auto-compaction prevents unbounded growth
- Session summaries compress history
- Archive old sessions to long-term memory

✅ **User-Friendly**
- Automatic context tracking (goals, decisions, entities)
- Slash commands for manual control
- Session restore for continuity

✅ **Multi-Agent Friendly**
- Shared context store
- Atomic updates (no race conditions)
- Agent attribution (know who did what)

✅ **Context-Window Aware**
- Token budget management
- Priority-based inclusion
- Emergency compaction when needed

---

## Comparison to Current Approach

| **Current (Markdown File)** | **Proposed (Structured Context)** |
|------------------------------|-----------------------------------|
| Manual updates at checkpoints | Automatic tracking |
| Entire file sent every time | Tiered, token-budgeted assembly |
| Grows unbounded | Auto-compaction + summarization |
| Hard to query/filter | Polars-queryable JSONL |
| Single file, race conditions | Atomic updates, multi-agent safe |
| No session restore | Full session restoration |
| ~2000-5000 tokens | ~1000-3000 tokens (40% reduction) |

---

## Next Steps

1. **Implement `ContextManager` class** with hot context, recent history, summaries
2. **Add slash commands** for context management (`/set_goal`, `/remember`, etc.)
3. **Integrate with LLM prompt builder** (tiered context assembly)
4. **Add auto-compaction** (every N messages)
5. **Implement session management** (create, restore, archive)
6. **Build context visualization** (show current context state)

---

## Open Questions

1. **Session summarization**: Use same LLM or lighter model?
2. **Importance scoring**: Train custom model or use heuristics?
3. **Fragment creation**: Auto-create fragments from sessions? Criteria?
4. **Context persistence**: How long to keep old sessions?
5. **Multi-user**: How to handle multiple users sharing context?

---

**Previous**: [22-Slash-Commands-Design.md](22-Slash-Commands-Design.md)

**Version**: 1.0  
**Last Updated**: 2025-11-09
