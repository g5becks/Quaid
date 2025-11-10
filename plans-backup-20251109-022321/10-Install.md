# 10 - Install

**Installation, Setup, and Deployment**

---

## 10. Installation & Setup

### Current Approach Issues

- Complex bundling strategy in brainstorming doc
- Unclear installation process
- Manual tool management

### Refined Installation Strategy

#### Installation via Package Manager

```bash
# Via Cargo (Rust)
cargo install quaid

# Via NPM (if Node wrapper preferred)
npm install -g quaid

# Via Homebrew (macOS)
brew install quaid
```

#### First-Run Auto-Setup

On first `quaid init`:

```bash
quaid init
  ↓
Checking dependencies...
  ✓ Git found
  ✗ NuShell not found - downloading...
  ✗ mq not found - downloading...
  ✗ aichat not found - downloading...
  ↓
Downloading tools to ~/.quaid/tools/...
  ✓ nushell v0.97.0 installed
  ✓ mq v0.3.4 installed
  ✓ aichat v0.16.0 installed
  ↓
Creating configuration...
  ✓ ~/.quaid/config.toml created
  ↓
Initializing project...
  ✓ .quaid/ directory created
  ✓ Memory structure initialized
  ↓
Generating slash commands...
  Detected AI tools: Cursor, Claude
  ✓ Generated .cursor/commands/quaid-*.md
  ✓ Generated .claude/commands/quaid/*.md
  ↓
Setup complete! 🚀

Next steps:
  1. Configure AI provider: quaid config set ai.provider openai
  2. Set API key: export OPENAI_API_KEY=sk-...
  3. Store your first memory: quaid store --content "Hello, Quaid!"
  4. Restart your AI tool to load slash commands
```

#### Tool Management

```bash
# Check installed tools
quaid tools list
  ╭──────────┬──────────┬──────────────────────────╮
  │   tool   │ version  │         path             │
  ├──────────┼──────────┼──────────────────────────┤
  │ nushell  │ 0.97.0   │ ~/.quaid/tools/nushell/..│
  │ mq       │ 0.3.4    │ ~/.quaid/tools/mq/bin/mq │
  │ aichat   │ 0.16.0   │ ~/.quaid/tools/aichat/.. │
  ╰──────────┴──────────┴──────────────────────────╯

# Update tools
quaid tools update
quaid tools update nushell

# Verify installation
quaid doctor
  ✓ Configuration valid
  ✓ All tools installed
  ✓ Git repository detected
  ✗ AIChat API key not set
  
  Suggestions:
    - Set API key: export OPENAI_API_KEY=sk-...
```

#### Project Initialization

```bash
# Initialize in existing project
cd my-project
quaid init

# Initialize with RAG
quaid init --rag

# Initialize with specific AI provider
quaid init --provider anthropic

# Initialize with custom config
quaid init --config custom-config.toml
```

### Upgrade Path

```bash
# Upgrade quaid
cargo install quaid --force

# Upgrade will preserve:
# - ~/.quaid/config.toml
# - Project .quaid/memory/ directories
# - Tool versions (unless --upgrade-tools flag)

# Upgrade with tools
quaid upgrade --tools
```

---


---

**Previous**: [09-Config.md](09-Config.md) | **Next**: [11-Advanced.md](11-Advanced.md)
