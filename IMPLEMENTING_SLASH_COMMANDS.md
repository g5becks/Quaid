# Implementing the OpenSpec Slash Command Pattern

A comprehensive guide to adding AI slash command integration to your CLI tool without MCP.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [File Format](#file-format)
- [Implementation Guide](#implementation-guide)
- [Tool-Specific Configurations](#tool-specific-configurations)
- [Advanced Patterns](#advanced-patterns)
- [Best Practices](#best-practices)
- [Testing](#testing)

---

## Overview

The OpenSpec pattern enables CLI tools to integrate with AI coding assistants (Cursor, Claude Code, Windsurf, etc.) by writing markdown instruction files to tool-specific directories. This approach:

- ✅ **Works without MCP** - No server process, no protocol overhead
- ✅ **Supports 15+ AI tools** - Single implementation works everywhere
- ✅ **Zero runtime cost** - Instructions read once at startup
- ✅ **Git-trackable** - Instructions are just markdown files
- ✅ **User-customizable** - Developers can modify instructions
- ✅ **Updateable** - Managed content blocks allow non-destructive updates

### When to Use This Pattern

Use this pattern if your tool:
- Has a CLI interface
- Orchestrates multi-step workflows
- Creates/modifies files
- Benefits from AI guidance during use
- Wants to work with multiple AI tools

**Don't use this if:**
- You only need real-time data queries (use MCP)
- No workflows to guide (just simple data access)
- Your tool has no CLI

---

## How It Works

### The Core Concept

1. **AI tools scan specific directories** for custom commands at startup
2. **Your CLI generates markdown files** in those directories during setup
3. **AI tools expose these as slash commands** to users
4. **Users trigger workflows** by typing `/yourtool-command`
5. **AI follows instructions** in the markdown file to execute the workflow

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│ User types: /mytool-deploy                              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ AI Tool (Cursor/Claude/etc.) reads:                     │
│ .cursor/commands/mytool-deploy.md                       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ AI executes instructions:                               │
│ 1. Run `mytool validate`                                │
│ 2. Review output                                        │
│ 3. Run `mytool deploy --env production`                 │
│ 4. Monitor with `mytool logs`                           │
└─────────────────────────────────────────────────────────┘
```

**Key Insight:** You're not creating tools for the AI - you're giving it a workflow playbook.

---

## Quick Start

Here's a minimal example to get slash commands working in 5 minutes.

### 1. Create Command Files

```bash
# For Cursor
mkdir -p .cursor/commands
cat > .cursor/commands/mytool-deploy.md << 'EOF'
---
name: /mytool-deploy
id: mytool-deploy
category: MyTool
description: Deploy application with validation
---

**Steps**
1. Run `mytool validate --strict`
2. Review validation output
3. Run `mytool deploy --env production`
4. Monitor with `mytool logs --follow`
EOF
```

### 2. Test It

1. Restart your AI tool (commands load at startup)
2. Type `/mytool-deploy` in the chat
3. Watch the AI execute the workflow

That's it! You now have a working slash command.

---

## Directory Structure

Different AI tools expect files in different locations. Here's the complete map:

```
project-root/
├── .cursor/commands/           # Cursor
│   ├── mytool-deploy.md
│   ├── mytool-init.md
│   └── mytool-rollback.md
│
├── .claude/commands/mytool/    # Claude Code
│   ├── deploy.md
│   ├── init.md
│   └── rollback.md
│
├── .windsurf/workflows/        # Windsurf
│   ├── mytool-deploy.md
│   ├── mytool-init.md
│   └── mytool-rollback.md
│
├── .github/prompts/            # GitHub Copilot
│   ├── mytool-deploy.md
│   ├── mytool-init.md
│   └── mytool-rollback.md
│
├── .amazonq/prompts/           # Amazon Q Developer
│   ├── mytool-deploy.md
│   ├── mytool-init.md
│   └── mytool-rollback.md
│
├── .opencode/command/          # OpenCode
│   ├── mytool-deploy.md
│   ├── mytool-init.md
│   └── mytool-rollback.md
│
├── .kilocode/workflows/        # Kilo Code
│   ├── mytool-deploy.md
│   ├── mytool-init.md
│   └── mytool-rollback.md
│
├── .codebuddy/commands/        # CodeBuddy Code
│   ├── mytool-deploy.md
│   ├── mytool-init.md
│   └── mytool-rollback.md
│
├── .qoder/commands/mytool/     # Qoder
│   ├── deploy.md
│   ├── init.md
│   └── rollback.md
│
├── .clinerules/                # Cline (rules, not commands)
│   ├── mytool-deploy.md
│   ├── mytool-init.md
│   └── mytool-rollback.md
│
└── AGENTS.md                   # Fallback for any AGENTS.md-compatible tool
```

### Tool Detection Strategy

For maximum compatibility, generate files for all tools during `init`:

```bash
mytool init
# Detects which AI tools user might have and creates all relevant files
```

---

## File Format

Each AI tool expects slightly different frontmatter, but the body is always the same.

### Anatomy of a Command File

```markdown
---
# YAML frontmatter (tool-specific metadata)
name: /mytool-deploy
id: mytool-deploy
category: MyTool
description: Deploy with validation and rollback safety
---

<!-- MYTOOL:START -->
**Guardrails**
- Always validate before deploying
- Confirm environment before proceeding
- Never skip health checks

**Steps**
1. Run `mytool validate --strict` to check readiness
2. Review validation output and resolve any issues
3. Ask user to confirm target environment
4. Run `mytool deploy <env> --wait`
5. Monitor with `mytool health-check <env>`
6. If issues arise, run `mytool rollback <env>`

**Reference**
- View deployments: `mytool list --env <env>`
- Check logs: `mytool logs <env> --tail 100`
- Rollback: `mytool rollback <env> --to <version>`
<!-- MYTOOL:END -->
```

### Key Components

1. **Frontmatter** - Tool-specific YAML metadata
2. **Markers** - `<!-- MYTOOL:START -->` and `<!-- MYTOOL:END -->` for managed updates
3. **Guardrails** - Rules the AI must follow
4. **Steps** - Sequential instructions for the workflow
5. **Reference** - Quick command reference

---

## Implementation Guide

Here's how to build `mytool init` to generate slash command files.

### Step 1: Define Your Commands

First, identify the key workflows in your tool:

```typescript
// commands.ts
export type CommandId = 'init' | 'deploy' | 'rollback' | 'validate';

export const COMMANDS: Record<CommandId, {
  description: string;
  content: string;
}> = {
  init: {
    description: 'Initialize a new project with best practices',
    content: `**Guardrails**
- Ask about project type before scaffolding
- Don't overwrite existing configuration

**Steps**
1. Run \`mytool init\` and answer prompts
2. Review generated configuration in \`mytool.config.json\`
3. Run \`mytool validate\` to ensure setup is correct
4. Commit configuration to version control

**Reference**
- View config: \`cat mytool.config.json\`
- Reconfigure: \`mytool init --reconfigure\``
  },
  
  deploy: {
    description: 'Deploy application with validation and rollback safety',
    content: `**Guardrails**
- Always validate before deploying
- Confirm environment before proceeding
- Monitor health checks after deployment

**Steps**
1. Run \`mytool validate --strict\` to check deployment readiness
2. Review validation output and resolve any issues
3. Ask user to confirm target environment
4. Run \`mytool deploy <env> --wait\`
5. Monitor with \`mytool health-check <env>\`
6. If health checks fail, run \`mytool rollback <env>\`

**Reference**
- View active deployments: \`mytool list --env <env>\`
- Check logs: \`mytool logs <env> --tail 100\`
- Manual rollback: \`mytool rollback <env> --to <version>\``
  },
  
  rollback: {
    description: 'Safely rollback a deployment',
    content: `**Guardrails**
- Confirm which deployment to rollback
- Never rollback without checking current state

**Steps**
1. Run \`mytool list --env <env>\` to see deployment history
2. Identify the target version to rollback to
3. Run \`mytool rollback <env> --to <version> --dry-run\` to preview
4. Confirm with user before proceeding
5. Run \`mytool rollback <env> --to <version>\`
6. Verify with \`mytool health-check <env>\`

**Reference**
- View history: \`mytool list --env <env> --history\`
- Check status: \`mytool status <env>\``
  },
  
  validate: {
    description: 'Validate project configuration and deployment readiness',
    content: `**Steps**
1. Run \`mytool validate --strict\`
2. Review all validation errors and warnings
3. For each error, run suggested fix command
4. Re-run \`mytool validate --strict\` until all checks pass
5. Document any warnings that can't be resolved

**Reference**
- Verbose output: \`mytool validate --strict --verbose\`
- Fix specific check: \`mytool validate --only <check-name>\``
  }
};
```

### Step 2: Create Tool Configurators

Build a registry of tool-specific file generators:

```typescript
// configurators.ts
import * as fs from 'fs/promises';
import * as path from 'path';

export interface ToolConfigurator {
  toolId: string;
  toolName: string;
  isAvailable: boolean;
  generate(projectPath: string, commands: CommandId[]): Promise<string[]>;
}

export class CursorConfigurator implements ToolConfigurator {
  toolId = 'cursor';
  toolName = 'Cursor';
  isAvailable = true;

  async generate(projectPath: string, commands: CommandId[]): Promise<string[]> {
    const createdFiles: string[] = [];
    const baseDir = path.join(projectPath, '.cursor/commands');
    
    await fs.mkdir(baseDir, { recursive: true });
    
    for (const cmd of commands) {
      const fileName = `mytool-${cmd}.md`;
      const filePath = path.join(baseDir, fileName);
      const content = this.generateFile(cmd);
      
      await fs.writeFile(filePath, content, 'utf-8');
      createdFiles.push(`.cursor/commands/${fileName}`);
    }
    
    return createdFiles;
  }
  
  private generateFile(cmd: CommandId): string {
    const { description, content } = COMMANDS[cmd];
    
    return `---
name: /mytool-${cmd}
id: mytool-${cmd}
category: MyTool
description: ${description}
---

<!-- MYTOOL:START -->
${content}
<!-- MYTOOL:END -->
`;
  }
}

export class ClaudeConfigurator implements ToolConfigurator {
  toolId = 'claude';
  toolName = 'Claude Code';
  isAvailable = true;

  async generate(projectPath: string, commands: CommandId[]): Promise<string[]> {
    const createdFiles: string[] = [];
    const baseDir = path.join(projectPath, '.claude/commands/mytool');
    
    await fs.mkdir(baseDir, { recursive: true });
    
    for (const cmd of commands) {
      const fileName = `${cmd}.md`;
      const filePath = path.join(baseDir, fileName);
      const content = this.generateFile(cmd);
      
      await fs.writeFile(filePath, content, 'utf-8');
      createdFiles.push(`.claude/commands/mytool/${fileName}`);
    }
    
    return createdFiles;
  }
  
  private generateFile(cmd: CommandId): string {
    const { description, content } = COMMANDS[cmd];
    
    return `---
name: MyTool: ${cmd.charAt(0).toUpperCase() + cmd.slice(1)}
description: ${description}
category: MyTool
tags: [mytool, ${cmd}]
---

<!-- MYTOOL:START -->
${content}
<!-- MYTOOL:END -->
`;
  }
}

// Add more configurators for other tools...
export class WindsurfConfigurator implements ToolConfigurator {
  toolId = 'windsurf';
  toolName = 'Windsurf';
  isAvailable = true;

  async generate(projectPath: string, commands: CommandId[]): Promise<string[]> {
    const createdFiles: string[] = [];
    const baseDir = path.join(projectPath, '.windsurf/workflows');
    
    await fs.mkdir(baseDir, { recursive: true });
    
    for (const cmd of commands) {
      const fileName = `mytool-${cmd}.md`;
      const filePath = path.join(baseDir, fileName);
      const content = this.generateFile(cmd);
      
      await fs.writeFile(filePath, content, 'utf-8');
      createdFiles.push(`.windsurf/workflows/${fileName}`);
    }
    
    return createdFiles;
  }
  
  private generateFile(cmd: CommandId): string {
    const { description, content } = COMMANDS[cmd];
    
    return `---
name: /mytool-${cmd}
description: ${description}
---

<!-- MYTOOL:START -->
${content}
<!-- MYTOOL:END -->
`;
  }
}

// Tool registry
export const CONFIGURATORS: ToolConfigurator[] = [
  new CursorConfigurator(),
  new ClaudeConfigurator(),
  new WindsurfConfigurator(),
  // Add more as needed
];
```

### Step 3: Build the Init Command

Create a command that generates all the files:

```typescript
// init.ts
import { CONFIGURATORS } from './configurators.js';
import { COMMANDS } from './commands.js';

export async function initCommand(projectPath: string) {
  console.log('🚀 Initializing MyTool slash commands...\n');
  
  const allCommands = Object.keys(COMMANDS) as CommandId[];
  const results: Record<string, string[]> = {};
  
  for (const configurator of CONFIGURATORS) {
    console.log(`📝 Setting up ${configurator.toolName}...`);
    
    try {
      const files = await configurator.generate(projectPath, allCommands);
      results[configurator.toolId] = files;
      
      console.log(`   ✅ Created ${files.length} command files`);
      files.forEach(f => console.log(`      - ${f}`));
    } catch (error) {
      console.error(`   ❌ Failed: ${error.message}`);
    }
  }
  
  console.log('\n✨ Slash command setup complete!\n');
  console.log('📌 Next steps:');
  console.log('   1. Restart your AI coding tool');
  console.log('   2. Type /mytool-init to get started');
  console.log('   3. Explore /mytool-deploy, /mytool-validate, etc.\n');
  
  return results;
}
```

### Step 4: Add Update Support

Support updating existing files without losing user customizations:

```typescript
// update.ts
import * as fs from 'fs/promises';
import * as path from 'path';

const MARKERS = {
  start: '<!-- MYTOOL:START -->',
  end: '<!-- MYTOOL:END -->'
};

export async function updateCommand(projectPath: string) {
  console.log('🔄 Updating MyTool slash commands...\n');
  
  const updated: string[] = [];
  
  for (const configurator of CONFIGURATORS) {
    console.log(`📝 Updating ${configurator.toolName}...`);
    
    const files = await findExistingFiles(projectPath, configurator.toolId);
    
    for (const filePath of files) {
      try {
        await updateFile(filePath);
        updated.push(filePath);
        console.log(`   ✅ Updated ${filePath}`);
      } catch (error) {
        console.error(`   ❌ Failed to update ${filePath}: ${error.message}`);
      }
    }
  }
  
  console.log(`\n✨ Updated ${updated.length} command files\n`);
}

async function updateFile(filePath: string) {
  const content = await fs.readFile(filePath, 'utf-8');
  
  const startIndex = content.indexOf(MARKERS.start);
  const endIndex = content.indexOf(MARKERS.end);
  
  if (startIndex === -1 || endIndex === -1) {
    throw new Error('Missing MYTOOL markers');
  }
  
  // Extract command ID from file path
  const cmdId = extractCommandId(filePath);
  const newContent = COMMANDS[cmdId].content;
  
  // Preserve frontmatter and user content outside markers
  const before = content.slice(0, startIndex + MARKERS.start.length);
  const after = content.slice(endIndex);
  const updated = `${before}\n${newContent}\n${after}`;
  
  await fs.writeFile(filePath, updated, 'utf-8');
}

function extractCommandId(filePath: string): CommandId {
  // Extract from file name, e.g., "mytool-deploy.md" -> "deploy"
  const fileName = path.basename(filePath, '.md');
  return fileName.replace('mytool-', '') as CommandId;
}

async function findExistingFiles(
  projectPath: string,
  toolId: string
): Promise<string[]> {
  // Tool-specific logic to find existing command files
  const patterns: Record<string, string> = {
    cursor: '.cursor/commands/mytool-*.md',
    claude: '.claude/commands/mytool/*.md',
    windsurf: '.windsurf/workflows/mytool-*.md',
    // etc.
  };
  
  // Use glob or manual directory scanning
  // Simplified example:
  const pattern = patterns[toolId];
  // Return matched files...
  return [];
}
```

---

## Tool-Specific Configurations

### Cursor

```typescript
// Path: .cursor/commands/mytool-{command}.md
const cursorTemplate = `---
name: /mytool-${cmd}
id: mytool-${cmd}
category: MyTool
description: ${description}
---

<!-- MYTOOL:START -->
${content}
<!-- MYTOOL:END -->
`;
```

**Invocation:** `/mytool-deploy`

### Claude Code

```typescript
// Path: .claude/commands/mytool/{command}.md
const claudeTemplate = `---
name: MyTool: ${capitalize(cmd)}
description: ${description}
category: MyTool
tags: [mytool, ${cmd}]
---

<!-- MYTOOL:START -->
${content}
<!-- MYTOOL:END -->
`;
```

**Invocation:** `/openspec:deploy` (uses colon separator)

### Windsurf

```typescript
// Path: .windsurf/workflows/mytool-{command}.md
const windsurfTemplate = `---
name: /mytool-${cmd}
description: ${description}
---

<!-- MYTOOL:START -->
${content}
<!-- MYTOOL:END -->
`;
```

**Invocation:** `/mytool-deploy`

### GitHub Copilot

```typescript
// Path: .github/prompts/mytool-{command}.md
const copilotTemplate = `---
title: mytool-${cmd}
description: ${description}
---

<!-- MYTOOL:START -->
${content}
<!-- MYTOOL:END -->
`;
```

**Invocation:** `/mytool-deploy`

### Amazon Q Developer

```typescript
// Path: .amazonq/prompts/mytool-{command}.md
const amazonQTemplate = `---
title: mytool-${cmd}
description: ${description}
---

<!-- MYTOOL:START -->
${content}
<!-- MYTOOL:END -->
`;
```

**Invocation:** `@mytool-deploy` (uses @ prefix)

### Cline (Special Case)

Cline uses "rules" instead of commands:

```typescript
// Path: .clinerules/mytool-{command}.md
const clineTemplate = `# MyTool: ${capitalize(cmd)}

${description}

<!-- MYTOOL:START -->
${content}
<!-- MYTOOL:END -->
`;
```

**Usage:** Cline reads these as context rules, not direct slash commands

### AGENTS.md Fallback

For tools that support AGENTS.md convention:

```markdown
<!-- MYTOOL:START -->
# MyTool Instructions

When the user mentions MyTool workflows or types requests related to deployment, 
follow these patterns:

## Deployment Workflow
${content for deploy}

## Initialization Workflow
${content for init}

<!-- MYTOOL:END -->
```

---

## Advanced Patterns

### 1. Dynamic Arguments

Support passing arguments to commands:

```markdown
---
name: /mytool-deploy
---

**Steps**
1. Determine the target environment:
   - If `<Environment>` block is present, use that value
   - Otherwise, ask the user which environment to deploy to
2. Run `mytool deploy <env> --wait`
3. Monitor with `mytool health-check <env>`

**Supported Arguments**
- Environment name (production, staging, development)
```

When user types `/mytool-deploy production`, some tools will inject:
```markdown
<Environment>production</Environment>
```

### 2. Conditional Workflows

```markdown
**Steps**
1. Run `mytool status` to check current state
2. If status shows "dirty":
   - Run `mytool clean`
   - Re-run `mytool status`
3. If status shows "ready":
   - Proceed with `mytool deploy`
4. Otherwise:
   - Stop and ask user to resolve issues
```

### 3. Error Handling

```markdown
**Steps**
1. Run `mytool validate --strict`
2. If validation fails:
   - Read the error output carefully
   - For each error, check if there's a suggested fix command
   - Run the fix command
   - Re-run `mytool validate --strict`
   - Repeat until validation passes
3. If validation passes:
   - Proceed to deployment
```

### 4. Interactive Confirmation

```markdown
**Steps**
1. Run `mytool deploy --dry-run production` to preview changes
2. Show the user what will change
3. Ask: "Do you want to proceed with deployment to production?"
4. Wait for explicit confirmation (yes/no)
5. Only if confirmed, run `mytool deploy production`
```

### 5. Multi-Tool Workflows

```markdown
**Steps**
1. Run `git status` to check for uncommitted changes
2. If there are changes:
   - Ask user if they should be committed first
   - If yes, run `git add .` and `git commit -m "Pre-deployment commit"`
3. Run `mytool validate --strict`
4. Run `mytool deploy production`
5. After deployment, run `git tag v$(mytool version)`
```

### 6. Skills / Reusable Functions

Create helper functions the AI can reference:

```markdown
**Reference: Common Tasks**

To check deployment readiness:
\`\`\`bash
mytool validate --strict && \
mytool test --suite=smoke && \
git diff --quiet
\`\`\`

To safely rollback:
\`\`\`bash
mytool rollback <env> --to $(mytool list --env <env> --format=json | jq -r '.[1].version')
\`\`\`
```

---

## Best Practices

### 1. Clear Guardrails

Always include guardrails for destructive operations:

```markdown
**Guardrails**
- Never deploy without validation
- Always confirm production deployments
- Check for uncommitted changes before deploying
- Monitor health checks after deployment
```

### 2. Progressive Disclosure

Start with simple commands, add advanced ones later:

**Phase 1:** `init`, `validate`
**Phase 2:** `deploy`, `rollback`
**Phase 3:** `debug`, `analyze`, `optimize`

### 3. Consistent Naming

Use consistent command naming:
- `mytool-init` not `mytool-setup`
- `mytool-deploy` not `mytool-ship`
- `mytool-rollback` not `mytool-revert`

### 4. Rich Reference Section

Provide quick command references:

```markdown
**Reference**
- List deployments: `mytool list --env <env>`
- View logs: `mytool logs <env> --tail 100 --follow`
- Check status: `mytool status <env>`
- Rollback: `mytool rollback <env> --to <version>`
- Health check: `mytool health-check <env>`
```

### 5. Version Compatibility

Include version checks:

```markdown
**Steps**
1. Run `mytool --version` to check version
2. If version is < 2.0.0:
   - Tell user to upgrade: `npm install -g mytool@latest`
   - Stop execution
3. Proceed with workflow
```

### 6. Use Managed Markers

Always use markers for updateable content:

```markdown
---
name: /mytool-deploy
---

<!-- MYTOOL:START -->
<!-- This content can be updated by `mytool update` -->
**Steps**
...
<!-- MYTOOL:END -->

<!-- Users can add custom notes here without losing them on update -->
```

### 7. Test Your Instructions

Actually use the AI to test your workflows:

```bash
# Generate commands
mytool init

# Restart AI tool

# Test each command
/mytool-init
/mytool-validate
/mytool-deploy staging
/mytool-rollback staging
```

### 8. Document Edge Cases

```markdown
**Steps**
1. Run `mytool deploy <env>`
2. If you see error "Port 8080 already in use":
   - Run `mytool stop <env>` first
   - Then retry deployment
3. If deployment hangs for > 5 minutes:
   - Check logs with `mytool logs <env> --tail 100`
   - Look for startup errors
```

---

## Testing

### Manual Testing Checklist

```markdown
## Slash Command Testing

### Setup
- [ ] Run `mytool init` in a test project
- [ ] Verify files created in `.cursor/commands/`
- [ ] Verify files created in `.claude/commands/mytool/`
- [ ] Restart AI tool

### Command Discovery
- [ ] Type `/` and verify mytool commands appear
- [ ] Verify command descriptions are clear
- [ ] Verify categorization (if supported)

### Execution Testing
- [ ] `/mytool-init` - creates valid configuration
- [ ] `/mytool-validate` - catches actual errors
- [ ] `/mytool-deploy staging` - deploys successfully
- [ ] `/mytool-rollback staging` - rolls back successfully

### Error Handling
- [ ] Trigger validation error, verify AI follows fix steps
- [ ] Cancel deployment mid-way, verify AI doesn't proceed
- [ ] Invalid environment, verify AI catches it

### Update Testing
- [ ] Modify a command file outside markers
- [ ] Run `mytool update`
- [ ] Verify markers updated, custom content preserved
```

### Automated Testing

```typescript
// test/slash-commands.test.ts
import { describe, it, expect } from 'vitest';
import { initCommand } from '../src/init';
import * as fs from 'fs/promises';
import * as path from 'path';

describe('Slash Command Generation', () => {
  it('should create Cursor command files', async () => {
    const testDir = './test-project';
    await initCommand(testDir);
    
    const deployFile = path.join(testDir, '.cursor/commands/mytool-deploy.md');
    const content = await fs.readFile(deployFile, 'utf-8');
    
    expect(content).toContain('name: /mytool-deploy');
    expect(content).toContain('<!-- MYTOOL:START -->');
    expect(content).toContain('<!-- MYTOOL:END -->');
    expect(content).toContain('**Steps**');
  });
  
  it('should create Claude command files', async () => {
    const testDir = './test-project';
    await initCommand(testDir);
    
    const deployFile = path.join(testDir, '.claude/commands/mytool/deploy.md');
    const content = await fs.readFile(deployFile, 'utf-8');
    
    expect(content).toContain('name: MyTool: Deploy');
    expect(content).toContain('category: MyTool');
  });
  
  it('should preserve custom content on update', async () => {
    const testDir = './test-project';
    await initCommand(testDir);
    
    const filePath = path.join(testDir, '.cursor/commands/mytool-deploy.md');
    let content = await fs.readFile(filePath, 'utf-8');
    
    // Add custom content outside markers
    content += '\n\n## My Custom Notes\nDon\'t forget to notify the team!';
    await fs.writeFile(filePath, content);
    
    // Update should preserve custom content
    await updateCommand(testDir);
    const updated = await fs.readFile(filePath, 'utf-8');
    
    expect(updated).toContain('## My Custom Notes');
    expect(updated).toContain('<!-- MYTOOL:START -->');
  });
});
```

---

## Complete Example: Deployment Tool

Here's a full implementation for a deployment tool:

```typescript
// src/commands.ts
export const DEPLOY_COMMAND = {
  id: 'deploy',
  description: 'Deploy application with validation and rollback safety',
  content: `**Guardrails**
- Always validate configuration before deployment
- Confirm target environment with user for production
- Never deploy with failing tests
- Monitor health checks after deployment
- Be ready to rollback if issues arise

**Steps**
1. Check prerequisites:
   - Run \`git status\` to ensure working directory is clean
   - Run \`mytool test --suite=smoke\` to verify tests pass
   - Run \`mytool validate --strict\` and resolve all issues

2. Prepare deployment:
   - Determine target environment (staging/production)
   - If production, ask user to explicitly confirm
   - Run \`mytool deploy --dry-run <env>\` to preview changes
   - Show user what will change and ask for confirmation

3. Execute deployment:
   - Run \`mytool deploy <env> --wait\`
   - Monitor deployment progress in real-time
   - Wait for deployment to complete

4. Verify deployment:
   - Run \`mytool health-check <env>\`
   - Check that all health checks pass
   - Run \`mytool test --suite=integration --env <env>\`

5. Post-deployment:
   - If all checks pass:
     - Create git tag: \`git tag v$(mytool version)\`
     - Notify user of successful deployment
   - If any checks fail:
     - Run \`mytool rollback <env>\`
     - Notify user of rollback
     - Show error logs: \`mytool logs <env> --tail 100\`

**Reference**
- View deployments: \`mytool list --env <env>\`
- Check status: \`mytool status <env>\`
- View logs: \`mytool logs <env> --tail 100 --follow\`
- Manual rollback: \`mytool rollback <env> --to <version>\`
- Health check: \`mytool health-check <env>\`

**Troubleshooting**

If deployment hangs:
\`\`\`bash
# Check deployment status
mytool status <env>

# View recent logs
mytool logs <env> --tail 200

# Check for resource constraints
mytool resources <env>
\`\`\`

If health checks fail:
\`\`\`bash
# View detailed health status
mytool health-check <env> --verbose

# Check individual services
mytool services <env>

# Rollback to previous version
mytool rollback <env>
\`\`\`
`
};

export const INIT_COMMAND = {
  id: 'init',
  description: 'Initialize a new project with MyTool',
  content: `**Steps**
1. Run \`mytool init\` and answer all prompts
2. Review generated configuration:
   - Open \`mytool.config.json\`
   - Verify project name, version, and settings
   - Adjust any values as needed

3. Validate configuration:
   - Run \`mytool validate --strict\`
   - Resolve any validation errors
   - Re-run until all checks pass

4. Set up environments:
   - Run \`mytool env add staging\`
   - Run \`mytool env add production\`
   - Configure environment-specific settings

5. Initialize version control:
   - Run \`git add mytool.config.json\`
   - Run \`git commit -m "Initialize MyTool configuration"\`

**Reference**
- View config: \`cat mytool.config.json\`
- Reconfigure: \`mytool init --reconfigure\`
- Add environment: \`mytool env add <name>\`
`
};

// src/init.ts
import { CursorConfigurator } from './configurators/cursor';
import { ClaudeConfigurator } from './configurators/claude';
import { DEPLOY_COMMAND, INIT_COMMAND } from './commands';

export async function init(projectPath: string) {
  console.log('🚀 Initializing MyTool...\n');
  
  const commands = [DEPLOY_COMMAND, INIT_COMMAND];
  const configurators = [
    new CursorConfigurator(),
    new ClaudeConfigurator(),
  ];
  
  for (const configurator of configurators) {
    console.log(`📝 Setting up ${configurator.toolName}...`);
    const files = await configurator.generate(projectPath, commands);
    files.forEach(f => console.log(`   ✅ ${f}`));
  }
  
  console.log('\n✨ Setup complete!\n');
  console.log('📌 Next steps:');
  console.log('   1. Restart your AI tool');
  console.log('   2. Type /mytool-init to initialize your project');
}

// Usage
if (import.meta.url === `file://${process.argv[1]}`) {
  init(process.cwd());
}
```

---

## Conclusion

The OpenSpec pattern provides a simple, effective way to integrate CLI tools with AI coding assistants:

1. **No server infrastructure** - Just markdown files
2. **Works with 15+ AI tools** - Single implementation
3. **User-customizable** - Developers can extend instructions
4. **Updateable** - Managed content blocks preserve customizations
5. **Context-efficient** - Zero token overhead vs. MCP

Start with 2-3 core commands, test thoroughly, then expand to more advanced workflows.

### Resources

- **OpenSpec Source**: https://github.com/Fission-AI/OpenSpec
- **AGENTS.md Convention**: https://agents.md/
- **Example Implementation**: See this repository's `/src/core/configurators/` directory

### Questions?

Open an issue or discussion in your project repo!
