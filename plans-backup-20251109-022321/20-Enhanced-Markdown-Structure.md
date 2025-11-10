# 20 - Enhanced Markdown Structure & Prompt Management

**Leveraging markdown-query's GFM Support and Promptdown for Rich Metadata**

---

## Executive Summary

We can significantly enhance Quaid's capabilities by:

1. **Rich Markdown Parsing** - Use markdown-query's GFM features to extract structured data (code languages, tables, task lists, etc.)
2. **Prompt Management** - Use Promptdown to define and manage AI prompts in version-controlled markdown files
3. **Enhanced Metadata** - Extract more structured information from fragments for better search and classification

**Key Innovation**: Treat markdown as a rich, queryable data structure rather than just text.

---

## Part 1: Enhanced Markdown Structure Parsing

### GFM Features in markdown-query

markdown-query supports GitHub Flavored Markdown, giving us powerful structural queries:

```python
import mq

# Code block language extraction
.code("python")        # Get all Python code blocks
.code("javascript")    # Get all JavaScript code blocks
.code.lang             # Get language names from all code blocks

# Task lists
.task_list             # Get all task lists
.task_list[checked]    # Get completed tasks
.task_list[!checked]   # Get pending tasks

# Tables
.table                 # Get all tables
.table.header          # Get table headers
.table.row             # Get table rows

# Other GFM features
.strikethrough         # Get strikethrough text
.autolink              # Get autolinks
```

### Enhanced Fragment Metadata Structure

**Current fragment structure**:
```python
{
    "id": "2025-11-09-auth-001",
    "title": "JWT Authentication",
    "content": "...",
    "tags": ["auth", "jwt"],
    "type": "implementation",
    "importance": "high"
}
```

**Enhanced fragment structure** (with markdown-query parsing):
```python
{
    "id": "2025-11-09-auth-001",
    "title": "JWT Authentication",
    "content": "...",
    
    # Basic metadata (as before)
    "tags": ["auth", "jwt"],
    "type": "implementation",
    "importance": "high",
    "completeness": "complete",
    
    # NEW: Code structure
    "code_blocks": [
        {
            "language": "python",
            "content": "def authenticate(token: str)...",
            "metadata": {
                "importance": "high",
                "status": "production",
                "keywords": ["auth", "jwt", "validate"]
            },
            "line_range": [45, 67]
        },
        {
            "language": "javascript",
            "content": "const verifyToken = (token) => {...",
            "metadata": {
                "importance": "medium",
                "status": "experimental"
            },
            "line_range": [102, 128]
        }
    ],
    
    # NEW: Structural elements
    "headings": [
        {"level": 1, "text": "JWT Authentication", "line": 1},
        {"level": 2, "text": "Overview", "line": 5},
        {"level": 2, "text": "Core Concept", "line": 15},
        {"level": 2, "text": "Implementation", "line": 35}
    ],
    
    # NEW: Decision records (from blockquotes)
    "decisions": [
        {
            "decision": "Use JWT over sessions",
            "date": "2025-11-09",
            "status": "approved",
            "impact": "high",
            "rationale": "Need stateless scaling...",
            "line_range": [20, 25]
        }
    ],
    
    # NEW: Task tracking
    "tasks": {
        "total": 5,
        "completed": 3,
        "pending": 2,
        "items": [
            {"checked": true, "text": "Implement token validation"},
            {"checked": true, "text": "Add refresh token support"},
            {"checked": true, "text": "Write unit tests"},
            {"checked": false, "text": "Add rate limiting"},
            {"checked": false, "text": "Document API"}
        ]
    },
    
    # NEW: Tables (for data, comparisons, etc.)
    "tables": [
        {
            "headers": ["Method", "Security", "Performance"],
            "rows": [
                ["JWT", "High", "Fast"],
                ["Session", "Medium", "Slower"]
            ],
            "purpose": "comparison"
        }
    ],
    
    # NEW: Links and references
    "links": {
        "internal": [
            {"text": "Session Management", "target": "./session-001.md", "priority": "high"},
            {"text": "OAuth Integration", "target": "./oauth-002.md", "priority": "medium"}
        ],
        "external": [
            {"text": "JWT RFC", "url": "https://tools.ietf.org/html/rfc7519"}
        ]
    },
    
    # NEW: Admonitions (important notes)
    "admonitions": [
        {
            "type": "important",
            "title": "Security Note",
            "content": "Always validate token signatures...",
            "boost": 5
        },
        {
            "type": "warning",
            "title": "Rate Limiting",
            "content": "Implement rate limiting to prevent abuse...",
            "boost": 3
        }
    ]
}
```

### Implementation: Enhanced Parser

```python
import mq
from typing import Dict, List, Any
import re

class EnhancedMarkdownParser:
    """
    Parse markdown fragments with rich structural extraction
    using markdown-query's GFM support
    """
    
    def __init__(self):
        self.options = mq.Options()
        self.options.input_format = mq.InputFormat.MARKDOWN
    
    def parse_fragment(self, content: str) -> Dict[str, Any]:
        """
        Extract rich structural metadata from markdown content
        """
        metadata = {
            "code_blocks": self._extract_code_blocks(content),
            "headings": self._extract_headings(content),
            "decisions": self._extract_decisions(content),
            "tasks": self._extract_tasks(content),
            "tables": self._extract_tables(content),
            "links": self._extract_links(content),
            "admonitions": self._extract_admonitions(content)
        }
        
        return metadata
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract all code blocks with language and metadata
        """
        # Get all code blocks
        code_blocks_result = mq.run(".code", content, self.options)
        
        # Get language names
        languages_result = mq.run(".code.lang", content, self.options)
        
        code_blocks = []
        for i, block in enumerate(code_blocks_result):
            lang = languages_result[i].text if i < len(languages_result) else "text"
            
            # Extract metadata from code block comments
            block_metadata = self._parse_code_metadata(block.text)
            
            code_blocks.append({
                "language": lang,
                "content": block.text,
                "metadata": block_metadata,
                "line_range": self._get_line_range(content, block.text)
            })
        
        return code_blocks
    
    def _parse_code_metadata(self, code: str) -> Dict[str, Any]:
        """
        Extract metadata from code block comments
        
        Expected format:
        # IMPORTANCE: high
        # STATUS: production
        # KEYWORDS: auth, jwt, validate
        """
        metadata = {}
        
        # Extract IMPORTANCE
        importance_match = re.search(r'#\s*IMPORTANCE:\s*(\w+)', code)
        if importance_match:
            metadata['importance'] = importance_match.group(1).lower()
        
        # Extract STATUS
        status_match = re.search(r'#\s*STATUS:\s*(\w+)', code)
        if status_match:
            metadata['status'] = status_match.group(1).lower()
        
        # Extract KEYWORDS
        keywords_match = re.search(r'#\s*KEYWORDS:\s*(.+)', code)
        if keywords_match:
            metadata['keywords'] = [
                k.strip() for k in keywords_match.group(1).split(',')
            ]
        
        return metadata
    
    def _extract_headings(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract all headings with level and position
        """
        headings = []
        
        for level in range(1, 7):  # H1 through H6
            selector = f".h{level}"
            result = mq.run(selector, content, self.options)
            
            for heading in result:
                text = heading.text.lstrip('#').strip()
                line_num = self._get_line_number(content, heading.text)
                
                headings.append({
                    "level": level,
                    "text": text,
                    "line": line_num
                })
        
        # Sort by line number
        headings.sort(key=lambda x: x['line'])
        return headings
    
    def _extract_decisions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract decision records from blockquotes
        
        Expected format:
        > **Decision**: Use JWT tokens
        > **Date**: 2025-11-09
        > **Status**: Approved
        > **Impact**: High
        """
        decisions = []
        
        # Get all blockquotes
        blockquotes_result = mq.run(".blockquote", content, self.options)
        
        for bq in blockquotes_result:
            text = bq.text
            
            # Check if it's a decision blockquote
            if "**Decision**:" in text or "**decision**:" in text:
                decision = self._parse_decision_blockquote(text)
                if decision:
                    decision['line_range'] = self._get_line_range(content, text)
                    decisions.append(decision)
        
        return decisions
    
    def _parse_decision_blockquote(self, text: str) -> Dict[str, Any]:
        """
        Parse structured decision information from blockquote
        """
        decision = {}
        
        # Extract decision
        decision_match = re.search(r'\*\*Decision\*\*:\s*(.+)', text, re.I)
        if decision_match:
            decision['decision'] = decision_match.group(1).strip()
        
        # Extract date
        date_match = re.search(r'\*\*Date\*\*:\s*(\d{4}-\d{2}-\d{2})', text, re.I)
        if date_match:
            decision['date'] = date_match.group(1)
        
        # Extract status
        status_match = re.search(r'\*\*Status\*\*:\s*(\w+)', text, re.I)
        if status_match:
            decision['status'] = status_match.group(1).lower()
        
        # Extract impact
        impact_match = re.search(r'\*\*Impact\*\*:\s*(\w+)', text, re.I)
        if impact_match:
            decision['impact'] = impact_match.group(1).lower()
        
        # Get rationale (remaining text)
        lines = text.split('\n')
        rationale_lines = [
            line for line in lines 
            if not any(marker in line for marker in ['**Decision**', '**Date**', '**Status**', '**Impact**'])
        ]
        if rationale_lines:
            decision['rationale'] = '\n'.join(rationale_lines).strip()
        
        return decision if 'decision' in decision else None
    
    def _extract_tasks(self, content: str) -> Dict[str, Any]:
        """
        Extract task lists with completion status
        """
        # Get all task items
        all_tasks_result = mq.run(".task_list", content, self.options)
        checked_tasks_result = mq.run(".task_list[checked]", content, self.options)
        
        tasks = []
        for task in all_tasks_result:
            text = task.text
            is_checked = any(ct.text == text for ct in checked_tasks_result)
            
            # Clean up task text
            clean_text = re.sub(r'^\s*[-*]\s*\[[x ]\]\s*', '', text, flags=re.I)
            
            tasks.append({
                "checked": is_checked,
                "text": clean_text.strip()
            })
        
        return {
            "total": len(tasks),
            "completed": sum(1 for t in tasks if t["checked"]),
            "pending": sum(1 for t in tasks if not t["checked"]),
            "items": tasks
        }
    
    def _extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract tables with headers and rows
        """
        tables = []
        
        # Get all tables
        tables_result = mq.run(".table", content, self.options)
        
        for table in tables_result:
            table_data = self._parse_table(table.text)
            if table_data:
                tables.append(table_data)
        
        return tables
    
    def _parse_table(self, table_text: str) -> Dict[str, Any]:
        """
        Parse markdown table into structured data
        """
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return None
        
        # First line is headers
        headers = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
        
        # Skip separator line (second line)
        # Remaining lines are rows
        rows = []
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                rows.append(cells)
        
        return {
            "headers": headers,
            "rows": rows,
            "purpose": self._infer_table_purpose(headers)
        }
    
    def _infer_table_purpose(self, headers: List[str]) -> str:
        """
        Infer table purpose from headers
        """
        headers_lower = [h.lower() for h in headers]
        
        if any(word in ' '.join(headers_lower) for word in ['vs', 'comparison', 'versus']):
            return "comparison"
        elif 'metric' in ' '.join(headers_lower) or 'value' in ' '.join(headers_lower):
            return "metrics"
        elif 'step' in ' '.join(headers_lower) or 'action' in ' '.join(headers_lower):
            return "process"
        else:
            return "data"
    
    def _extract_links(self, content: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract internal and external links
        """
        # Get all links
        links_result = mq.run(".link", content, self.options)
        
        internal = []
        external = []
        
        for link in links_result:
            text = link.text
            
            # Extract link target from markdown [text](target)
            match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', text)
            if match:
                link_text = match.group(1)
                target = match.group(2)
                
                # Check if internal or external
                if target.startswith('http://') or target.startswith('https://'):
                    external.append({
                        "text": link_text,
                        "url": target
                    })
                else:
                    # Extract priority if present
                    priority = self._extract_link_priority(content, text)
                    
                    internal.append({
                        "text": link_text,
                        "target": target,
                        "priority": priority
                    })
        
        return {
            "internal": internal,
            "external": external
        }
    
    def _extract_link_priority(self, content: str, link_text: str) -> str:
        """
        Extract priority marker from link (e.g., [[high-priority]])
        """
        # Look for pattern: - [[priority]] [Link Text](target)
        pattern = r'-\s*\[\[(\w+-priority)\]\]\s*' + re.escape(link_text)
        match = re.search(pattern, content)
        
        if match:
            return match.group(1).replace('-priority', '')
        
        return "medium"  # Default priority
    
    def _extract_admonitions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract admonition blocks (!!!)
        
        Format:
        !!! type "Title"
            Content here
        !!!
        """
        admonitions = []
        
        # Pattern for admonitions
        pattern = r'!!!\s+(\w+)\s+"([^"]+)"\s*\n(.*?)\n!!!'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            adm_type = match.group(1).lower()
            title = match.group(2)
            adm_content = match.group(3).strip()
            
            # Get boost value based on type
            boost = {
                'important': 5,
                'critical': 5,
                'warning': 3,
                'tip': 4,
                'note': 1,
                'deprecated': -5
            }.get(adm_type, 0)
            
            admonitions.append({
                "type": adm_type,
                "title": title,
                "content": adm_content,
                "boost": boost
            })
        
        return admonitions
    
    def _get_line_range(self, content: str, text: str) -> List[int]:
        """
        Get line range for a text block within content
        """
        lines = content.split('\n')
        search_lines = text.split('\n')
        
        for i in range(len(lines) - len(search_lines) + 1):
            if lines[i:i+len(search_lines)] == search_lines:
                return [i + 1, i + len(search_lines)]
        
        return [0, 0]
    
    def _get_line_number(self, content: str, text: str) -> int:
        """
        Get line number for a text within content
        """
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if text in line:
                return i + 1
        return 0
```

---

## Part 2: Prompt Management with Promptdown

### Why Use Promptdown?

**Benefits**:
1. **Version Control** - Prompts as markdown files in git
2. **Readable** - Easy to review and modify
3. **Template Support** - Dynamic prompt customization
4. **Separation of Concerns** - Prompts separate from code
5. **Collaboration** - Non-developers can edit prompts

### Prompt Structure for Quaid

**Directory structure**:
```
.quaid/
├── prompts/
│   ├── classification.prompt.md
│   ├── tag_extraction.prompt.md
│   ├── entity_extraction.prompt.md
│   ├── summarization.prompt.md
│   └── decision_analysis.prompt.md
```

### Example Prompts

**`.quaid/prompts/classification.prompt.md`**:
```markdown
# Fragment Classification

## Developer Message

You are a technical documentation classifier. Your job is to analyze markdown 
fragments and classify them accurately based on their content and structure.

You must return ONLY valid JSON with this exact structure:
{
  "type": "concept|implementation|decision|reference|pattern",
  "importance": "high|medium|low",
  "completeness": "complete|partial|stub",
  "confidence": 0-100
}

## Conversation

**User:**
Please classify this fragment:

{fragment_content}

Key indicators:
- Type: Look for code blocks (implementation), decision markers (decision), 
  conceptual explanations (concept), external links (reference), reusable 
  patterns (pattern)
- Importance: Consider keywords like "critical", "important", length > 2000 chars,
  presence of admonitions
- Completeness: Check for multiple sections, examples, references

**Assistant:**
I'll analyze the fragment structure and content to provide an accurate classification.
```

**`.quaid/prompts/tag_extraction.prompt.md`**:
```markdown
# Tag Extraction

## Developer Message

You are an expert at extracting relevant tags from technical documentation.
Focus on technologies, concepts, and domain-specific terms.

Return ONLY a JSON array of 3-7 tags:
["tag1", "tag2", "tag3"]

## Conversation

**User:**
Extract tags from this content:

{fragment_content}

Additional context:
- Detected entities: {entities}
- Code languages: {code_languages}
- Main heading: {title}

**Assistant:**
I'll extract the most relevant and specific tags based on the content, 
focusing on technologies, concepts, and key terms.
```

**`.quaid/prompts/entity_extraction.prompt.md`**:
```markdown
# Entity Extraction

## Developer Message

You are a named entity extraction specialist for technical documentation.
Extract specific entities like technologies, libraries, frameworks, products,
and domain-specific concepts.

Return JSON:
{
  "technologies": ["tech1", "tech2"],
  "concepts": ["concept1", "concept2"],
  "products": ["product1"]
}

## Conversation

**User:**
Extract entities from:

Title: {title}
Content: {fragment_content}

**Assistant:**
I'll identify and categorize all relevant technical entities.
```

### Implementation: Prompt Manager

```python
from promptdown import StructuredPrompt
from pathlib import Path
from typing import Dict, Any, Optional
import json

class PromptManager:
    """
    Manage AI prompts using Promptdown
    """
    
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self._cache = {}
    
    def get_prompt(self, name: str) -> StructuredPrompt:
        """
        Load a prompt by name (with caching)
        """
        if name in self._cache:
            return self._cache[name]
        
        prompt_file = self.prompts_dir / f"{name}.prompt.md"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt not found: {name}")
        
        prompt = StructuredPrompt.from_promptdown_file(str(prompt_file))
        self._cache[name] = prompt
        
        return prompt
    
    def render_prompt(
        self, 
        name: str, 
        template_values: Dict[str, Any]
    ) -> StructuredPrompt:
        """
        Load and render a prompt with template values
        """
        prompt = self.get_prompt(name)
        
        # Create a copy to avoid modifying cached version
        rendered_prompt = StructuredPrompt.from_promptdown_string(
            prompt.to_promptdown_string()
        )
        
        # Apply template values
        rendered_prompt.apply_template_values(template_values)
        
        return rendered_prompt
    
    def get_messages(
        self, 
        name: str, 
        template_values: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Get chat completion messages for a prompt
        """
        if template_values:
            prompt = self.render_prompt(name, template_values)
        else:
            prompt = self.get_prompt(name)
        
        return prompt.to_chat_completion_messages()

class LocalLLMClassifier:
    """
    Classification using local LLM with Promptdown prompts
    """
    
    def __init__(self, prompts_dir: Path, model: str = "phi-2"):
        import ollama
        self.ollama = ollama
        self.model = model
        self.prompt_manager = PromptManager(prompts_dir)
    
    def classify_fragment(
        self, 
        fragment_content: str,
        title: str = ""
    ) -> Dict[str, Any]:
        """
        Classify a fragment using prompt template
        """
        # Render classification prompt
        messages = self.prompt_manager.get_messages(
            "classification",
            template_values={
                "fragment_content": fragment_content[:1500],  # Truncate for speed
                "title": title
            }
        )
        
        # Get classification from LLM
        response = self.ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": 0.3,
                "num_predict": 200
            }
        )
        
        try:
            return json.loads(response['message']['content'])
        except:
            # Fallback to rule-based
            return self._rule_based_fallback(fragment_content)
    
    def extract_tags(
        self, 
        fragment_content: str,
        entities: List[str],
        code_languages: List[str],
        title: str
    ) -> List[str]:
        """
        Extract tags using prompt template
        """
        messages = self.prompt_manager.get_messages(
            "tag_extraction",
            template_values={
                "fragment_content": fragment_content[:1000],
                "entities": ", ".join(entities),
                "code_languages": ", ".join(code_languages),
                "title": title
            }
        )
        
        response = self.ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": 0.4,
                "num_predict": 100
            }
        )
        
        try:
            return json.loads(response['message']['content'])
        except:
            return []
```

---

## Part 3: Integration

### Enhanced Fragment Processing Pipeline

```python
from pathlib import Path

class FragmentProcessor:
    """
    Process markdown fragments with rich structure extraction
    """
    
    def __init__(self, config: dict):
        self.parser = EnhancedMarkdownParser()
        self.prompt_manager = PromptManager(Path(".quaid/prompts"))
        
        # Optional: LLM for classification
        if config.get('use_llm_classification'):
            self.classifier = LocalLLMClassifier(
                Path(".quaid/prompts"),
                model=config.get('llm_model', 'phi-2')
            )
        else:
            self.classifier = None
    
    def process_fragment(
        self, 
        content: str, 
        frontmatter: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a fragment with enhanced parsing
        """
        # Extract rich structure
        structure = self.parser.parse_fragment(content)
        
        # Combine with frontmatter
        fragment = {
            **frontmatter,
            **structure
        }
        
        # Enhance with LLM classification if enabled
        if self.classifier:
            classification = self.classifier.classify_fragment(
                content,
                title=frontmatter.get('title', '')
            )
            fragment.update(classification)
            
            # Extract tags
            tags = self.classifier.extract_tags(
                content,
                entities=frontmatter.get('entities', []),
                code_languages=[cb['language'] for cb in structure['code_blocks']],
                title=frontmatter.get('title', '')
            )
            fragment['tags'] = tags
        
        return fragment
```

---

## Benefits

### Rich Metadata Benefits

1. **Better Search** - Query by code language, task status, decision status
2. **Smart Filtering** - Find incomplete fragments, pending tasks
3. **Code Discovery** - Search for Python implementations, JavaScript examples
4. **Decision Tracking** - Find all approved decisions, high-impact changes
5. **Progress Tracking** - See task completion across fragments

### Promptdown Benefits

1. **Maintainability** - Prompts are readable markdown, easy to modify
2. **Version Control** - Track prompt changes in git
3. **Collaboration** - Non-developers can improve prompts
4. **Testing** - Easy to test different prompts
5. **Consistency** - Standardized prompt structure

---

## Configuration

**`.quaid/config.toml`**:
```toml
[markdown_parsing]
# Enable enhanced markdown structure parsing
enabled = true

# Extract code block metadata
parse_code_metadata = true

# Extract decision records
parse_decisions = true

# Extract task lists
parse_tasks = true

# Extract tables
parse_tables = true

# Extract admonitions
parse_admonitions = true

[prompts]
# Use Promptdown for prompt management
enabled = true

# Prompts directory
dir = ".quaid/prompts"

# Use LLM for classification (requires phi-2 or similar)
use_llm_classification = false  # Start with false, enable if needed

# LLM model for classification
llm_model = "phi-2"
```

---

## Implementation Roadmap

### Phase 1: Enhanced Parsing (Week 1)
- [ ] Implement EnhancedMarkdownParser
- [ ] Add code block language extraction
- [ ] Add heading extraction
- [ ] Test with sample fragments

### Phase 2: Structured Metadata (Week 2)
- [ ] Add decision extraction
- [ ] Add task list extraction
- [ ] Add table extraction
- [ ] Add link extraction
- [ ] Update fragment schema

### Phase 3: Promptdown Integration (Week 3)
- [ ] Install Promptdown
- [ ] Create prompt templates
- [ ] Implement PromptManager
- [ ] Test prompt rendering

### Phase 4: LLM Integration (Week 4)
- [ ] Integrate with local LLM (optional)
- [ ] Create classification prompts
- [ ] Create tag extraction prompts
- [ ] Test end-to-end

---

## Conclusion

By leveraging markdown-query's GFM support and Promptdown for prompt management, we create a system that:

✅ **Extracts rich structure** from markdown (code languages, tasks, decisions, tables)
✅ **Manages prompts** as version-controlled markdown files
✅ **Improves search** with structured metadata
✅ **Enables tracking** of code, tasks, and decisions
✅ **Maintains simplicity** - still 100% local and private

**The Result**: Markdown becomes a rich, queryable database rather than just formatted text.

---

**Previous**: [19-Reranking-Integration.md](19-Reranking-Integration.md)

**Version**: 1.0  
**Last Updated**: 2025-11-09
