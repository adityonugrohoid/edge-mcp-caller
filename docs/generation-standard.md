# Generation Standard — All 14 Tools

Data generation rules for the edge MCP caller specialist model. See `docs/mcp-query-behavior-analysis.md` for the failure analysis that motivated these extraction-only rules.

---

## Principle

The specialist model is a **router + extractor**. It picks the correct tool and extracts arguments verbatim from the user's query. It does NOT invent paths, generate code, or guess filenames.

> **Every argument in the model's output must be extractable from the user's query.**

## User Assumption

The user knows how to use MCP servers. They:
- Specify exact paths: "read src/config.json", not "show me the config"
- Specify exact content for writes: "write 'hello' to output.txt"
- Know which tools exist, but may phrase requests naturally

---

## Output Format

Each training example is a single JSONL line:

```json
{"messages":[{"role":"user","content":"<query>"},{"role":"assistant","content":"{\"tool\":\"<name>\",\"args\":{...}}"}]}
```

The assistant content is a **JSON string escaped within the outer JSON**. Critical escaping rules:
- Double quotes inside content args must be escaped: `\"` in the inner JSON becomes `\\\"` in the JSONL
- Newlines in content: `\n` in the inner JSON becomes `\\n` in the JSONL
- Backslashes: `\\` becomes `\\\\`

Example with content containing quotes:
```json
{"messages":[{"role":"user","content":"write 'name=\"test\"' to config.txt"},{"role":"assistant","content":"{\"tool\":\"write_file\",\"args\":{\"path\":\"config.txt\",\"content\":\"name=\\\"test\\\"\"}}"}]}
```

---

## Tools

### Filesystem MCP Server (8 tools)

#### 1. list_directory
- **Args**: `{"path": "<dir_path>"}`
- **Convention**: path always ends with `/`
- **Examples**:
  - "list files in src/" → `{"tool":"list_directory","args":{"path":"src/"}}`
  - "what's in the docs/ directory" → `{"tool":"list_directory","args":{"path":"docs/"}}`
  - "show contents of tests/unit/" → `{"tool":"list_directory","args":{"path":"tests/unit/"}}`

#### 2. read_file
- **Args**: `{"path": "<file_path>"}`
- **Convention**: path never ends with `/`
- **Examples**:
  - "read README.md" → `{"tool":"read_file","args":{"path":"README.md"}}`
  - "show me src/main.py" → `{"tool":"read_file","args":{"path":"src/main.py"}}`
  - "open package.json" → `{"tool":"read_file","args":{"path":"package.json"}}`

#### 3. search_files
- **Args**: `{"path": "<dir_path>", "pattern": "<glob>"}`
- **Convention**: path ends with `/`, pattern is Unix glob (`*`, `?`, `[abc]`)
- **Examples**:
  - "find all .py files in src/" → `{"tool":"search_files","args":{"path":"src/","pattern":"*.py"}}`
  - "search for *.yaml in config/" → `{"tool":"search_files","args":{"path":"config/","pattern":"*.yaml"}}`
  - "find test_*.js files in tests/" → `{"tool":"search_files","args":{"path":"tests/","pattern":"test_*.js"}}`

#### 4. write_file
- **Args**: `{"path": "<file_path>", "content": "<text>"}`
- **Convention**: path never ends with `/`
- **Content rules**: explicitly quoted in query, under ~100 chars, plain text (not code)
- **Examples**:
  - "write 'hello world' to output.txt" → `{"tool":"write_file","args":{"path":"output.txt","content":"hello world"}}`
  - "save 'DB_HOST=localhost' to config.env" → `{"tool":"write_file","args":{"path":"config.env","content":"DB_HOST=localhost"}}`
  - "put 'TODO: refactor' in notes.txt" → `{"tool":"write_file","args":{"path":"notes.txt","content":"TODO: refactor"}}`

#### 5. create_directory
- **Args**: `{"path": "<dir_path>"}`
- **Convention**: path always ends with `/`
- **Examples**:
  - "create a utils/ directory" → `{"tool":"create_directory","args":{"path":"utils/"}}`
  - "make tests/unit/" → `{"tool":"create_directory","args":{"path":"tests/unit/"}}`
  - "create folder src/components/" → `{"tool":"create_directory","args":{"path":"src/components/"}}`

#### 6. edit_file
- **Args**: `{"path": "<file_path>", "old_text": "<find>", "new_text": "<replace>"}`
- **Convention**: both old_text and new_text must be quoted/extractable from query
- **Bridge note**: converts to MCP format `{path, edits: [{oldText, newText}]}`
- **Examples**:
  - "in config.json replace 'DEBUG=false' with 'DEBUG=true'" → `{"tool":"edit_file","args":{"path":"config.json","old_text":"DEBUG=false","new_text":"DEBUG=true"}}`
  - "change 'port=3000' to 'port=8080' in server.conf" → `{"tool":"edit_file","args":{"path":"server.conf","old_text":"port=3000","new_text":"port=8080"}}`
  - "in .env replace 'dev' with 'prod'" → `{"tool":"edit_file","args":{"path":".env","old_text":"dev","new_text":"prod"}}`

#### 7. move_file
- **Args**: `{"source": "<path>", "destination": "<path>"}`
- **Convention**: both paths in the query, use file or dir conventions as appropriate
- **Examples**:
  - "move old.txt to archive/old.txt" → `{"tool":"move_file","args":{"source":"old.txt","destination":"archive/old.txt"}}`
  - "rename utils.py to helpers.py" → `{"tool":"move_file","args":{"source":"utils.py","destination":"helpers.py"}}`
  - "move src/temp/ to src/legacy/" → `{"tool":"move_file","args":{"source":"src/temp/","destination":"src/legacy/"}}`

#### 8. directory_tree
- **Args**: `{"path": "<dir_path>"}`
- **Convention**: path always ends with `/`
- **Examples**:
  - "show the full tree of src/" → `{"tool":"directory_tree","args":{"path":"src/"}}`
  - "get directory tree for lib/" → `{"tool":"directory_tree","args":{"path":"lib/"}}`
  - "tree structure of project/" → `{"tool":"directory_tree","args":{"path":"project/"}}`

### Git MCP Server (6 tools)

`repo_path` is injected by the MCP client bridge — the model does NOT output it.

#### 9. git_status
- **Args**: `{}`
- **Examples**:
  - "check git status" → `{"tool":"git_status","args":{}}`
  - "show repo status" → `{"tool":"git_status","args":{}}`
  - "any uncommitted changes?" → `{"tool":"git_status","args":{}}`

#### 10. git_diff_staged
- **Args**: `{}`
- **Examples**:
  - "show staged changes" → `{"tool":"git_diff_staged","args":{}}`
  - "what's staged for commit" → `{"tool":"git_diff_staged","args":{}}`
  - "diff staged files" → `{"tool":"git_diff_staged","args":{}}`

#### 11. git_commit
- **Args**: `{"message": "<commit_message>"}`
- **Convention**: message must be quoted/extractable from query
- **Examples**:
  - "commit with message 'fix auth bug'" → `{"tool":"git_commit","args":{"message":"fix auth bug"}}`
  - "commit: 'add user model'" → `{"tool":"git_commit","args":{"message":"add user model"}}`
  - "git commit 'refactor: clean up utils'" → `{"tool":"git_commit","args":{"message":"refactor: clean up utils"}}`

#### 12. git_log
- **Args**: `{}` or `{"max_count": <integer>}`
- **Convention**: max_count is an **integer**, not a string. Only include if user specifies a number.
- **Examples**:
  - "show commit history" → `{"tool":"git_log","args":{}}`
  - "show last 5 commits" → `{"tool":"git_log","args":{"max_count":5}}`
  - "git log 10" → `{"tool":"git_log","args":{"max_count":10}}`
  - "recent commits" → `{"tool":"git_log","args":{}}`

#### 13. git_branch
- **Args**: `{}`
- **Examples**:
  - "list branches" → `{"tool":"git_branch","args":{}}`
  - "show all branches" → `{"tool":"git_branch","args":{}}`
  - "what branches exist" → `{"tool":"git_branch","args":{}}`

#### 14. git_create_branch
- **Args**: `{"branch_name": "<name>"}` or `{"branch_name": "<name>", "base_branch": "<base>"}`
- **Convention**: branch names and base branch must be in query
- **Examples**:
  - "create branch feature/auth" → `{"tool":"git_create_branch","args":{"branch_name":"feature/auth"}}`
  - "create branch fix/login from develop" → `{"tool":"git_create_branch","args":{"branch_name":"fix/login","base_branch":"develop"}}`
  - "new branch hotfix/crash off main" → `{"tool":"git_create_branch","args":{"branch_name":"hotfix/crash","base_branch":"main"}}`

---

## Categories

### Clean (70%)
Well-formed queries from an MCP-savvy user. Clear intent, explicit paths/args.

```
"list files in src/"
"read README.md"
"find *.py files in lib/"
"write 'hello' to output.txt"
"in config.json replace 'v1' with 'v2'"
"move old.txt to archive/old.txt"
"show tree of docs/"
"check git status"
"commit with message 'fix: auth bug'"
"show last 5 commits"
"create branch feature/auth from develop"
```

### Messy (15%)
Same intent, sloppy typing/speech. All args are still extractable.

Messy types:
- **Typos**: "reed README.md", "lits files in src/", "comit 'fix bug'"
- **Abbreviations**: "ls src/", "mk tests/", "mv old.txt archive/"
- **Filler**: "um can you read README.md", "like show me src/"
- **Voice transcription**: "read readme dot md", "list src slash utils"
- **Grammar errors**: "show me the file README.md please", "what is in src/ directory"

```
"reed README.md plz" → read_file, path="README.md"
"lits src/" → list_directory, path="src/"
"comit msg 'fix bug'" → git_commit, message="fix bug"
"um find *.py in lib/" → search_files, path="lib/", pattern="*.py"
"mv old.txt to backup/old.txt" → move_file, source="old.txt", destination="backup/old.txt"
```

### Disambiguation (15%)
Queries where multiple tools could apply. Tests routing intelligence. Path/args are still obvious — only the TOOL is ambiguous.

| Query | Correct | Confused With | Why |
|-------|---------|---------------|-----|
| "is there a tests/ folder?" | list_directory | create_directory | Checking existence = list |
| "what Python files are in src/" | search_files | list_directory | Extension filter = search |
| "show full structure of lib/" | directory_tree | list_directory | "full structure" = tree, not flat list |
| "config.txt should say 'max_retries=3'" | write_file | read_file | "should say" = write intent |
| "in app.py replace 'v1' with 'v2'" | edit_file | write_file | Targeted replacement = edit, not overwrite |
| "put old.txt in archive/" | move_file | write_file | Relocating = move |
| "any uncommitted work?" | git_status | git_diff_staged | Status overview, not staged diff |
| "what changed recently?" | git_log | git_status | History = log |
| "we need a hotfix/crash branch" | git_create_branch | git_branch | "need" = create |
| "show directory contents of src/" | list_directory | directory_tree | "contents" = flat list |
| "save tree output of docs/" | directory_tree | write_file | "tree output" = tree tool |

---

## Extraction Rules

1. **Every path must come from the query** — user says "src/utils/", model outputs "src/utils/"
2. **write_file content must be quoted in query** — user says "write 'hello' to...", model extracts "hello"
3. **edit_file old_text/new_text must be quoted in query** — user says "replace 'X' with 'Y'", model extracts both
4. **git_commit message must be quoted/extractable** — user says "commit 'fix bug'", model extracts "fix bug"
5. **search_files pattern must be unambiguous** — "find *.py" → pattern="*.py" (clear glob)
6. **git_log max_count must be a number in query** — "last 5 commits" → max_count=5 (integer, not string)
7. **move_file needs both source and destination in query** — "move X to Y" → source=X, destination=Y
8. **git_create_branch name must be in query** — "create branch feature/auth" → branch_name="feature/auth"
9. **No invented args** — if you can't determine the exact arg from the query, rewrite the query

## Path Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Directory args | Trailing `/` | `src/`, `tests/unit/` |
| File args | No trailing `/` | `README.md`, `src/main.py` |
| Relative paths | No leading `/` or `./` | `src/`, not `/src/` or `./src/` |
| Nested paths | Full path in query | "in src/components/" → `src/components/` |
| move_file dirs | Follow dir/file convention | `src/temp/` (dir), `old.txt` (file) |

---

## Target Distribution

**~14,000 total examples** (~1,000 per tool), 90/10 train/eval split.

| Per tool | Clean (70%) | Messy (15%) | Disambiguation (15%) | Total |
|----------|-------------|-------------|----------------------|-------|
| Target | ~700 | ~150 | ~150 | ~1,000 |

Final: ~12,600 train + ~1,400 eval (100/tool in eval).

Overshoot by ~15% to account for dedup/validation losses.

Eval examples include `"category"` tag for per-category reporting:
```json
{"messages":[...],"category":"messy"}
```

---

## Validation Checklist

Before accepting each example, verify:

- [ ] Valid JSON with `messages` array containing exactly 2 messages (user + assistant)
- [ ] Assistant content parses as valid JSON with `tool` and `args` fields
- [ ] `tool` is one of the 14 valid tools
- [ ] All required args present for the tool
- [ ] All string arg values are strings, max_count is integer
- [ ] Directory paths end with `/` (list_directory, create_directory, directory_tree, search_files path)
- [ ] File paths don't end with `/` (read_file, write_file, edit_file)
- [ ] No leading `/` or `./` in any path
- [ ] search_files pattern is not bare `*` (too broad)
- [ ] write_file content is non-empty and under ~100 chars
- [ ] edit_file old_text and new_text are non-empty and different from each other
- [ ] git_commit message is non-empty
- [ ] git_create_branch branch_name is non-empty
- [ ] **Every arg value is extractable from the user query** (the fundamental rule)
- [ ] move_file source ≠ destination

---

## Scaling Benchmark Plan

Train ONE model on all 14 tools, then benchmark subsets:

| Eval subset | Tools | What it tests |
|-------------|-------|---------------|
| 3-tool | list_directory, read_file, search_files | Baseline read operations |
| 5-tool | + write_file, create_directory | Read/write disambiguation |
| 8-tool | + edit_file, move_file, directory_tree | Similar-tool disambiguation |
| 14-tool | + 6 git tools | Cross-server routing |

This produces a scaling curve showing where 270M accuracy degrades as tool count increases.
