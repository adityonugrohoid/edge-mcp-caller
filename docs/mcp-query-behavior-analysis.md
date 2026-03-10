# MCP Query Behavior Analysis — Dataset Generation Rules

## Context

An early benchmark showed 95.3% tool routing but only 48.6% args exact-match. Root cause analysis revealed the synthetic training data violated how MCP queries actually work in deployment — the data generator invented arbitrary paths and long code content instead of using extractable arguments. This document captures the failure analysis that motivated the extraction-only rules in `docs/generation-standard.md`.

## How MCP Filesystem Queries Actually Work

### The deployment model

```
User says something → Specialist extracts tool + args → MCP server executes
```

The specialist is a **router and extractor**, not a generator. It:
1. **Routes** the query to the correct tool (one of 14 tools across filesystem + git)
2. **Extracts** arguments from what the user said
3. Does NOT invent, guess, or hallucinate arguments

### What the MCP server does with arguments

| Tool | Args | Server behavior |
|------|------|-----------------|
| list_directory | path | Resolves relative to allowed_dir. Returns `[FILE]`/`[DIR]` listing. Fails with ENOENT if missing. |
| read_file | path | Resolves relative to allowed_dir. Returns raw file text. Fails with ENOENT if missing. |
| search_files | path, pattern | Recursive glob under path. Returns absolute paths. "No matches found" if empty. |
| write_file | path, content | Creates/overwrites file at path with content. Creates parent dirs. |
| create_directory | path | Creates directory (and parents). No-op if exists. |

**Key facts:**
- All paths are **relative** (no leading `/`). Absolute paths get "Access denied."
- Trailing slashes don't matter to the server (`docs` = `docs/`), but our eval does exact-match
- Glob patterns are standard Unix: `*`, `?`, `[abc]`. NOT regex. NOT recursive `**`.
- The server never crashes — bad paths return text errors (ENOENT, ENOTDIR)

---

## The Fundamental Rule

> **Every argument in the model's output must be extractable or directly inferable from the user's query.**

The user tells the model what to do. The model parses it. The model does not make things up.

### Three categories of arguments

| Category | Definition | Valid for training? |
|----------|-----------|-------------------|
| **EXTRACTABLE** | Argument is explicitly stated in the query. "read README.md" → path=README.md | YES — this is the primary case |
| **INFERABLE** | Argument follows from obvious convention. "show me the config" → path=config.json (or config.yaml) | YES — but only with common conventions |
| **INVENTED** | Argument is arbitrary, cannot be determined from query. "show me the audit report" → path=archaeology/admin/excavation_permit.md | NO — this tests memorization, not intelligence |

---

## Dataset Generation Rules

### Rule 1: Path must come from the query

The user query must contain enough information to determine the path.

**Good examples:**
```
"read README.md"                    → path="README.md"          (explicit)
"list files in src/"                → path="src/"               (explicit)
"what's in the tests directory"     → path="tests/"             (explicit)
"show me package.json"              → path="package.json"       (explicit)
"find Python files in lib"          → path="lib/", pattern="*.py" (explicit)
```

**Bad examples (BANNED):**
```
"show me the audit report"          → path="archaeology/admin/excavation_permit.md"   (invented)
"check the color target data"       → path="calibration/colorchecker_reference.csv"   (invented)
"pull up the article draft"         → path="journalism/drafts/tech_industry_expose.md" (invented)
```

The user doesn't say "archaeology/admin/excavation_permit.md" — the data generator invented that path. A real user who wants that file would say "read archaeology/admin/excavation_permit.md" or "show me the excavation permit in the archaeology admin folder."

### Rule 2: For write_file, content must be in the query

A 270M model cannot generate code, configs, or structured text from descriptions. The content arg must be provided verbatim in the query.

**Good examples:**
```
"write 'hello world' to output.txt"
→ {"tool":"write_file","args":{"path":"output.txt","content":"hello world"}}

"save this to config.env: DB_HOST=localhost\nDB_PORT=5432"
→ {"tool":"write_file","args":{"path":"config.env","content":"DB_HOST=localhost\nDB_PORT=5432"}}
```

**Bad examples (BANNED):**
```
"write a Hardhat deploy script to scripts/deploy.js"
→ content must be Solidity code? Model can't generate that.

"create a Docker Compose file at docker-compose.yml"
→ content must be full YAML? 270M model will hallucinate.
```

**Exception:** Very short, simple content that follows directly from the query is OK:
```
"write TODO to notes.txt"           → content="TODO"
"save 'fix: auth bug' to CHANGES"   → content="fix: auth bug"
```

### Rule 3: For write_file content, keep it short and extractable

Content should be:
- Explicitly quoted in the query
- Under ~100 characters
- Plain text, not code or structured formats

The model's job is to **extract and route**, not to **author**. If you want the model to write a Dockerfile, that's a code generation task — not an MCP tool routing task.

### Rule 4: Glob patterns must be unambiguous

The query must make the glob pattern obvious.

**Good examples:**
```
"find all .py files"                → pattern="*.py"            (clear)
"search for test files"             → pattern="test*" or "*test*" — AMBIGUOUS, avoid
"find files ending in .yaml"        → pattern="*.yaml"          (clear)
"search for anything with 'auth'"   → pattern="*auth*"          (clear)
```

**Bad examples:**
```
"find test stuff"                   → pattern=??? (test*, *test*, test_*, test*.py?)
"search for configs"                → pattern=??? (*.config, config*, *.cfg, *.yaml?)
```

When the pattern is ambiguous, either:
- Make the query more specific ("find files matching test_*.py")
- Use the most common convention and document it

### Rule 5: Normalize path conventions

| Convention | Rule | Example |
|-----------|------|---------|
| Directories | Always include trailing `/` | `src/`, not `src` |
| Files | Never include trailing `/` | `README.md`, not `README.md/` |
| Relative paths | No leading `/` or `./` | `src/`, not `./src/` or `/src/` |
| create_directory | Always trailing `/` | `utils/`, not `utils` |
| Nested paths | User must say the full path | "in the src components folder" → `src/components/` |

### Rule 6: Messy queries must still contain extractable args

Typos, slang, and voice transcription are fine for the QUERY — but the expected ANSWER must still be deterministic.

**Good messy example:**
```
"yo lemme see package dot json"     → path="package.json"       (extractable despite messy phrasing)
"reed the readme plz"               → path="README.md"          (common convention: readme → README.md)
"wats in src slash utils"           → path="src/utils/"          (extractable despite typos)
```

**Bad messy example:**
```
"show me that config thing"         → path=??? (which config? .env? config.json? settings.yaml?)
```

### Rule 7: Adversarial disambiguation must test routing, not args

Adversarial examples should confuse the TOOL choice, not the PATH.

**Good adversarial:**
```
"is there a tests folder?"          → list_directory (checking existence), not create_directory
"config.txt should have retries=3"  → write_file (updating content), not read_file
"I need a utils directory"          → create_directory (creating), not list_directory
```

**Bad adversarial:**
```
"show me the data"                  → ??? (which data? read what file? list what directory?)
```

The adversarial dimension is tool routing confusion. The path should still be obvious.

---

## Failure Breakdown (from early dataset)

| Failure type | Count | % | Root cause | Fix |
|-------------|-------|---|-----------|-----|
| Invented paths | ~60 | 14% | Eval has paths that can't be determined from query | Remove/regenerate |
| Extractable path failures | ~240 | 56% | Model fails to extract paths that ARE in the query | More extraction-focused training data |
| Inferable path failures | ~129 | 30% | Ambiguous queries with multiple valid answers | Tighten query specificity |
| Content generation (write_file) | 76 | 18% | Model can't generate code/configs from descriptions | Content must be in query |
| Content drift | 25 | 6% | Model starts right but degrades mid-sequence | Keep content short |
| Tool routing errors | 25 | 6% | Genuine routing confusion (excl. timeouts) | More disambiguation training |
| Timeouts/JSON errors | 18 | 4% | Long write_file content exceeds generation capacity | Keep content under 100 chars |

---

## Checklist for Data Generator Agents

Before generating each example, verify:

- [ ] Can you determine the EXACT path from the query alone? If not, rewrite the query.
- [ ] For write_file: is the content explicitly quoted in the query? If not, rewrite or shorten.
- [ ] For search_files: is the glob pattern unambiguous from the query?
- [ ] Is there exactly ONE correct tool for this query? If ambiguous, rewrite.
- [ ] Does the path use the correct trailing-slash convention?
- [ ] For messy queries: are the args still extractable despite the noise?
- [ ] For adversarial queries: does the query test tool routing, not path guessing?
