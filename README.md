# Coding Assistant

A multi-agent coding assistant that transforms natural language requirements into runnable Python projects. Six specialized AI agents collaborate through a shared workspace, guided by a host scheduler with human checkpoints at key decision points.

## How It Works

```
User: "Build a TODO API with user auth"
        │
        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   PM Agent   │───▶│  Architect   │───▶│  Dev Agent   │───▶│  Reviewer    │───▶│  QA Agent    │
│  produces    │    │  selects     │    │  writes all  │    │  audits code │    │  runs pytest │
│  PRD,        │    │  framework,  │    │  source code │    │  security,   │    │  coverage,   │
│  user stories│    │  DB schema,  │    │  to disk     │    │  static      │    │  classifies  │
│              │    │  API design  │    │              │    │  analysis    │    │  failures    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼                   ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────┐
  │                          Host Scheduler (PMgr Agent)                                     │
  │  • Routes handoffs between agents  • Enforces checkpoints  • Tracks progress            │
  │  • Retry policy (minor → auto-retry, critical → human review)  • Stuck detection        │
  └─────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                                   ┌──────────────────────────┐
                                   │  Documentation + Git     │
                                   │  README, API docs, etc.  │
                                   │  git commit at checkpoint│
                                   └──────────────────────────┘
```

**Checkpoint gates** (human confirmation) appear after PM and Architect phases, plus whenever a critical issue is detected. You review and approve the output at each gate before the pipeline continues.

## Installation

Requires Python 3.10 or later.

```bash
# Clone and install
git clone git@github.com:Go-Hub-l/codeassistant.git
cd codeassistant
pip install -e .

# Or install from PyPI (future)
# pip install coding-assistant
```

## Configuration

### OpenAI API Key

The assistant uses OpenAI models. Set your API key via **one** of:

```bash
# Option 1: Environment variable (recommended)
export OPENAI_API_KEY="sk-..."

# Option 2: Config file
mkdir -p ~/.coding-assistant
echo "api_key: sk-..." > ~/.coding-assistant/config.yaml
```

If neither is set, the CLI will prompt you interactively.

### Proxy Configuration

If you are behind a firewall and need a proxy to reach the OpenAI API:

```bash
# Option 1: Environment variable (recommended)
export HTTPS_PROXY=http://127.0.0.1:7890

# Option 2: Config file
echo "base_url: https://your-proxy.com/v1" >> ~/.coding-assistant/config.yaml
```

Supported proxy env vars: `HTTPS_PROXY`, `https_proxy`, `HTTP_PROXY`, `ALL_PROXY`.

### Default Models

| Agent                   | Model          | Reasoning                                     |
|-------------------------|----------------|-----------------------------------------------|
| PM, Architect, Dev      | `deepseek-v4-pro`       | Deep reasoning for requirements, design, code |
| Reviewer, QA, PMgr      | `deepseek-v4-pro`  | Faster, cheaper for analysis and orchestration|

Override via environment variables:
```bash
export CODING_ASSISTANT_MODEL_PM=deepseek-v4-pro
export CODING_ASSISTANT_MODEL_DEV=deepseek-v4-pro
```

## Usage

### New Project

```bash
coding-assistant new my-project
# You will be prompted: "Describe your project requirement:"
```

This creates a new project directory and runs the full pipeline from scratch.

### Iterate on an Existing Project

```bash
coding-assistant iter my-project
# You enter a continuous conversation loop
```

Add features, fix bugs, or modify existing code. Supports explicit commands:

| Command                | Example                                                     |
|-----------------------|-------------------------------------------------------------|
| `/add-feature <desc>` | `/add-feature add rate limiting to all endpoints`            |
| `/modify <desc>`      | `/modify change the database from SQLite to PostgreSQL`      |
| `/fix <desc>`         | `/fix the login endpoint returns 500 on empty username`      |
| `exit` / `quit`       | End the session                                              |

Type any natural language instruction directly — the system understands plain English requirements.

### Pipeline Phases

The default pipeline runs 7 phases:

| Phase             | Agent     | Output                                                          |
|-------------------|-----------|-----------------------------------------------------------------|
| **Requirements**  | PM        | PRD, user stories, feature list, acceptance criteria            |
| **Architecture**  | Architect | Tech stack, project structure, API contracts, DB schema, security|
| **Development**   | Dev       | Source code written to disk, Code partition updated             |
| **Review**        | Reviewer  | Code audit with severity (minor / major / critical)             |
| **Testing**       | QA        | Pytest test cases generated and executed, coverage report       |
| **Documentation** | Dev       | README.md, API.md, DATABASE.md, DEPLOYMENT.md, CHANGELOG.md     |
| **Git Commit**    | PMgr      | Commits all changes at confirmed checkpoint                     |

#### Review & Security Checks

The Reviewer agent performs automated checks:

| Tool   | Purpose                                |
|--------|----------------------------------------|
| ruff   | Code style and linting violations      |
| bandit | Security vulnerability scanning        |
| mypy   | Static type checking                   |

Plus 13 built-in security patterns: hardcoded secrets, `eval()`/`exec()`/`os.system()` calls, SQL injection patterns, unsafe pickle deserialization, bare except clauses, and more.

#### Retry and Escalation

- **Minor issues** (code style, formatting): Dev auto-retries up to 3 times
- **Critical issues** (security vulnerability, data loss risk): Pipeline pauses at checkpoint for human review
- **Stuck agent**: If consecutive handoff summaries are 85%+ similar, the system escalates to checkpoint

## Workspace

All project state is stored in `.coding-assistant/workspace.json`:

```
.coding-assistant/
└── workspace.json    # Full project state with 6 partitions
```

The workspace holds:

| Partition       | Contents                                                    |
|-----------------|-------------------------------------------------------------|
| `requirements`  | PRD, user stories, feature list                             |
| `architecture`  | Tech stack, project structure, API contracts, DB schema     |
| `code`          | File path references with descriptions                      |
| `review`        | Issues found with severity classification                   |
| `test`          | Test cases, execution results, coverage metrics             |
| `progress`      | Phase summaries, retry count, decisions log                 |

Workspace saves after every agent handoff — **Ctrl+C won't lose progress**.

### Corruption Recovery

If `workspace.json` becomes corrupt (e.g., write interrupted), the system:
1. Backs up the corrupt file as `workspace.corrupt.<timestamp>.json`
2. Raises `WorkspaceCorruptionError` with the backup path
3. Creates a fresh workspace on next run

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
# Unit tests
pytest tests/unit/

# E2E tests (mocked LLM, full pipeline)
pytest tests/e2e/ -m e2e

# All tests with coverage
pytest tests/ --cov=src/coding_assistant
```

### Lint and Type Check

```bash
ruff check src/ tests/
mypy src/
```

## Project Structure

```
src/coding_assistant/
├── agents/
│   ├── base.py              # Agent base class, conversation history, handoff tool
│   ├── registry.py          # AgentRegistry — manages agent lifecycle
│   ├── pm_agent.py           # PM Agent — requirements analysis
│   ├── architect_agent.py    # Architect Agent — technical design
│   ├── dev_agent.py          # Dev Agent — code generation + documentation
│   ├── reviewer_agent.py     # Reviewer Agent — code audit + security scan
│   └── qa_agent.py           # QA Agent — test generation + execution
├── cli/
│   └── main.py              # Click CLI (new, iter commands)
├── core/
│   ├── host.py              # Host scheduler — handoff routing, retry policy
│   ├── pipeline.py          # Phase definitions, ordering, checkpoints
│   ├── workspace.py         # Workspace model (6 data partitions)
│   ├── workspace_manager.py # Persistence, migration, corruption recovery
│   ├── error_handler.py     # Error classification, recovery strategies
│   └── types.py             # Enums: AgentRole, HandoffStatus, Severity
├── llm/
│   ├── client.py            # OpenAI client with streaming, retry, model selection
│   ├── config.py            # API key resolution (env → config → prompt)
│   └── templates.py         # Per-agent prompt templates
└── tools/
    ├── file_system.py       # Safe file I/O with path traversal protection
    ├── code_executor.py     # Docker container + local shell execution
    └── git_operations.py    # Git init, commit, status, log
```

## License

MIT
