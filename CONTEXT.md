# Domain Glossary

## Core Concepts

- **Coding Assistant**: The system itself — a multi-agent tool that transforms natural language requirements into runnable code projects through agent collaboration.
- **Session**: A continuous conversation between the user and the Coding Assistant. A session may span multiple iterations of a project. The session persists until the user exits.
- **Project**: A codebase being built or iterated on by the Coding Assistant. A project has a directory on disk, a Git repository, and associated metadata.

## Agent Roles

- **Agent**: An autonomous role with a specific responsibility, powered by an LLM with a dedicated system prompt and tool access.
- **PM Agent** (Product Manager): Analyzes natural language requirements, produces structured requirement documents, and prioritizes features.
- **Architect Agent**: Designs technical architecture, selects technology stack, defines API contracts and database schemas.
- **Dev Agent** (Developer): Generates and modifies source code, configuration files, and database scripts.
- **Reviewer Agent**: Audits code for quality, security, and convention compliance. Produces review reports with improvement suggestions.
- **QA Agent** (Quality Assurance): Generates test cases, executes tests, and produces test reports with coverage metrics.
- **PMgr Agent** (Project Manager): Orchestrates agent scheduling, tracks progress, and manages milestones. Acts as the **Host** of the session.

## Orchestration

- **Host**: The PMgr Agent in its scheduling role. The Host decides which Agent speaks next based on the current task state, and manages handoffs, retries, and flow control.
- **Handoff**: The act of one Agent completing its turn and the Host selecting the next Agent to act. An Agent signals completion by calling a `handoff` tool, whose parameters include task status, output summary, and suggested next step. The Host then decides which Agent acts next.
- **Phase**: The basic scheduling unit for the Host. Each Agent is dispatched for one phase (e.g., requirements analysis, architecture design, code implementation). Within a phase, the active Agent may produce multiple outputs before handing off. The Dev Agent implements all sub-features in a single phase before handing off to Reviewer and QA.
- **Retry Policy**: When Reviewer or QA finds issues, the Host decides the response based on severity. **Minor** issues (code style, simple bugs) trigger an automatic re-dispatch to the Dev Agent. **Critical** issues (architectural flaws, security vulnerabilities) trigger a Checkpoint for human review. Severity is carried in the Handoff tool's `severity` field (`minor` / `major` / `critical`). The maximum retry count is configurable (default: 3). If consecutive Handoff summaries are highly similar (indicating the Dev Agent is stuck), the Host escalates to a Checkpoint before reaching the retry limit.
- **Checkpoint**: A point in the workflow where human confirmation is required before proceeding. At a Checkpoint, the Host presents the current artifact to the user and asks for feedback in natural language. The user may approve, request modifications, or provide additional direction. The Host then decides whether to continue or route back to the relevant Agent.

## State & Context

- **Workspace**: The shared state object that all Agents read from and write to. It is the single source of truth for project knowledge — requirements, architecture decisions, code artifacts, test results, and review reports. Each Agent has its own conversation history but communicates outcomes through the Workspace.
- **Workspace Partition**: A named section of the Workspace owned by a specific Agent role. Partitions include: Requirements, Architecture, Code (file index only), Review, Test, and Progress. Code itself lives on the file system; the Workspace stores only metadata and file path references.
- **Workspace Persistence**: The Workspace is serialized as a single JSON file (`.coding-assistant/workspace.json`) stored in the project root directory. It includes a `version` field for backward-compatible migration across tool versions. It is loaded on project iteration and updated after each phase.
- **Workspace Write Policy**: The Workspace is written to disk after every significant change (not only on Handoff). On user interrupt (Ctrl+C), the latest Workspace state is already persisted, allowing recovery from the last saved point on next iteration.

## Workflow

- **Default Pipeline**: The Host follows a fixed pipeline: PM → Architect → Dev → Reviewer → QA → Documentation → Git Commit. Reviewer always runs before QA to catch code issues early and avoid unnecessary test runs.
- **Checkpoint Positions**: By default, Checkpoints are placed after PM (requirements confirmation) and Architect (architecture confirmation). Additional Checkpoints are triggered on critical issues from Reviewer or QA, and before version release.
- **Git Commit Strategy**: Git commits are made only at confirmed Checkpoints — after requirements confirmation, architecture confirmation, and final delivery. This ensures each commit represents a verified stable state.

## User Interaction

- **SDK**: The core Python library that provides programmatic access to the Coding Assistant's capabilities.
- **CLI**: A terminal-based user interface that wraps the SDK, providing interactive access to the Coding Assistant. Launch with `coding-assistant <project-name>` to create a new project, or `coding-assistant --iter <project-name>` to iterate on an existing project.
- **Iteration Trigger**: A new iteration can be initiated by natural language (e.g., "add payment feature") or explicit command (e.g., `/add-feature payment`). The PMgr Agent identifies new requirements and re-dispatches the PM Agent.
- **Conversation Memory**: Each Agent maintains its own conversation history within a phase. Upon Handoff, a summary of the Agent's conversation is generated and stored in the Workspace. New phase Agents receive the summary plus the current Workspace state — not the full conversation history of prior Agents.

## LLM Configuration

- **Model Assignment**: Each Agent role can be configured with a specific LLM model. Default tiering: PM, Architect, and Dev Agents use GPT-4o (deep reasoning required); Reviewer, QA, and PMgr Agents use GPT-4o-mini (execution and scheduling tasks). Users can override model assignments per role via configuration.
- **API Key Management**: OpenAI API key is resolved from environment variable (`OPENAI_API_KEY`) first, then from config file (`~/.coding-assistant/config.yaml`). If neither exists, the CLI guides the user through interactive setup on first launch.

## Technology Scope

- **MVP Target Stack**: Python backend only (FastAPI, Django, Flask, etc.). The Architect Agent freely recommends the best Python framework and libraries based on the specific requirements, rather than selecting from preset templates. Frontend and other language support are out of scope for MVP.
