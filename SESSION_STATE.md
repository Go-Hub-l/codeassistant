# Session State — Multi-Agent Coding Assistant

> 此文件记录了当前开发会话的完整状态，用于跨会话恢复。下次启动 opencode 时，读取此文件即可恢复上下文。

## 项目信息

- **项目名称**: codeassistant (多Agent代码编程助手)
- **GitHub 仓库**: git@github.com:Go-Hub-l/codeassistant.git
- **技术栈**: Python 3.11+ / LangGraph / OpenAI / Rich+Click CLI
- **当前分支**: main
- **最近提交**: 73e623f — Implement Reviewer Agent (#10) and iteration mode (#15)

## 28 项核心设计决策

| # | 决策 | 选择 |
|---|------|------|
| 1 | 工具形态 | SDK 核心 + CLI 入口 |
| 2 | 对话边界 | 持续对话，不断迭代 |
| 3 | Agent 轮次 | PMgr 作为 Host 调度 |
| 4 | 上下文共享 | Workspace 对象，独立对话历史 |
| 5 | 代码存储 | Workspace 引用文件系统 |
| 6 | Checkpoint 交互 | 自然语言协商 |
| 7 | Handoff 机制 | 工具调用 |
| 8 | 任务粒度 | 按阶段调度，Dev 一次性实现全部子功能 |
| 9 | Agent 工具 | 最小权限，Reviewer 有代码执行能力 |
| 10 | 审查/测试失败 | 分级处理（minor 自动，critical 人工） |
| 11 | 迭代触发 | 自然语言 + 显式命令 |
| 12 | 对话记忆 | 摘要 + Workspace |
| 13 | LLM 模型 | 可配置，分级默认值 |
| 14 | Reviewer/QA 顺序 | 固定 Reviewer → QA |
| 15 | CLI 启动 | `coding-assistant <name>` / `--iter <name>` |
| 16 | Workspace 持久化 | JSON，项目根目录 `.coding-assistant/workspace.json` |
| 17 | MVP 技术栈 | Python 后端，Architect 自由推荐 |
| 18 | Git 提交 | 仅 Checkpoint 确认后 |
| 19 | 文档生成 | QA 通过后独立阶段 |
| 20 | LLM 失败 | 指数退避重试 + 失败后 Checkpoint |
| 21 | CLI 展示 | 实时流式 + 工具调用摘要 |
| 22 | API Key | 环境变量 → 配置文件 → 交互引导 |
| 23 | Agent 未调用 Handoff | 强制 Handoff + 标记 incomplete |
| 24 | 工具调用失败 | 分级：可恢复 Agent 自重试，不可恢复 Checkpoint |
| 25 | 用户中断 | Workspace 及时写盘，Ctrl+C 不丢进度 |
| 26 | 有害代码防御 | 工具层硬限制 + Reviewer 软审查 |
| 27 | Workspace 版本兼容 | version 字段 + 向后兼容迁移 |
| 28 | Agent 打转检测 | 连续摘要相似度 ≥ 0.85 → 提前升级 Checkpoint |

## GitHub Issues 进度

### 已完成 (13/16)
- **#1** PRD: Multi-Agent Coding Assistant — 已发布
- **#2** Agent 基类与注册机制 ✅ (`agents/base.py`, `agents/registry.py`)
- **#3** LLM 集成层 + Prompt 模板 ✅ (`llm/client.py`, `llm/templates.py`, `llm/config.py`)
- **#4** Workspace 对象与持久化 ✅ (`core/workspace.py`, `core/workspace_manager.py`)
- **#5** Handoff 工具与 Host 调度器 ✅ (`core/host.py`, `core/pipeline.py`)
- **#6** PM Agent 实现 ✅ (`agents/pm_agent.py`)
- **#7** Architect Agent 实现 ✅ (`agents/architect_agent.py`)
- **#8** 文件系统与代码执行工具 ✅ (`tools/file_system.py`, `tools/code_executor.py`)
- **#9** Dev Agent 实现 ✅ (`agents/dev_agent.py`)
- **#10** Reviewer Agent 实现 ✅ (`agents/reviewer_agent.py`)
- **#12** Git 集成 ✅ (`tools/git_operations.py`)
- **#13** CLI 交互界面 ✅ (`cli/main.py`)
- **#14** 错误边界处理 ✅ (`core/error_handler.py`, `core/workspace_manager.py`)
- **#15** 迭代模式支持 ✅ (`core/host.py`, `cli/main.py`)

### 待实施 (3/16)
- **#11** QA Agent — 依赖 #9 ✅ + #10 ✅
- **#16** 端到端集成测试 — 依赖 #11 + #13 ✅ + #15 ✅

## 项目目录结构

```
src/coding_assistant/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── architect_agent.py   # Architect Agent — 架构设计
│   ├── base.py              # Agent 基类、Handoff 工具
│   ├── dev_agent.py          # Dev Agent — 代码生成/文档
│   ├── pm_agent.py           # PM Agent — 需求分析
│   ├── registry.py          # AgentRegistry + create_default_registry
│   └── reviewer_agent.py    # Reviewer Agent — 代码审查/安全扫描
├── cli/
│   ├── __init__.py
│   └── main.py              # CLI 入口（new/iter 命令）
├── core/
│   ├── __init__.py
│   ├── error_handler.py    # ErrorHandler — 错误分类与恢复策略
│   ├── host.py              # Host 调度器、HostAction、HostDecision
│   ├── pipeline.py          # PipelinePhase、PHASE_ORDER、CHECKPOINT_PHASES
│   ├── types.py             # AgentRole、HandoffStatus、Severity、HandoffResult
│   ├── workspace.py         # Workspace 模型（6 分区）
│   └── workspace_manager.py # WorkspaceManager（create/load/save/migrate）
├── llm/
│   ├── __init__.py
│   ├── client.py            # AsyncOpenAI 封装、流式、重试
│   ├── config.py            # API Key 解析链
│   └── templates.py         # PromptTemplate + PromptTemplateManager
└── tools/
    ├── __init__.py
    ├── code_executor.py     # CodeExecutionTool (Docker) + ShellTool (白名单)
    ├── file_system.py       # FileSystemTool（路径遍历防护）
    └── git_operations.py    # GitTool（init/commit/status/log/branch）
```

## 测试状态

- **205 个单元测试全部通过**
- **ruff lint clean**

## 恢复指南

下次启动 opencode 后，执行以下步骤恢复上下文：

1. 读取 `SESSION_STATE.md`（本文件）
2. 读取 `CONTEXT.md`（领域术语表）
3. 读取 `AGENTS.md`（Agent skills 配置）
4. 读取 `docs/agents/` 下的配置文件
5. 查看 `git log` 确认最新提交
6. 查看 GitHub Issues 确认待办项
7. 继续实施下一个 Issue

### 优先实施顺序
1. #11 QA Agent
2. #16 E2E 集成测试
