from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from coding_assistant.agents.architect_agent import ArchitectAgent
from coding_assistant.agents.dev_agent import DevAgent
from coding_assistant.agents.pm_agent import PMAgent
from coding_assistant.agents.qa_agent import QAAgent
from coding_assistant.agents.registry import create_default_registry
from coding_assistant.agents.reviewer_agent import ReviewerAgent
from coding_assistant.core.host import Host, HostAction, HostDecision
from coding_assistant.core.pipeline import PHASE_AGENT_MAP, PipelinePhase
from coding_assistant.core.types import AgentRole
from coding_assistant.core.workspace_manager import WorkspaceManager
from coding_assistant.llm.client import LLMClient
from coding_assistant.llm.config import prompt_api_key_interactive, resolve_api_key
from coding_assistant.tools.code_executor import ShellTool
from coding_assistant.tools.file_system import FileSystemTool
from coding_assistant.tools.git_operations import GitTool

console = Console()

AGENT_COLORS: dict[AgentRole, str] = {
    AgentRole.PM: "cyan",
    AgentRole.ARCHITECT: "magenta",
    AgentRole.DEV: "green",
    AgentRole.REVIEWER: "yellow",
    AgentRole.QA: "blue",
    AgentRole.PMGR: "white",
}


def _ensure_api_key() -> str:
    api_key = resolve_api_key()
    if api_key:
        return api_key
    console.print("[yellow]OpenAI API key not found.[/yellow]")
    api_key = prompt_api_key_interactive()
    if not api_key:
        console.print("[red]API key required. Exiting.[/red]")
        sys.exit(1)
    return api_key


def _create_project_dir(project_name: str, base_dir: Path) -> Path:
    project_dir = base_dir / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def _agent_label(role: AgentRole) -> Text:
    color = AGENT_COLORS.get(role, "white")
    return Text(f"[{role.value.upper()}]", style=f"bold {color}")


class CodingAssistantSession:
    def __init__(
        self,
        project_dir: Path,
        project_name: str,
        llm_client: LLMClient,
        is_iteration: bool = False,
    ) -> None:
        self.project_dir = project_dir
        self.project_name = project_name
        self.llm_client = llm_client
        self.is_iteration = is_iteration

        self.workspace_manager = WorkspaceManager(project_dir)
        self.fs_tool = FileSystemTool(project_dir)
        self.shell_tool = ShellTool(project_dir)
        self.registry = create_default_registry(
            llm_client=llm_client,
            project_dir=project_dir,
            fs_tool=self.fs_tool,
            shell_tool=self.shell_tool,
        )
        self.git_tool = GitTool(project_dir)

        if is_iteration:
            self.workspace_manager.load()
        else:
            self.workspace_manager.create(project_name)
            self.git_tool.init()

        self.host = Host(
            registry=self.registry,
            workspace_manager=self.workspace_manager,
            max_retries=3,
            checkpoint_callback=self._handle_checkpoint,
        )

        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        self._interrupted = True
        console.print("\n[yellow]Interrupted. Workspace saved. Exiting.[/yellow]")
        sys.exit(0)

    async def _handle_checkpoint(self, phase: PipelinePhase, workspace: Any) -> str | None:
        console.print()
        console.rule(f"[bold]Checkpoint: {phase.value}[/bold]")

        partition = workspace.get_partition(PHASE_AGENT_MAP.get(phase, AgentRole.PM))
        console.print(Panel(str(partition), title=f"{phase.value} output"))

        console.print("[bold]Do you approve? Provide feedback or type 'ok' to continue:[/bold]")
        try:
            feedback = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if feedback.lower() in ("ok", "yes", "y", ""):
            return None
        return feedback

    async def run(self, initial_input: str) -> None:
        console.print(
            Panel(
                f"[bold]Coding Assistant[/bold]\nProject: {self.project_name}",
                style="bold blue",
            )
        )

        workspace = self.workspace_manager.workspace
        workspace.progress.current_phase = self.host.state.current_phase.value

        if self.is_iteration:
            self.host.state.iteration_count += 1
            console.print(f"[dim]Iteration {self.host.state.iteration_count}[/dim]")
            parsed_input = self._parse_iteration_input(initial_input)
        else:
            parsed_input = initial_input

        agent = self.host.get_current_agent()
        agent.model = self.llm_client.get_model(agent.role)

        console.print(_agent_label(agent.role), "Analyzing requirements...")
        handoff = await self._dispatch_agent(agent, parsed_input)

        architect_feedback: str | None = None

        while True:
            if self._interrupted:
                break

            decision = await self.host.handle_handoff(handoff)
            self._display_decision(decision)

            if decision.action == HostAction.COMPLETE:
                console.print("[bold green]Pipeline complete![/bold green]")
                if self.is_iteration:
                    commit_msg = (
                        f"Iteration {self.host.state.iteration_count}: "
                        f"feature update after pipeline completion"
                    )
                    self.git_tool.commit(commit_msg)
                break

            if decision.action == HostAction.CHECKPOINT:
                if decision.phase:
                    feedback = await self.host.run_checkpoint(decision.phase)
                    if feedback and decision.phase == PipelinePhase.ARCHITECTURE:
                        architect_feedback = feedback
                    if decision.phase == PipelinePhase.GIT_COMMIT:
                        self._git_commit_at_checkpoint()
                continue

            if decision.action == HostAction.RETRY_DEV:
                agent = self.registry.get(AgentRole.DEV)
                agent.model = self.llm_client.get_model(agent.role)
                console.print(_agent_label(agent.role), "Retrying with fixes...")
                handoff = await self._dispatch_agent(
                    agent,
                    f"Previous attempt had issues: {handoff.summary}. Please fix.",
                    extra={"dev_feedback": handoff.summary},
                )
                continue

            if decision.action in (
                HostAction.ADVANCE,
                HostAction.CHECKPOINT_THEN_ADVANCE,
            ):
                if decision.next_phase:
                    self.host.advance_to(decision.next_phase)
                agent = self.host.get_current_agent()
                agent.model = self.llm_client.get_model(agent.role)
                console.print(
                    _agent_label(agent.role),
                    f"Starting {decision.next_phase.value} phase...",
                )
                handoff = await self._dispatch_agent(
                    agent,
                    self._build_agent_input(agent.role),
                    extra={
                        "architect_feedback": architect_feedback,
                        "is_documentation": decision.next_phase == PipelinePhase.DOCUMENTATION,
                    },
                )
                continue

            if decision.action == HostAction.FORCE_HANDOFF:
                console.print("[yellow]Agent did not complete. Moving on.[/yellow]")
                if decision.next_phase:
                    self.host.advance_to(decision.next_phase)
                    agent = self.host.get_current_agent()
                    handoff = await self._dispatch_agent(
                        agent,
                        self._build_agent_input(agent.role),
                    )
                else:
                    break
                continue

            break

        self.workspace_manager.save()

    async def _dispatch_agent(
        self, agent: Any, input_text: str, extra: dict[str, Any] | None = None
    ) -> Any:
        extra = extra or {}
        if isinstance(agent, PMAgent):
            return await agent.analyze_requirements(input_text, self.workspace_manager.workspace)
        if isinstance(agent, ArchitectAgent):
            return await agent.design_architecture(
                self.workspace_manager.workspace,
                user_feedback=extra.get("architect_feedback"),
            )
        if isinstance(agent, DevAgent):
            if extra.get("is_documentation"):
                return await agent.generate_documentation(self.workspace_manager.workspace)
            return await agent.implement_code(
                self.workspace_manager.workspace,
                feedback=extra.get("dev_feedback"),
            )
        if isinstance(agent, ReviewerAgent):
            return await agent.review_code(self.workspace_manager.workspace)
        if isinstance(agent, QAAgent):
            return await agent.test_code(self.workspace_manager.workspace)
        return await agent.run(input_text)

    def _build_agent_input(self, role: AgentRole) -> str:
        ws = self.workspace_manager.workspace
        if role == AgentRole.ARCHITECT:
            return f"Design architecture based on these requirements:\n{ws.requirements.prd}"
        if role == AgentRole.DEV:
            if self.host.state.current_phase == PipelinePhase.DOCUMENTATION:
                return "Generate project documentation"
            return (
                f"Implement code based on this architecture:\n"
                f"{ws.architecture.tech_stack}\n{ws.architecture.project_structure}"
            )
        if role == AgentRole.REVIEWER:
            files = [f.path for f in ws.code.files]
            return f"Review these files: {', '.join(files)}"
        if role == AgentRole.QA:
            files = [f.path for f in ws.code.files]
            return f"Write and run tests for: {', '.join(files)}"
        return "Continue with your role."

    def _parse_iteration_input(self, user_input: str) -> str:
        stripped = user_input.strip()
        if stripped.startswith("/add-feature"):
            feature_desc = stripped[len("/add-feature") :].strip()
            return (
                f"[ITERATION] Add new feature: {feature_desc or 'as described'}. "
                "Preserve all existing functionality and extend the codebase."
            )
        if stripped.startswith("/modify"):
            modify_desc = stripped[len("/modify") :].strip()
            return (
                f"[ITERATION] Modify existing feature: {modify_desc or 'as described'}. "
                "Update the relevant code without breaking existing functionality."
            )
        if stripped.startswith("/fix"):
            fix_desc = stripped[len("/fix") :].strip()
            return (
                f"[ITERATION] Fix bug: {fix_desc or 'as described'}. "
                "Fix the issue while maintaining backward compatibility."
            )
        return f"[ITERATION] New requirement: {stripped}. Preserve existing functionality."

    def _git_commit_at_checkpoint(self) -> None:
        phase = self.host.state.current_phase.value
        iteration = self.host.state.iteration_count
        if iteration > 0:
            msg = f"Checkpoint: {phase} (iteration {iteration})"
        else:
            msg = f"Checkpoint: {phase}"
        self.git_tool.commit(msg)

    def _display_decision(self, decision: HostDecision) -> None:
        console.print(f"[dim]{decision.action.value}: {decision.reason}[/dim]")


@click.group()
def cli() -> None:
    """Multi-agent coding assistant."""


@cli.command()
@click.argument("project_name")
@click.option("--base-dir", default=".", help="Base directory for the project")
def new(project_name: str, base_dir: str) -> None:
    """Create a new project."""
    import asyncio

    api_key = _ensure_api_key()
    project_dir = _create_project_dir(project_name, Path(base_dir))
    llm_client = LLMClient(api_key=api_key)

    session = CodingAssistantSession(
        project_dir=project_dir,
        project_name=project_name,
        llm_client=llm_client,
        is_iteration=False,
    )

    console.print("[bold]Describe your project requirement:[/bold]")
    try:
        requirement = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if requirement:
        asyncio.run(session.run(requirement))


@cli.command("iter")
@click.argument("project_name")
@click.option("--base-dir", default=".", help="Base directory for the project")
def iter_project(project_name: str, base_dir: str) -> None:
    """Iterate on an existing project."""
    import asyncio

    api_key = _ensure_api_key()
    project_dir = Path(base_dir) / project_name

    if not project_dir.exists():
        console.print(f"[red]Project directory not found: {project_dir}[/red]")
        sys.exit(1)

    llm_client = LLMClient(api_key=api_key)

    session = CodingAssistantSession(
        project_dir=project_dir,
        project_name=project_name,
        llm_client=llm_client,
        is_iteration=True,
    )

    console.print("[bold]What would you like to add or change?[/bold]")
    console.print("[dim]Commands: /add-feature <desc>, /modify <desc>, /fix <desc>[/dim]")
    console.print("[dim]Type 'exit' or 'quit' to end the session.[/dim]")

    while True:
        try:
            requirement = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not requirement:
            continue

        if requirement.lower() in ("exit", "quit", "q"):
            console.print("[bold]Session ended.[/bold]")
            break

        asyncio.run(session.run(requirement))

        console.print("\n[bold]Anything else?[/bold]")


if __name__ == "__main__":
    cli()
