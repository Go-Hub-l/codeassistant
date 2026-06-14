import pytest

from coding_assistant.core.types import AgentRole
from coding_assistant.llm.templates import PromptTemplate, PromptTemplateManager


class TestPromptTemplate:
    def test_render_with_variables(self):
        template = PromptTemplate("Hello {name}, your role is {role}.")
        result = template.render(name="Alice", role="PM")
        assert result == "Hello Alice, your role is PM."

    def test_render_with_default_variables(self):
        template = PromptTemplate("Hello {name}", variables={"name": "World"})
        result = template.render()
        assert result == "Hello World"

    def test_render_kwargs_override_defaults(self):
        template = PromptTemplate("Hello {name}", variables={"name": "World"})
        result = template.render(name="Alice")
        assert result == "Hello Alice"

    def test_render_missing_variable_raises(self):
        template = PromptTemplate("Hello {name}")
        with pytest.raises(KeyError):
            template.render()


class TestPromptTemplateManager:
    def test_has_all_six_roles(self):
        manager = PromptTemplateManager()
        for role in AgentRole:
            assert role.value in manager.list_roles()

    def test_get_template_for_role(self):
        manager = PromptTemplateManager()
        for role in AgentRole:
            template = manager.get(role)
            assert isinstance(template, PromptTemplate)

    def test_get_nonexistent_role_raises(self):
        manager = PromptTemplateManager()
        with pytest.raises(KeyError):
            manager.get("nonexistent")

    def test_render_pm_template(self):
        manager = PromptTemplateManager()
        result = manager.render(AgentRole.PM, context="test context")
        assert "Product Manager" in result
        assert "test context" in result

    def test_render_architect_template(self):
        manager = PromptTemplateManager()
        result = manager.render(
            AgentRole.ARCHITECT,
            context="test context",
            requirements="user auth system",
        )
        assert "Architect" in result
        assert "user auth system" in result

    def test_render_dev_template(self):
        manager = PromptTemplateManager()
        result = manager.render(
            AgentRole.DEV,
            context="test context",
            architecture="FastAPI + PostgreSQL",
        )
        assert "Developer" in result
        assert "FastAPI + PostgreSQL" in result

    def test_render_reviewer_template(self):
        manager = PromptTemplateManager()
        result = manager.render(
            AgentRole.REVIEWER,
            context="test context",
            code="def hello(): pass",
        )
        assert "Reviewer" in result

    def test_render_qa_template(self):
        manager = PromptTemplateManager()
        result = manager.render(
            AgentRole.QA,
            context="test context",
            code="def hello(): pass",
            architecture="FastAPI",
        )
        assert "QA" in result

    def test_render_pmgr_template(self):
        manager = PromptTemplateManager()
        result = manager.render(
            AgentRole.PMGR,
            context="test context",
            max_retries=3,
        )
        assert "Project Manager" in result

    def test_register_custom_template(self):
        manager = PromptTemplateManager()
        custom = PromptTemplate("Custom {task} prompt")
        manager.register(AgentRole.PM, custom)
        result = manager.render(AgentRole.PM, task="test")
        assert result == "Custom test prompt"

    def test_all_templates_mention_handoff(self):
        manager = PromptTemplateManager()
        for role in AgentRole:
            template = manager.get(role)
            assert (
                "handoff" in template.template.lower()
            ), f"Template for {role.value} missing handoff instruction"
