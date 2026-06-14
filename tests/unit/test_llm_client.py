
from coding_assistant.core.types import AgentRole
from coding_assistant.llm.client import DEFAULT_MODELS, LLMClient


class TestLLMClientDefaults:
    def test_default_models_for_each_role(self):
        assert DEFAULT_MODELS[AgentRole.PM] == "gpt-4o"
        assert DEFAULT_MODELS[AgentRole.ARCHITECT] == "gpt-4o"
        assert DEFAULT_MODELS[AgentRole.DEV] == "gpt-4o"
        assert DEFAULT_MODELS[AgentRole.REVIEWER] == "gpt-4o-mini"
        assert DEFAULT_MODELS[AgentRole.QA] == "gpt-4o-mini"
        assert DEFAULT_MODELS[AgentRole.PMGR] == "gpt-4o-mini"

    def test_get_model_default(self):
        client = LLMClient(api_key="test-key")
        assert client.get_model(AgentRole.PM) == "gpt-4o"
        assert client.get_model(AgentRole.REVIEWER) == "gpt-4o-mini"

    def test_get_model_override(self):
        client = LLMClient(
            api_key="test-key",
            model_overrides={AgentRole.PM: "gpt-4o-mini"},
        )
        assert client.get_model(AgentRole.PM) == "gpt-4o-mini"
        assert client.get_model(AgentRole.DEV) == "gpt-4o"

    def test_token_usage_starts_zero(self):
        client = LLMClient(api_key="test-key")
        usage = client.get_token_usage()
        assert usage["prompt"] == 0
        assert usage["completion"] == 0
        assert usage["total"] == 0
