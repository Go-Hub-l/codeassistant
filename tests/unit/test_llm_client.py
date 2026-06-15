from coding_assistant.core.types import AgentRole
from coding_assistant.llm.client import DEFAULT_MODELS, LLMClient


class TestLLMClientDefaults:
    def test_default_models_for_each_role(self):
        assert DEFAULT_MODELS[AgentRole.PM] == "deepseek-v4-pro"
        assert DEFAULT_MODELS[AgentRole.ARCHITECT] == "deepseek-v4-pro"
        assert DEFAULT_MODELS[AgentRole.DEV] == "deepseek-v4-pro"
        assert DEFAULT_MODELS[AgentRole.REVIEWER] == "deepseek-v4-pro"
        assert DEFAULT_MODELS[AgentRole.QA] == "deepseek-v4-pro"
        assert DEFAULT_MODELS[AgentRole.PMGR] == "deepseek-v4-pro"

    def test_get_model_default(self):
        client = LLMClient(api_key="test-key")
        assert client.get_model(AgentRole.PM) == "deepseek-v4-pro"
        assert client.get_model(AgentRole.REVIEWER) == "deepseek-v4-pro"

    def test_get_model_override(self):
        client = LLMClient(
            api_key="test-key",
            model_overrides={AgentRole.PM: "deepseek-chat"},
        )
        assert client.get_model(AgentRole.PM) == "deepseek-chat"
        assert client.get_model(AgentRole.DEV) == "deepseek-v4-pro"

    def test_token_usage_starts_zero(self):
        client = LLMClient(api_key="test-key")
        usage = client.get_token_usage()
        assert usage["prompt"] == 0
        assert usage["completion"] == 0
        assert usage["total"] == 0
