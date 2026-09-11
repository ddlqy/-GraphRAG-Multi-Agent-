"""LLM客户端测试"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLLMClientInit:
    """LLM客户端初始化测试"""

    def test_import(self):
        """测试导入"""
        from src.llm.client import LLMClient
        assert LLMClient is not None

    def test_missing_api_key_raises(self, monkeypatch):
        """测试缺少API Key时抛出异常"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        # 确保dotenv不会覆盖
        from src.llm.client import LLMClient
        with pytest.raises(ValueError, match="API Key"):
            LLMClient(provider="deepseek", api_key=None)

    def test_from_config(self):
        """测试从配置创建"""
        from src.llm.client import LLMClient
        from src.config import LLMConfig
        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="test-key",
            api_base="https://api.deepseek.com"
        )
        client = LLMClient.from_config(config)
        assert client.provider == "deepseek"
        assert client.model == "deepseek-chat"


class TestLLMClientInvoke:
    """LLM客户端调用测试"""

    def test_invoke_none_raises(self):
        """测试None输入"""
        from src.llm.client import LLMClient
        client = LLMClient(provider="deepseek", api_key="test-key")
        with pytest.raises(ValueError, match="None"):
            client.invoke(None)

    def test_invoke_empty_string_raises(self):
        """测试空字符串"""
        from src.llm.client import LLMClient
        client = LLMClient(provider="deepseek", api_key="test-key")
        with pytest.raises(ValueError, match="空"):
            client.invoke("")

    def test_invoke_empty_list_raises(self):
        """测试空列表"""
        from src.llm.client import LLMClient
        client = LLMClient(provider="deepseek", api_key="test-key")
        with pytest.raises(ValueError, match="空"):
            client.invoke([])

    def test_invoke_string_prompt(self):
        """测试字符串prompt调用（需要真实API Key）"""
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            pytest.skip("需要DEEPSEEK_API_KEY环境变量")

        from src.llm.client import LLMClient
        client = LLMClient(
            provider="deepseek",
            model="deepseek-chat",
            api_key=api_key,
        )
        result = client.invoke("你好，请用一句话回复")
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"LLM回复: {result}")

    def test_invoke_message_list(self):
        """测试消息列表调用（模拟GraphAgent的用法）"""
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            pytest.skip("需要DEEPSEEK_API_KEY环境变量")

        from src.llm.client import LLMClient
        from langchain_core.messages import SystemMessage, HumanMessage

        client = LLMClient(
            provider="deepseek",
            model="deepseek-chat",
            api_key=api_key,
        )

        messages = [
            SystemMessage(content="你是一个校园问答助手"),
            HumanMessage(content="图书馆开放时间是？请用一句话回答"),
        ]
        result = client.invoke(messages)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"LLM回复: {result}")

    def test_invoke_with_chat_prompt_template(self):
        """测试ChatPromptTemplate.format_messages()的输出"""
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            pytest.skip("需要DEEPSEEK_API_KEY环境变量")

        from src.llm.client import LLMClient
        try:
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError:
            from langchain.prompts import ChatPromptTemplate

        client = LLMClient(
            provider="deepseek",
            model="deepseek-chat",
            api_key=api_key,
        )

        # 模拟GraphAgent中的prompt构建方式
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有帮助的校园AI助手"),
            ("human", "问题：{question}\n{scratchpad}"),
        ])
        messages = prompt.format_messages(
            question="图书馆开放时间？",
            scratchpad=""
        )

        result = client.invoke(messages)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"LLM回复: {result}")


class TestLLMClientRetry:
    """重试逻辑测试"""

    def test_retry_config(self):
        """测试重试参数配置"""
        from src.llm.client import LLMClient
        client = LLMClient(
            provider="deepseek",
            api_key="test-key",
            max_retries=5,
            retry_delay=2.0
        )
        assert client.max_retries == 5
        assert client.retry_delay == 2.0
