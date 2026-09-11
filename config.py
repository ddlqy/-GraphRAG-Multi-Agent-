"""全局配置"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "deepseek"  # deepseek / openai / ollama
    model: str = "deepseek-chat"
    api_key: str = ""
    api_base: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class EmbeddingConfig:
    """Embedding配置"""
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    use_gpu: bool = False


@dataclass
class GraphConfig:
    """知识图谱配置"""
    graph_dir: str = str(PROJECT_ROOT / "data" / "graph" / "graph.json")
    node_text_keys: dict = field(default_factory=lambda: {
        "Policy": ["name", "description"],
        "Scholarship": ["name", "amount", "conditions"],
        "Course": ["name", "code", "credits"],
        "Location": ["name", "address", "hours"],
        "Department": ["name", "contact"],
        "Procedure": ["name", "steps"],
        "Competition": ["name", "level", "category"],
        "Condition": ["name", "requirement"],
    })


@dataclass
class RetrieverConfig:
    """检索器配置"""
    embed_cache: bool = True
    embed_cache_dir: str = str(PROJECT_ROOT / "data" / "graph" / "cache")
    faiss_gpu: bool = False
    top_k: int = 10


@dataclass
class AgentConfig:
    """Agent配置"""
    max_steps: int = 10
    max_reflect: int = 3
    reflexion_strategy: str = "Last_attempt_and_Reflexion"  # None / Last_attempt / Reflexion / Last_attempt_and_Reflexion
    compound_strategy: str = "plan_compound"  # None / compound / plan / plan_compound
    llm_version: str = "deepseek-chat"
    reflect_version: str = "deepseek-chat"


@dataclass
class AppConfig:
    """应用配置"""
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = True


@dataclass
class Config:
    """主配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    app: AppConfig = field(default_factory=AppConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置"""
        config = cls()

        # LLM配置 — 优先读 DEEPSEEK_API_KEY，兼容 LLM_API_KEY
        config.llm.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY", "")
        config.llm.api_base = os.getenv("DEEPSEEK_API_BASE") or os.getenv("LLM_API_BASE", config.llm.api_base)
        config.llm.model = os.getenv("LLM_MODEL", config.llm.model)
        config.llm.provider = os.getenv("LLM_PROVIDER", config.llm.provider)

        # 温度和token数
        if os.getenv("LLM_TEMPERATURE"):
            try:
                config.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
            except ValueError:
                pass
        if os.getenv("LLM_MAX_TOKENS"):
            try:
                config.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
            except ValueError:
                pass

        return config


# 全局配置实例
config = Config.from_env()
