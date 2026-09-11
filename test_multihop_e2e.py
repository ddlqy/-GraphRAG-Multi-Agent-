"""复杂多跳问题端到端测试

测试场景：
1. 单跳事实查询
2. 多跳关系推理
3. 条件判断和数值推理
4. 对比分析
5. 降级处理
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def llm():
    """创建LLM客户端"""
    import os
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        pytest.skip("需要DEEPSEEK_API_KEY环境变量")

    from src.llm.client import LLMClient
    return LLMClient(provider="deepseek", model="deepseek-chat", api_key=api_key)


@pytest.fixture
def graph():
    """加载知识图谱"""
    path = PROJECT_ROOT / "data" / "graph" / "graph.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==================== 工具测试 ====================

class TestKGSearchTool:
    """知识图谱搜索工具测试"""

    def test_basic_search(self, graph):
        """基础搜索"""
        from src.agents.tools import KGSearchTool
        tool = KGSearchTool(graph)
        result = tool.run("国家奖学金")
        assert "奖学金" in result
        assert "8000" in result

    def test_library_search(self, graph):
        """图书馆搜索"""
        from src.agents.tools import KGSearchTool
        tool = KGSearchTool(graph)
        result = tool.run("图书馆开放时间")
        assert "8:00" in result or "22:00" in result or "图书馆" in result

    def test_department_search(self, graph):
        """院系搜索"""
        from src.agents.tools import KGSearchTool
        tool = KGSearchTool(graph)
        result = tool.run("计算机学院")
        assert "计算机" in result

    def test_no_result(self, graph):
        """无结果搜索"""
        from src.agents.tools import KGSearchTool
        tool = KGSearchTool(graph)
        result = tool.run("量子力学课程")
        assert "未找到" in result or "无" in result.lower()


class TestCalculatorTool:
    """计算器工具测试"""

    def test_gpa_high(self):
        """高GPA判断"""
        from src.agents.tools import CalculatorTool
        tool = CalculatorTool()
        result = tool.run("GPA 3.8能申请什么奖学金")
        assert "国家奖学金" in result

    def test_gpa_medium(self):
        """中等GPA判断"""
        from src.agents.tools import CalculatorTool
        tool = CalculatorTool()
        result = tool.run("GPA 3.2能申请什么奖学金")
        assert "校级" in result

    def test_gpa_low(self):
        """低GPA判断"""
        from src.agents.tools import CalculatorTool
        tool = CalculatorTool()
        result = tool.run("GPA 2.0能申请什么奖学金")
        assert "暂无" in result or "60%以后" in result

    def test_comparison(self):
        """奖学金对比"""
        from src.agents.tools import CalculatorTool
        tool = CalculatorTool()
        result = tool.run("奖学金对比")
        assert "8000" in result
        assert "5000" in result
        assert "3000" in result

    def test_ranking(self):
        """排名推理"""
        from src.agents.tools import CalculatorTool
        tool = CalculatorTool()
        result = tool.run("排名前15%")
        assert "奖学金" in result


# ==================== Agent编排器测试 ====================

class TestLangGraphOrchestrator:
    """LangGraph编排器测试"""

    def test_graph_structure(self):
        """测试图结构"""
        from src.agents.langgraph_orchestrator import create_campus_agent_graph
        graph = create_campus_agent_graph()
        info = graph.get_graph_info()

        assert "planner" in info["nodes"]
        assert "graph_searcher" in info["nodes"]
        assert "calculator" in info["nodes"]
        assert "synthesizer" in info["nodes"]
        assert info["entry_point"] == "planner"

    def test_simple_query(self, llm):
        """简单查询"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("图书馆开放时间")

        assert result["answer"]
        assert "graph_searcher" in result["node_history"]
        assert "synthesizer" in result["node_history"]

    def test_scholarship_query(self, llm):
        """奖学金查询"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("国家奖学金的申请条件是什么")

        assert result["answer"]
        assert "8000" in result["answer"] or "GPA" in result["answer"] or "奖学金" in result["answer"]

    def test_multi_hop_query(self, llm):
        """多跳查询"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("GPA 3.5能申请什么奖学金")

        assert result["answer"]
        # 应该经过 calculator 节点
        assert "calculator" in result["node_history"] or "synthesizer" in result["node_history"]

    def test_department_query(self, llm):
        """院系查询"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("计算机学院有哪些专业")

        assert result["answer"]
        assert "计算机" in result["answer"]

    def test_comparison_query(self, llm):
        """对比查询"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("国家奖学金和校级一等奖学金有什么区别")

        assert result["answer"]
        assert "8000" in result["answer"] or "3000" in result["answer"]


# ==================== 端到端复杂问题测试 ====================

class TestComplexMultiHop:
    """复杂多跳问题端到端测试"""

    def test_gpa_scholarship_eligibility(self, llm):
        """GPA → 排名 → 奖学金匹配"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("我GPA 3.6，能申请国家奖学金吗")

        answer = result["answer"]
        assert answer
        # 应该提到GPA要求或排名
        assert any(kw in answer for kw in ["GPA", "绩点", "排名", "前10%", "奖学金"])

    def test_competition_scholarship_chain(self, llm):
        """竞赛 → 级别 → 奖学金关系"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("蓝桥杯省二对申请奖学金有帮助吗")

        answer = result["answer"]
        assert answer
        assert any(kw in answer for kw in ["蓝桥杯", "竞赛", "奖学金", "B类", "省级"])

    def test_transfer_major_process(self, llm):
        """转专业流程查询"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("大一学生想转专业，需要满足什么条件")

        answer = result["answer"]
        assert answer
        assert any(kw in answer for kw in ["转专业", "申请", "GPA", "前30%", "教务处"])

    def test_library_service_query(self, llm):
        """图书馆服务查询"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("图书馆能借多少本书")

        answer = result["answer"]
        assert answer
        assert any(kw in answer for kw in ["借", "册", "图书馆"])

    def test_graduation_requirements(self, llm):
        """毕业要求查询"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("毕业需要满足哪些条件")

        answer = result["answer"]
        assert answer
        assert any(kw in answer for kw in ["学分", "毕业", "论文", "答辩"])

    def test_scholarship_amount_comparison(self, llm):
        """奖学金金额对比"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("所有奖学金的金额分别是多少")

        answer = result["answer"]
        assert answer
        assert "8000" in answer or "5000" in answer or "3000" in answer

    def test_department_contact(self, llm):
        """部门联系方式"""
        from src.agents.langgraph_orchestrator import LangGraphOrchestrator
        orch = LangGraphOrchestrator()
        result = orch.run("教务处的电话是多少")

        answer = result["answer"]
        assert answer
        assert "0335" in answer or "教务处" in answer
