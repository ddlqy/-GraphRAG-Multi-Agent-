"""测试LangGraph状态图编排器"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state_graph import StateGraph, AgentState, START, END


class TestStateGraph:
    """状态图测试"""
    
    def test_create_graph(self):
        """测试创建状态图"""
        graph = StateGraph(AgentState)
        assert graph is not None
        assert graph.entry_point is None
    
    def test_add_node(self):
        """测试添加节点"""
        graph = StateGraph(AgentState)
        
        def test_func(state):
            return {"step_count": state.step_count + 1}
        
        graph.add_node("test_node", test_func)
        assert "test_node" in graph.nodes
    
    def test_add_edge(self):
        """测试添加边"""
        graph = StateGraph(AgentState)
        
        def func1(state):
            return {}
        
        def func2(state):
            return {}
        
        graph.add_node("node1", func1)
        graph.add_node("node2", func2)
        graph.add_edge("node1", "node2")
        
        assert len(graph.edges) == 1
        assert graph.edges[0].source == "node1"
        assert graph.edges[0].target == "node2"
    
    def test_compile(self):
        """测试编译图"""
        graph = StateGraph(AgentState)
        
        def start_func(state):
            return {"thoughts": ["started"]}
        
        def end_func(state):
            return {"final_answer": "done", "is_finished": True}
        
        graph.add_node("start", start_func)
        graph.add_node("end", end_func)
        graph.set_entry_point("start")
        graph.add_edge("start", "end")
        graph.set_finish_point("end")
        
        compiled = graph.compile()
        assert compiled is not None
        assert compiled.entry_point == "start"
    
    def test_invoke(self):
        """测试执行图"""
        graph = StateGraph(AgentState)
        
        def double_step(state):
            return {"step_count": state.step_count + 1}
        
        def finish(state):
            return {"final_answer": "test answer", "is_finished": True}
        
        graph.add_node("step", double_step)
        graph.add_node("finish", finish)
        graph.set_entry_point("step")
        graph.add_edge("step", "finish")
        graph.set_finish_point("finish")
        
        compiled = graph.compile()
        result = compiled.invoke(AgentState(question="test"))
        
        assert result.is_finished == True
        assert result.final_answer == "test answer"
        assert result.step_count == 2  # 执行了step和finish两个节点
    
    def test_conditional_edges(self):
        """测试条件边"""
        graph = StateGraph(AgentState)
        
        def router(state):
            if "A" in state.question:
                return "node_a"
            return "node_b"
        
        def node_a(state):
            return {"final_answer": "A", "is_finished": True}
        
        def node_b(state):
            return {"final_answer": "B", "is_finished": True}
        
        graph.add_node("node_a", node_a)
        graph.add_node("node_b", node_b)
        graph.set_entry_point("node_a")  # 简化测试
        
        graph.add_conditional_edges("node_a", router, {"node_a": "node_a", "node_b": "node_b"})
        
        compiled = graph.compile()
        result = compiled.invoke(AgentState(question="test A"))
        assert result.final_answer == "A"
    
    def test_max_steps(self):
        """测试最大步数限制"""
        graph = StateGraph(AgentState)
        
        def infinite_loop(state):
            return {"step_count": state.step_count + 1}
        
        graph.add_node("loop", infinite_loop)
        graph.add_edge("loop", "loop")  # 无限循环
        graph.set_entry_point("loop")
        
        compiled = graph.compile()
        result = compiled.invoke(AgentState(question="test"), max_steps=5)
        
        assert result.step_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
