"""测试图操作工具"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.graph_funcs import GraphFuncs


class TestGraphFuncs:
    """图操作工具测试"""
    
    def test_init(self, sample_graph):
        """测试初始化"""
        gf = GraphFuncs(sample_graph)
        assert gf.graph == sample_graph
        assert len(gf.graph_index) > 0
    
    def test_check_neighbours(self, sample_graph):
        """测试检查邻居"""
        gf = GraphFuncs(sample_graph)
        
        # 测试获取所有邻居
        result = gf.check_neighbours("scholarship_001")
        assert "MANAGED_BY" in result
        assert "REQUIRES" in result
        
        # 测试获取指定类型邻居
        result = gf.check_neighbours("scholarship_001", "MANAGED_BY")
        assert "dept_jwc" in result
    
    def test_check_nodes(self, sample_graph):
        """测试检查节点属性"""
        gf = GraphFuncs(sample_graph)
        
        # 测试获取所有属性
        result = gf.check_nodes("scholarship_001")
        assert "国家奖学金" in result
        
        # 测试获取指定属性
        result = gf.check_nodes("scholarship_001", "name")
        assert result == "国家奖学金"
        
        result = gf.check_nodes("scholarship_001", "amount")
        assert result == "8000元/年"
    
    def test_check_degree(self, sample_graph):
        """测试检查度数"""
        gf = GraphFuncs(sample_graph)
        
        result = gf.check_degree("scholarship_001", "REQUIRES")
        assert result == "2"
    
    def test_check_all_neighbour(self, sample_graph):
        """测试查找所有指向指定节点的邻居"""
        gf = GraphFuncs(sample_graph)
        
        result = gf.check_all_neighbour("dept_jwc")
        assert "scholarship_001" in result
        assert "scholarship_002" in result
    
    def test_get_node_type(self, sample_graph):
        """测试获取节点类型"""
        gf = GraphFuncs(sample_graph)
        
        assert gf.get_node_type("scholarship_001") == "Scholarship"
        assert gf.get_node_type("dept_jwc") == "Department"
        assert gf.get_node_type("nonexistent") is None
    
    def test_search_nodes_by_keyword(self, sample_graph):
        """测试关键词搜索"""
        gf = GraphFuncs(sample_graph)
        
        results = gf.search_nodes_by_keyword("奖学金")
        assert len(results) > 0
        assert any("奖学金" in r.get('matched_value', '') for r in results)
