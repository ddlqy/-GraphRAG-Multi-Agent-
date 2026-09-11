"""GraphRAG 端到端检索链路测试

测试覆盖：
1. 实体链接器：用户问题 → 知识图谱实体
2. 局部搜索：实体邻居扩展 + 子图上下文
3. 全局搜索：Map-Reduce 社区摘要检索
4. 混合检索：KG + 向量融合排序
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def graph():
    """加载知识图谱"""
    path = PROJECT_ROOT / "data" / "graph" / "graph.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def communities():
    """加载社区数据"""
    path = PROJECT_ROOT / "data" / "graph" / "communities.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


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


# ==================== 实体链接器测试 ====================

class TestEntityLinker:
    """实体链接器测试"""

    def test_init(self, graph):
        """测试初始化"""
        from src.graphrag.entity_linker import EntityLinker
        linker = EntityLinker(graph)
        assert len(linker.node_index) == 50

    def test_exact_match(self, graph):
        """测试精确匹配"""
        from src.graphrag.entity_linker import EntityLinker
        linker = EntityLinker(graph)

        results = linker.link("国家奖学金")
        assert len(results) > 0
        assert any(r['match_type'] == 'exact' for r in results)
        assert results[0]['name'] == '国家奖学金'

    def test_substring_match(self, graph):
        """测试子串匹配"""
        from src.graphrag.entity_linker import EntityLinker
        linker = EntityLinker(graph)

        results = linker.link("奖学金申请条件")
        assert len(results) > 0
        names = [r['name'] for r in results]
        assert any('奖学金' in n for n in names)

    def test_synonym_match(self, graph):
        """测试同义词匹配"""
        from src.graphrag.entity_linker import EntityLinker
        linker = EntityLinker(graph)

        results = linker.link("绩点要求")
        assert len(results) > 0

    def test_type_inference(self, graph):
        """测试类型推断"""
        from src.graphrag.entity_linker import EntityLinker
        linker = EntityLinker(graph)

        results = linker.link("有哪些竞赛")
        assert len(results) > 0
        types = [r['type'] for r in results]
        assert 'Competition' in types

    def test_library_query(self, graph):
        """测试图书馆查询"""
        from src.graphrag.entity_linker import EntityLinker
        linker = EntityLinker(graph)

        results = linker.link("图书馆开放时间")
        assert len(results) > 0
        assert any('图书馆' in r['name'] for r in results)

    def test_department_query(self, graph):
        """测试院系查询"""
        from src.graphrag.entity_linker import EntityLinker
        linker = EntityLinker(graph)

        results = linker.link("计算机学院有哪些专业")
        assert len(results) > 0

    def test_get_entity_neighbors(self, graph):
        """测试获取实体邻居"""
        from src.graphrag.entity_linker import EntityLinker
        linker = EntityLinker(graph)

        neighbors = linker.get_entity_neighbors("dept_jwc", max_hops=1)
        assert "dept_jwc" in neighbors
        assert len(neighbors) > 1


# ==================== 局部搜索测试 ====================

class TestLocalSearch:
    """局部搜索测试"""

    def test_init(self, graph):
        """测试初始化"""
        from src.graphrag.local_search import LocalSearch
        search = LocalSearch(graph)
        assert len(search.node_index) == 50

    def test_keyword_search(self, graph):
        """测试关键词搜索"""
        from src.graphrag.local_search import LocalSearch
        search = LocalSearch(graph)

        results = search.search("国家奖学金", max_hops=1)
        assert results['entities']
        assert results['context'] != "未找到相关实体"

    def test_neighbor_expansion(self, graph):
        """测试邻居扩展"""
        from src.graphrag.local_search import LocalSearch
        search = LocalSearch(graph)

        results = search.search("教务处", max_hops=2)
        assert results['subgraph']
        # 应该扩展到奖学金和流程节点
        types = set(n['type'] for n in results['subgraph'].values())
        assert len(types) > 1

    def test_context_extraction(self, graph):
        """测试上下文提取"""
        from src.graphrag.local_search import LocalSearch
        search = LocalSearch(graph)

        results = search.search("图书馆", max_hops=1)
        context = results['context']
        assert '图书馆' in context
        assert '[Department]' in context or '[Location]' in context

    def test_multi_hop(self, graph):
        """测试多跳扩展"""
        from src.graphrag.local_search import LocalSearch
        search = LocalSearch(graph)

        results_1hop = search.search("国家奖学金", max_hops=1)
        results_2hop = search.search("国家奖学金", max_hops=2)

        # 2跳应该包含更多节点
        assert len(results_2hop['subgraph']) >= len(results_1hop['subgraph'])

    def test_get_entity_neighbors(self, graph):
        """测试获取实体邻居"""
        from src.graphrag.local_search import LocalSearch
        search = LocalSearch(graph)

        neighbors = search.get_entity_neighbors("scholarship_national")
        assert len(neighbors) > 0
        assert any(n['relation'] == 'MANAGED_BY' for n in neighbors)


# ==================== 全局搜索测试 ====================

class TestGlobalSearch:
    """全局搜索测试"""

    def test_init(self, graph, communities):
        """测试初始化"""
        from src.graphrag.global_search import GlobalSearch
        search = GlobalSearch(llm_client=None, communities=communities)
        assert len(search.communities) == 5

    def test_no_communities(self):
        """测试无社区数据"""
        from src.graphrag.global_search import GlobalSearch
        search = GlobalSearch(llm_client=None, communities=[])
        result = search.search("测试问题")
        assert "无法回答" in result.response

    def test_map_phase(self, graph, communities, llm):
        """测试Map阶段"""
        from src.graphrag.global_search import GlobalSearch
        search = GlobalSearch(llm_client=llm, communities=communities)

        map_responses = search._map_phase("国家奖学金的条件是什么")
        assert len(map_responses) > 0

    def test_full_search(self, graph, communities, llm):
        """测试完整全局搜索"""
        from src.graphrag.global_search import GlobalSearch
        search = GlobalSearch(llm_client=llm, communities=communities)

        result = search.search("学校有哪些奖学金")
        assert result.response
        assert result.response != "根据现有数据，无法回答此问题。"
        assert len(result.map_responses) > 0
        assert result.completion_time > 0


# ==================== 混合检索测试 ====================

class TestHybridRetriever:
    """混合检索器测试"""

    def test_init(self, graph, communities):
        """测试初始化"""
        from src.graphrag import LocalSearch, GlobalSearch, HybridRetriever

        local = LocalSearch(graph)
        global_s = GlobalSearch(llm_client=None, communities=communities)
        hybrid = HybridRetriever(local_search=local, global_search=global_s)
        assert hybrid.local_search is not None
        assert hybrid.global_search is not None

    def test_strategy_auto(self, graph):
        """测试自动策略选择"""
        from src.graphrag import LocalSearch, HybridRetriever

        local = LocalSearch(graph)
        hybrid = HybridRetriever(local_search=local)

        # 具体问题应该选local
        assert hybrid._determine_strategy("国家奖学金多少钱") == "local"

        # 全局问题应该选global
        assert hybrid._determine_strategy("所有奖学金有哪些") == "global"

    def test_local_only(self, graph):
        """测试仅局部搜索"""
        from src.graphrag import LocalSearch, HybridRetriever

        local = LocalSearch(graph)
        hybrid = HybridRetriever(local_search=local)

        result = hybrid.retrieve("图书馆开放时间", strategy="local")
        assert result['strategy'] == 'local'
        assert 'local' in result['results']

    def test_hybrid_search(self, graph, communities, llm):
        """测试混合检索"""
        from src.graphrag import LocalSearch, GlobalSearch, HybridRetriever

        local = LocalSearch(graph)
        global_s = GlobalSearch(llm_client=llm, communities=communities)
        hybrid = HybridRetriever(local_search=local, global_search=global_s, llm_client=llm)

        result = hybrid.retrieve("国家奖学金的申请条件", strategy="hybrid")
        assert result['strategy'] == 'hybrid'
        assert result['answer']
        assert result['answer'] != "未找到相关信息"


# ==================== 端到端测试 ====================

class TestEndToEnd:
    """端到端检索链路测试"""

    def test_full_pipeline(self, graph, communities, llm):
        """完整检索链路测试"""
        from src.graphrag import LocalSearch, GlobalSearch, HybridRetriever, EntityLinker

        # 1. 初始化所有组件
        entity_linker = EntityLinker(graph)
        local_search = LocalSearch(graph)
        global_search = GlobalSearch(llm_client=llm, communities=communities)
        hybrid = HybridRetriever(
            local_search=local_search,
            global_search=global_search,
            llm_client=llm
        )

        # 2. 测试问题列表
        test_queries = [
            "国家奖学金的金额是多少？",
            "图书馆的开放时间？",
            "计算机学院有哪些专业？",
            "如何转专业？",
        ]

        for query in test_queries:
            # 实体链接
            entities = entity_linker.link(query)
            assert len(entities) > 0, f"实体链接失败: {query}"

            # 局部搜索
            local_result = local_search.search(query, max_hops=2)
            assert local_result['context'] != "未找到相关实体", f"局部搜索失败: {query}"

            # 混合检索
            hybrid_result = hybrid.retrieve(query, strategy="auto")
            assert hybrid_result['answer'], f"混合检索失败: {query}"

    def test_entity_linking_coverage(self, graph):
        """实体链接覆盖率测试"""
        from src.graphrag.entity_linker import EntityLinker

        linker = EntityLinker(graph)

        test_queries = [
            ("国家奖学金", "Scholarship"),
            ("图书馆", "Location"),
            ("教务处", "Department"),
            ("蓝桥杯", "Competition"),
            ("转专业", "Procedure"),
            ("GPA", "Condition"),
        ]

        for query, expected_type in test_queries:
            results = linker.link(query)
            assert len(results) > 0, f"链接失败: {query}"
            types = [r['type'] for r in results]
            assert expected_type in types, f"类型不匹配: {query}, 期望 {expected_type}, 实际 {types}"

    def test_search_result_quality(self, graph, llm):
        """搜索结果质量测试"""
        from src.graphrag.local_search import LocalSearch

        search = LocalSearch(graph)

        # 搜索图书馆应该返回图书馆相关节点
        results = search.search("图书馆", max_hops=1)
        context = results['context']
        assert '图书馆' in context

        # 搜索奖学金应该返回奖学金相关节点
        results = search.search("国家奖学金", max_hops=2)
        context = results['context']
        assert '奖学金' in context or 'GPA' in context
