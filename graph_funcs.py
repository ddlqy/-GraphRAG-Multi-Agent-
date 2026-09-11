"""图操作工具 - 适配校园知识图谱"""
import networkx as nx
from typing import Dict, List, Optional, Any


class GraphFuncs:
    """校园知识图谱操作工具集
    
    支持的操作：
    - Retrieve[keyword]: 向量检索相关节点
    - Feature[Node, feature]: 获取节点属性
    - Degree[Node, neighbor_type]: 计算节点的邻居数量
    - Neighbor[Node, neighbor_type]: 列出节点的邻居
    """
    
    def __init__(self, graph: Dict[str, Any]):
        """初始化图操作工具
        
        Args:
            graph: 图数据，格式为 {node_type: {node_id: {features: {}, neighbors: {}}}}
        """
        self._reset(graph)
    
    def _reset(self, graph: Dict[str, Any]):
        """重置图索引"""
        self.graph = graph
        graph_index = {}
        nid_set = set()
        
        for node_type in graph:
            for nid in graph[node_type]:
                assert nid not in nid_set, f"Duplicate node ID: {nid}"
                nid_set.add(nid)
                graph_index[nid] = graph[node_type][nid]
                graph_index[nid]['node_type'] = node_type.replace('_nodes', '')
        
        self.graph_index = graph_index
    
    def check_neighbours(self, node: str, neighbor_type: Optional[str] = None) -> str:
        """检查节点的邻居
        
        Args:
            node: 节点ID
            neighbor_type: 邻居类型（可选）
            
        Returns:
            邻居节点ID列表的字符串表示
        """
        if node not in self.graph_index:
            return f"Node {node} not found in graph"
        
        node_info = self.graph_index[node]
        if 'neighbors' not in node_info:
            return f"Node {node} has no neighbors information"
        
        if neighbor_type:
            if neighbor_type not in node_info['neighbors']:
                return f"Node {node} has no {neighbor_type} neighbors"
            return str(node_info['neighbors'][neighbor_type])
        else:
            return str(node_info['neighbors'])
    
    def check_nodes(self, node: str, feature: Optional[str] = None) -> str:
        """检查节点属性
        
        Args:
            node: 节点ID
            feature: 特征名（可选）
            
        Returns:
            节点属性的字符串表示
        """
        if node not in self.graph_index:
            return f"Node {node} not found in graph"
        
        node_info = self.graph_index[node]
        if 'features' not in node_info:
            return f"Node {node} has no features information"
        
        if feature:
            if feature not in node_info['features']:
                return f"Node {node} has no feature '{feature}'"
            return str(node_info['features'][feature])
        else:
            return str(node_info['features'])
    
    def check_degree(self, node: str, neighbor_type: str) -> str:
        """检查节点指定类型邻居的数量
        
        Args:
            node: 节点ID
            neighbor_type: 邻居类型
            
        Returns:
            邻居数量的字符串表示
        """
        if node not in self.graph_index:
            return f"Node {node} not found in graph"
        
        node_info = self.graph_index[node]
        if 'neighbors' not in node_info:
            return "0"
        
        if neighbor_type not in node_info['neighbors']:
            return "0"
        
        return str(len(node_info['neighbors'][neighbor_type]))
    
    def check_all_neighbour(self, node_q: str) -> Dict[str, str]:
        """查找所有指向指定节点的邻居
        
        Args:
            node_q: 目标节点ID
            
        Returns:
            {source_node: neighbor_type} 字典
        """
        nodes = {}
        for node, node_info in self.graph_index.items():
            if 'neighbors' not in node_info:
                continue
            for neighbor_type, neighbours in node_info['neighbors'].items():
                if node_q in neighbours:
                    nodes[node] = neighbor_type
        return nodes
    
    def get_node_type(self, node: str) -> Optional[str]:
        """获取节点类型
        
        Args:
            node: 节点ID
            
        Returns:
            节点类型字符串
        """
        if node in self.graph_index:
            return self.graph_index[node].get('node_type')
        return None
    
    def search_nodes_by_keyword(self, keyword: str, node_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """根据关键词搜索节点
        
        Args:
            keyword: 搜索关键词
            node_type: 限定节点类型（可选）
            
        Returns:
            匹配的节点列表
        """
        results = []
        keyword_lower = keyword.lower()
        
        for nid, info in self.graph_index.items():
            # 检查节点类型
            if node_type and info.get('node_type') != node_type:
                continue
            
            # 在特征中搜索关键词
            if 'features' in info:
                for key, value in info['features'].items():
                    if keyword_lower in str(value).lower():
                        results.append({
                            'id': nid,
                            'type': info.get('node_type'),
                            'features': info['features'],
                            'matched_field': key,
                            'matched_value': value
                        })
                        break
        
        return results
