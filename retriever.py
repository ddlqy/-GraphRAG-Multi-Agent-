"""向量检索器 - 适配校园知识图谱"""
import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """向量检索器
    
    使用 sentence-transformers 进行文本嵌入，FAISS 进行向量检索
    """
    
    def __init__(self, 
                 graph: Dict[str, Any],
                 node_text_keys: Dict[str, List[str]],
                 model_name: str = "BAAI/bge-m3",
                 cache: bool = True,
                 cache_dir: str = "./cache",
                 use_gpu: bool = False):
        """初始化检索器
        
        Args:
            graph: 图数据
            node_text_keys: 节点类型到文本字段的映射
            model_name: 嵌入模型名称
            cache: 是否使用缓存
            cache_dir: 缓存目录
            use_gpu: 是否使用GPU
        """
        logger.info("Initializing retriever...")
        
        self.use_gpu = use_gpu
        self.node_text_keys = node_text_keys
        self.model_name = model_name
        self.graph = graph
        self.cache = cache
        self.cache_dir = cache_dir
        
        # 延迟导入，避免未安装时报错
        try:
            import sentence_transformers
            import faiss
            self.model = sentence_transformers.SentenceTransformer(model_name)
            self.faiss = faiss
        except ImportError:
            logger.warning("sentence-transformers or faiss not installed. Retriever will use keyword search fallback.")
            self.model = None
            self.faiss = None
        
        self._reset()
    
    def _reset(self):
        """重置检索器"""
        docs, ids, meta_type = self._process_graph()
        
        if self.model is None:
            # 使用关键词搜索作为后备
            self.doc_lookup = ids
            self.doc_type = meta_type
            self.doc_texts = docs
            return
        
        # 检查缓存
        os.makedirs(self.cache_dir, exist_ok=True)
        save_model_name = self.model_name.split('/')[-1]
        cache_path = os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl')
        
        if self.cache and os.path.isfile(cache_path):
            embeds, self.doc_lookup, self.doc_type = pickle.load(open(cache_path, 'rb'))
            assert self.doc_lookup == ids
            assert self.doc_type == meta_type
        else:
            embeds = self._encode_docs(docs)
            self.doc_lookup = ids
            self.doc_type = meta_type
            pickle.dump([embeds, ids, meta_type], open(cache_path, 'wb'))
        
        self._init_index(embeds)
    
    def _process_graph(self) -> Tuple[List[str], List[str], List[str]]:
        """处理图数据，提取文档文本
        
        Returns:
            (docs, ids, meta_type) 三元组
        """
        docs = []
        ids = []
        meta_type = []
        
        for node_type_key in self.graph.keys():
            node_type = node_type_key.split('_nodes')[0] if '_nodes' in node_type_key else node_type_key
            logger.info(f'Loading text for {node_type}')
            
            for nid in self.graph[node_type_key]:
                # 获取节点文本
                features = self.graph[node_type_key][nid].get('features', {})
                text_keys = self.node_text_keys.get(node_type, ['name'])
                
                # 拼接所有文本字段
                text_parts = []
                for key in text_keys:
                    if key in features:
                        text_parts.append(str(features[key]))
                
                if text_parts:
                    docs.append(' '.join(text_parts))
                    ids.append(nid)
                    meta_type.append(node_type)
        
        return docs, ids, meta_type
    
    def _encode_docs(self, docs: List[str]) -> np.ndarray:
        """编码文档
        
        Args:
            docs: 文档列表
            
        Returns:
            嵌入矩阵
        """
        logger.info(f"Encoding {len(docs)} documents...")
        embeds = self.model.encode(docs, show_progress_bar=True, batch_size=32)
        return embeds
    
    def _init_index(self, embeds: np.ndarray):
        """初始化FAISS索引
        
        Args:
            embeds: 嵌入矩阵
        """
        logger.info("Initialize the index...")
        dim = embeds.shape[1]
        self.index = self.faiss.IndexFlatIP(dim)
        self.index.add(embeds)
        
        if self.use_gpu:
            self._move_index_to_gpu()
    
    def _move_index_to_gpu(self):
        """移动索引到GPU"""
        logger.info("Moving index to GPU")
        ngpu = self.faiss.get_num_gpus()
        if ngpu > 0:
            gpu_resources = []
            for i in range(ngpu):
                res = self.faiss.StandardGpuResources()
                gpu_resources.append(res)
            co = self.faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.usePrecomputed = False
            vres = self.faiss.GpuResourcesVector()
            vdev = self.faiss.Int32Vector()
            for i in range(ngpu):
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            self.index = self.faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)
    
    def search_single(self, query: str, topk: int = 1) -> Tuple[str, Dict[str, Any]]:
        """单次检索
        
        Args:
            query: 查询文本
            topk: 返回结果数量
            
        Returns:
            (node_id, node_info) 元组
        """
        if self.model is None:
            # 后备：关键词搜索
            return self._keyword_search(query)
        
        if self.index is None:
            raise ValueError("Index is not initialized")
        
        query_embed = self.model.encode(query, show_progress_bar=False)
        D, I = self.index.search(query_embed[None, :], topk)
        
        original_indice = np.array(self.doc_lookup)[I].tolist()[0][0]
        original_type = np.array(self.doc_type)[I].tolist()[0][0]
        
        return original_indice, self.graph[f'{original_type}_nodes'][original_indice]
    
    def search_batch(self, queries: List[str], topk: int = 10) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """批量检索
        
        Args:
            queries: 查询列表
            topk: 每个查询返回的结果数量
            
        Returns:
            结果列表的列表
        """
        if self.model is None:
            return [self._keyword_search_batch(q, topk) for q in queries]
        
        query_embeds = self.model.encode(queries, show_progress_bar=False)
        D, I = self.index.search(query_embeds, topk)
        
        results = []
        for i in range(len(queries)):
            query_results = []
            for j in range(topk):
                idx = I[i][j]
                if idx < 0:
                    continue
                nid = self.doc_lookup[idx]
                ntype = self.doc_type[idx]
                node_info = self.graph.get(f'{ntype}_nodes', {}).get(nid, {})
                query_results.append((nid, node_info))
            results.append(query_results)
        
        return results
    
    def _keyword_search(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """关键词搜索（后备方案）
        
        Args:
            query: 查询文本
            
        Returns:
            (node_id, node_info) 元组
        """
        query_lower = query.lower()
        best_match = None
        best_score = 0
        
        for i, (nid, ntype, text) in enumerate(zip(self.doc_lookup, self.doc_type, self.doc_texts)):
            # 简单的关键词匹配
            score = sum(1 for word in query_lower.split() if word in text.lower())
            if score > best_score:
                best_score = score
                best_match = (nid, ntype)
        
        if best_match:
            nid, ntype = best_match
            return nid, self.graph.get(f'{ntype}_nodes', {}).get(nid, {})
        
        # 如果没有匹配，返回第一个节点
        if self.doc_lookup:
            nid = self.doc_lookup[0]
            ntype = self.doc_type[0]
            return nid, self.graph.get(f'{ntype}_nodes', {}).get(nid, {})
        
        raise ValueError("No nodes in graph")
    
    def _keyword_search_batch(self, query: str, topk: int) -> List[Tuple[str, Dict[str, Any]]]:
        """关键词批量搜索（后备方案）
        
        Args:
            query: 查询文本
            topk: 返回数量
            
        Returns:
            结果列表
        """
        query_lower = query.lower()
        scored = []
        
        for i, (nid, ntype, text) in enumerate(zip(self.doc_lookup, self.doc_type, self.doc_texts)):
            score = sum(1 for word in query_lower.split() if word in text.lower())
            scored.append((score, nid, ntype))
        
        scored.sort(reverse=True)
        
        results = []
        for score, nid, ntype in scored[:topk]:
            node_info = self.graph.get(f'{ntype}_nodes', {}).get(nid, {})
            results.append((nid, node_info))
        
        return results
    
    def add_node(self, node_type: str, node_id: str, node_info: Dict[str, Any]):
        """添加新节点到索引
        
        Args:
            node_type: 节点类型
            node_id: 节点ID
            node_info: 节点信息
        """
        # 更新图数据
        key = f'{node_type}_nodes'
        if key not in self.graph:
            self.graph[key] = {}
        self.graph[key][node_id] = node_info
        
        # 获取文本
        features = node_info.get('features', {})
        text_keys = self.node_text_keys.get(node_type, ['name'])
        text_parts = []
        for k in text_keys:
            if k in features:
                text_parts.append(str(features[k]))
        
        if text_parts and self.model is not None:
            text = ' '.join(text_parts)
            embed = self.model.encode([text], show_progress_bar=False)
            self.index.add(embed)
            self.doc_lookup.append(node_id)
            self.doc_type.append(node_type)
