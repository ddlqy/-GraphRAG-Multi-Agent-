# GraphRAG + Multi-Agent 校园智能体系统

## 项目简介

本项目是一个基于 **GraphRAG** 和 **Multi-Agent** 技术的校园智能问答系统。与传统的RAG系统不同，本系统：

1. **使用知识图谱**替代纯向量检索，能理解实体间的关系
2. **多Agent协作**，像小团队一样协同完成复杂任务
3. **支持多跳推理**，能回答需要多步推理的复杂问题

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GraphRAG + Multi-Agent 校园智能体                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  知识图谱构建  │───▶│  GraphRAG    │───▶│   Multi-Agent 编排    │  │
│  │  (离线Pipeline)│    │  检索引擎     │    │   (在线推理)          │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    数据层                                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │  │
│  │  │ 校园文档   │  │ 知识图谱   │  │ 向量数据库  │  │ 社区摘要      │ │  │
│  │  │ (原始文本) │  │ (Neo4j)  │  │ (ChromaDB)│  │ (GraphRAG)   │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    展示层 (Streamlit)                          │  │
│  │  对话界面 │ Agent可视化 │ 知识图谱浏览 │ 推理过程展示             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 核心功能

### 1. 知识图谱构建
- LLM驱动的实体抽取
- 关系抽取和图构建
- 社区发现和摘要生成

### 2. GraphRAG检索
- **局部搜索**：实体锚定 + 邻居扩展
- **全局搜索**：社区摘要聚合 + Map-Reduce
- **混合检索**：KG + 向量融合排序

### 3. Multi-Agent协作
- **Planner Agent**：任务分解
- **Graph Search Agent**：知识图谱检索
- **Vector Search Agent**：向量文本检索
- **Calculator Agent**：计算推理
- **Synthesizer Agent**：答案整合

### 4. 反思机制
- 支持多种反思策略
- 自动改进推理过程

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的API密钥：

```bash
cp .env.example .env
# 编辑 .env 文件
```

### 3. 运行系统

```bash
# Streamlit界面模式
python run.py --mode streamlit

# 或直接运行Streamlit
streamlit run src/app/streamlit_app.py
```

## 项目结构

```
campus-graphrag-agent/
├── data/                          # 数据目录
│   ├── raw/                       # 原始校园文档
│   │   ├── policies/              # 规章制度
│   │   ├── scholarships/          # 奖学金
│   │   ├── courses/               # 课程信息
│   │   ├── campus_life/           # 校园生活
│   │   └── procedures/            # 办事流程
│   ├── processed/                 # 处理后数据
│   ├── graph/                     # 知识图谱
│   └── eval/                      # 评估数据
│
├── src/                           # 源代码
│   ├── config.py                  # 配置
│   ├── knowledge_graph/           # 知识图谱构建
│   │   ├── entity_extractor.py    # 实体抽取
│   │   ├── relation_extractor.py  # 关系抽取
│   │   └── graph_builder.py       # 图构建
│   ├── graphrag/                  # GraphRAG检索
│   │   ├── local_search.py        # 局部搜索
│   │   ├── global_search.py       # 全局搜索
│   │   └── hybrid_retriever.py    # 混合检索
│   ├── agents/                    # Multi-Agent系统
│   │   ├── graph_agent.py         # 图推理Agent
│   │   ├── orchestrator.py        # 编排器
│   │   ├── prompts.py             # Prompt模板
│   │   ├── fewshots.py            # Few-shot示例
│   │   └── utils.py               # 工具函数
│   ├── llm/                       # LLM抽象层
│   │   └── client.py              # 统一客户端
│   └── app/                       # 应用层
│       └── streamlit_app.py       # Streamlit界面
│
├── tests/                         # 测试
├── notebooks/                     # 实验Notebook
├── requirements.txt               # 依赖
├── .env.example                   # 环境变量示例
└── run.py                         # 运行入口
```

## 技术栈

| 层级 | 组件 | 技术 |
|------|------|------|
| LLM | 推理模型 | DeepSeek-V3 / Qwen3 |
| Embedding | 文本向量化 | BGE-M3 |
| KG存储 | 图数据库 | NetworkX + Neo4j |
| 向量库 | 文本检索 | ChromaDB / FAISS |
| Agent框架 | 编排引擎 | LangGraph / 自研 |
| 前端 | 界面 | Streamlit |

## 示例问题

系统支持回答复杂多跳问题，例如：

- "我GPA 3.5、拿过蓝桥杯省二，能申国家奖学金吗？"
- "国家奖学金和校级一等奖学金有什么不同？"
- "从提交转专业申请到最终批准，需要经过哪些步骤？"

## 开发计划

- [x] 项目骨架搭建
- [x] 图操作工具集成（来自Graph-Counselor）
- [x] Agent推理循环集成
- [ ] 校园数据采集
- [ ] 知识图谱构建Pipeline
- [ ] GraphRAG检索引擎完善
- [ ] Multi-Agent编排完善
- [ ] Streamlit界面开发
- [ ] 评估体系搭建

## 致谢

本项目参考和集成了以下开源项目：

- [Graph-Counselor](https://github.com/gjq100/Graph-Counselor) - Agent推理和图检索
- [bupt-assistant](https://github.com/SMZXzbc/bupt-assistant) - 校园数据和爬虫
- [rag_api](https://github.com/danny-avila/rag_api) - 文档处理和向量存储
- [paperless-ai](https://github.com/clusterzx/paperless-ai) - BM25+向量+重排序
