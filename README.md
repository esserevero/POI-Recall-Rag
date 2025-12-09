# POI-Recall: 《疑犯追踪》字幕 RAG 问答系统

基于 LangChain 和 DeepSeek API 构建的《Person of Interest》第一季字幕检索增强生成（RAG）问答系统。

## 📖 项目简介

POI-Recall 是一个专注于影视剧集内容的智能问答系统。它通过 RAG 技术，将《疑犯追踪》（Person of Interest）的字幕数据转化为向量索引，结合大语言模型的理解能力，能够精准回答关于剧情、人物和对话的细节问题，并提供原始台词作为佐证。

## ✨ 核心功能

- **语义检索**: 基于语义理解而非关键词匹配，精准定位相关剧情片段。
- **智能问答**: 利用 DeepSeek LLM 生成流畅、准确的自然语言回答。
- **证据溯源**: 答案中包含引用的原始英文台词及中文解释，确保回答有据可依。
- **来源定位**: 明确标注答案来源的具体集数（如 S01E01）。

## 🛠️ 技术栈

- **编程语言**: Python 3.12
- **RAG 框架**: LangChain
- **LLM**: DeepSeek-V3 (deepseek-chat)
- **Embedding**: Qwen/Qwen3-Embedding-0.6B (本地运行)
- **向量数据库**: Chroma
- **配置管理**: python-dotenv

## 📂 项目结构

```text
POI-Recall/
├── Data/                   # 字幕数据目录
│   └── examples/           # 示例数据
├── db/                     # Chroma 向量数据库持久化目录
├── src/                    # 源代码目录
│   ├── config.py           # 配置管理
│   ├── data_loader.py      # 数据加载与预处理
│   ├── chunking.py         # 文本分块策略
│   ├── build_index.py      # 向量索引构建脚本
│   ├── rag_chain.py        # RAG 核心链路
│   └── cli_app.py          # 命令行交互界面
├── tests/                  # 测试用例
├── .env.example            # 环境变量示例文件
├── requirements.txt        # 项目依赖清单
└── README.md               # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

推荐使用 Conda 创建独立的虚拟环境：

```bash
# 创建并激活 Python 3.12 环境
conda create -n python312 python=3.12
conda activate python312

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

项目根目录下提供了 `.env.example` 模板。请复制该文件并重命名为 `.env`，然后填入您的 API Key。

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
# 填入您的 DeepSeek API Key
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 准备数据

请将《疑犯追踪》的英文字幕文件（.txt 格式）放入 `Data/` 目录下。
文件命名规范：`S01E01.txt`, `S01E02.txt` 等。

### 4. 构建索引

运行索引构建脚本，将字幕数据转化为向量并存储到本地 Chroma 数据库中：

```bash
python src/build_index.py
```

### 5. 启动问答

运行命令行应用，开始与系统交互：

```bash
python src/cli_app.py
```

## 📝 使用指南

### 字幕文件格式
- **格式**: 纯文本 (.txt)
- **内容**: 仅包含英文对话内容，推荐每行一句。
- **命名**: 必须遵循 `SxxExx.txt` 格式（如 `S01E01.txt`），以便系统提取元数据。

### 示例提问
- "What is the Machine?"
- "Who is John Reese?"
- "Describe the relationship between Finch and Reese."

## 📅 开发计划

- [x] **数据层**: 字幕加载、清洗与文本分块
- [x] **索引层**: 本地 Embedding 生成与 Chroma 向量库构建
- [x] **逻辑层**: 基于 LangChain 的 RAG 检索与问答链路
- [x] **交互层**: 命令行交互界面 (CLI)
- [ ] **测试**: 单元测试覆盖
- [ ] **界面**: Web UI 可视化界面

## ⚠️ 免责声明

- 本项目仅供技术学习与交流使用。
- 项目不包含任何受版权保护的字幕文件，请用户自行获取合法授权的数据。
- LLM 生成的回答可能存在幻觉，请以原始剧集内容为准。

## 📄 License

MIT License
