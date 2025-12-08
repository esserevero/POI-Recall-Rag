# 《疑犯追踪》字幕 RAG 问答系统

基于 LangChain 和 OpenAI 的《Person of Interest》第一季字幕检索增强生成（RAG）问答系统。

## 功能特性

- 📚 从纯文本字幕文件构建向量索引
- 🔍 基于语义检索相关剧情片段
- 💬 通过 LLM 生成自然语言回答
- 🎯 引用原始英文台词并附中文解释
- 📍 标注答案来源的具体集数

## 技术栈

- **语言**: Python 3.12
- **RAG 框架**: LangChain
- **嵌入模型**: OpenAI text-embedding-3-small
- **LLM**: OpenAI gpt-4o-mini
- **向量数据库**: Chroma
- **配置管理**: python-dotenv

## 项目结构

```
POI-Recall/
├── data/                   # 字幕文件目录（git 忽略）
│   └── examples/          # 示例字幕片段
├── db/                    # Chroma 向量库（git 忽略）
├── src/                   # 源代码
│   ├── config.py         # 配置与常量
│   ├── data_loader.py    # 字幕加载
│   ├── chunking.py       # 文本分块
│   ├── build_index.py    # 索引构建
│   ├── rag_chain.py      # RAG 链路
│   └── cli_app.py        # 命令行界面
├── tests/                 # 测试文件
├── .env                   # 环境变量（需自行创建）
├── .env.example          # 环境变量示例
├── requirements.txt      # Python 依赖
└── README.md             # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境（如果使用 conda）
conda activate python312

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. 准备字幕文件

将字幕文件放入 `data/` 目录，命名格式：`S01E01.txt`, `S01E02.txt` 等。

### 4. 构建索引

```bash
python src/build_index.py
```

### 5. 开始问答

```bash
python src/cli_app.py
```

## 使用说明

### 字幕文件要求

- 纯文本格式（.txt）
- 每集一个文件
- 文件名格式：`S01E01.txt`（第一季第一集）
- 内容为英文对话文本

### 示例问题

- "What is the Machine?"
- "Who is John Reese?"
- "What happened in the first episode?"
- "Tell me about Finch and Reese's relationship"

## 开发进度

- [x] 阶段 1：数据加载 & 分块
- [ ] 阶段 2：索引构建
- [ ] 阶段 3：RAG Pipeline
- [ ] 阶段 4：CLI 交互
- [ ] 阶段 5：测试 & 文档
- [ ] 阶段 6：Web UI（可选）

## 注意事项

- 本项目仅用于学习和研究目的
- 字幕文件不包含在仓库中，需自行准备
- 使用 OpenAI API 会产生费用
- 系统回答基于字幕文本，不保证 100% 准确

## License

MIT
