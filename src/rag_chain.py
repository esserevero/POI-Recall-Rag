"""
RAG 链路模块
实现完整的检索增强生成流程
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

try:
    from .config import get_config
except ImportError:
    from config import get_config


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    获取嵌入模型实例（与 build_index.py 保持一致）
    
    Returns:
        配置好的 HuggingFaceEmbeddings 实例
    """
    config = get_config()
    
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return embeddings


def load_vectorstore(persist_directory: Optional[Path] = None) -> Chroma:
    """
    加载持久化的向量库
    
    Args:
        persist_directory: 向量库目录，默认使用配置中的目录
    
    Returns:
        Chroma 向量库实例
    
    Raises:
        ValueError: 如果向量库不存在
    """
    config = get_config()
    
    if persist_directory is None:
        persist_directory = config.CHROMA_PERSIST_DIR
    
    if not persist_directory.exists():
        raise ValueError(
            f"向量库不存在: {persist_directory}\n"
            f"请先运行 'python src/build_index.py' 构建索引"
        )
    
    print(f"正在加载向量库: {persist_directory}")
    
    embeddings = get_embeddings()
    
    vectorstore = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings
    )
    
    print(f"✓ 向量库加载完成")
    
    return vectorstore


def get_retriever(vectorstore: Optional[Chroma] = None, k: int = 5):
    """
    获取检索器
    
    Args:
        vectorstore: Chroma 向量库实例，默认自动加载
        k: 检索的文档数量
    
    Returns:
        配置好的检索器
    """
    config = get_config()
    
    if vectorstore is None:
        vectorstore = load_vectorstore()
    
    if k is None:
        k = config.RETRIEVAL_K
    
    # 创建检索器，使用相似度搜索
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    return retriever


def format_docs(docs: List[Document]) -> str:
    """
    格式化检索到的文档为上下文字符串
    
    Args:
        docs: 检索到的文档列表
    
    Returns:
        格式化后的上下文字符串
    """
    if not docs:
        return "没有找到相关内容。"
    
    formatted_chunks = []
    
    for i, doc in enumerate(docs, 1):
        episode = doc.metadata.get('episode', 'Unknown')
        chunk_id = doc.metadata.get('chunk_id', 'N/A')
        content = doc.page_content.strip()
        
        formatted_chunk = f"""[来源 {i}] {episode} (chunk {chunk_id})
{content}
"""
        formatted_chunks.append(formatted_chunk)
    
    return "\n---\n".join(formatted_chunks)


def create_prompt_template() -> ChatPromptTemplate:
    """
    创建 RAG 提示模板
    
    Returns:
        配置好的 ChatPromptTemplate
    """
    template = """你是《疑犯追踪》(Person of Interest) 剧集的专家助手。你的任务是基于提供的字幕片段回答用户的问题。

回答要求：
1. 基于提供的上下文回答问题，不要编造信息
2. 引用相关的英文原台词（使用引号）
3. 标注台词来源的集数（格式：S01E01）
4. 如果需要，可以提供简短的中文解释
5. 如果上下文中没有相关信息，请明确说明

上下文信息：
{context}

用户问题：{question}

请提供详细的回答："""

    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt


def get_llm() -> ChatOpenAI:
    """
    获取 LLM 实例（使用 DeepSeek API）
    
    Returns:
        配置好的 ChatOpenAI 实例
    """
    config = get_config()
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        openai_api_key=config.DEEPSEEK_API_KEY,
        openai_api_base=config.DEEPSEEK_API_BASE
    )
    
    return llm


def create_rag_chain(retriever=None):
    """
    创建完整的 RAG 链路
    
    Args:
        retriever: 检索器实例，默认自动创建
    
    Returns:
        配置好的 RAG 链路
    """
    if retriever is None:
        retriever = get_retriever()
    
    prompt = create_prompt_template()
    llm = get_llm()
    
    # 使用 LangChain Expression Language (LCEL) 构建链路
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def query(
    question: str,
    retriever=None,
    return_sources: bool = True
) -> Dict[str, Any]:
    """
    执行 RAG 查询
    
    Args:
        question: 用户问题
        retriever: 检索器实例，默认自动创建
        return_sources: 是否返回来源文档
    
    Returns:
        包含答案和来源的字典
    """
    if retriever is None:
        retriever = get_retriever()
    
    # 检索相关文档
    retrieved_docs = retriever.get_relevant_documents(question)
    
    # 创建并执行 RAG 链路
    rag_chain = create_rag_chain(retriever)
    answer = rag_chain.invoke(question)
    
    result = {
        "question": question,
        "answer": answer
    }
    
    if return_sources:
        sources = []
        for doc in retrieved_docs:
            source_info = {
                "episode": doc.metadata.get('episode', 'Unknown'),
                "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
                "filename": doc.metadata.get('filename', 'Unknown'),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        
        result["sources"] = sources
        result["num_sources"] = len(sources)
    
    return result


def print_result(result: Dict[str, Any]):
    """
    格式化打印查询结果
    
    Args:
        result: query() 函数返回的结果字典
    """
    print("\n" + "=" * 80)
    print(f"问题: {result['question']}")
    print("=" * 80)
    print(f"\n回答:\n{result['answer']}\n")
    
    if "sources" in result and result["sources"]:
        print("-" * 80)
        print(f"来源文档 ({result['num_sources']} 个):")
        print("-" * 80)
        
        for i, source in enumerate(result["sources"], 1):
            print(f"\n[{i}] {source['episode']} (chunk {source['chunk_id']})")
            print(f"    文件: {source['filename']}")
            print(f"    内容预览: {source['content'][:150]}...")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # 测试 RAG 链路
    print("=" * 80)
    print("《疑犯追踪》字幕 RAG 系统 - 测试")
    print("=" * 80)
    
    # 验证配置
    config = get_config()
    try:
        config.validate()
    except ValueError as e:
        print(f"\n✗ 配置错误: {e}")
        exit(1)
    
    # 测试问题列表
    test_questions = [
        "What is the Machine?",
        "Who is John Reese?",
        "Tell me about Harold Finch",
        "What happened in the first episode?"
    ]
    
    print("\n正在初始化 RAG 系统...\n")
    
    try:
        # 加载向量库和检索器
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore)
        
        print(f"\n✓ RAG 系统初始化完成")
        print(f"\n开始测试查询...\n")
        
        # 执行测试查询
        for question in test_questions:
            print(f"\n{'=' * 80}")
            print(f"测试问题: {question}")
            print('=' * 80)
            
            result = query(question, retriever=retriever)
            print_result(result)
            
            # 添加分隔
            input("按 Enter 继续下一个问题...")
    
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
