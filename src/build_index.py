"""
向量索引构建模块
负责使用嵌入模型和 Chroma 向量库构建索引
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from .config import get_config
    from .data_loader import load_all_subtitles
    from .chunking import chunk_documents
except ImportError:
    from config import get_config
    from data_loader import load_all_subtitles
    from chunking import chunk_documents


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    获取嵌入模型实例
    使用本地 Qwen3-Embedding-0.6B 模型（无需 API key）
    
    Returns:
        配置好的 HuggingFaceEmbeddings 实例
    """
    config = get_config()
    
    print(f"正在加载嵌入模型: {config.EMBEDDING_MODEL}")
    print("首次运行会自动下载模型文件（约 1.2GB），请稍候...")
    
    # 使用 HuggingFace 本地嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # 使用 CPU，如果有 GPU 可改为 'cuda'
        encode_kwargs={'normalize_embeddings': True}  # 归一化嵌入向量
    )
    
    print(f"✓ 嵌入模型加载完成: {config.EMBEDDING_MODEL}")
    
    return embeddings


def initialize_vectorstore(
    chunks,
    embeddings: HuggingFaceEmbeddings,
    persist_directory: Optional[Path] = None
) -> Chroma:
    """
    初始化向量库
    
    Args:
        chunks: 文档块列表
        embeddings: 嵌入模型实例
        persist_directory: 持久化目录
    
    Returns:
        Chroma 向量库实例
    """
    config = get_config()
    
    if persist_directory is None:
        persist_directory = config.CHROMA_PERSIST_DIR
    
    # 确保目录存在
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    print(f"\n正在构建向量索引...")
    print(f"  文档块数量: {len(chunks)}")
    print(f"  持久化目录: {persist_directory}")
    
    # 创建向量库
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory)
    )
    
    print(f"✓ 向量索引构建完成")
    
    return vectorstore


def build_index(rebuild: bool = False, data_dir: Optional[Path] = None):
    """
    构建向量索引
    
    Args:
        rebuild: 是否清空重建（True）或增量追加（False）
        data_dir: 字幕数据目录，默认使用配置中的目录
    """
    config = get_config()
    
    print("=" * 60)
    print("《疑犯追踪》字幕 RAG 系统 - 索引构建")
    print("=" * 60)
    
    # 验证配置
    try:
        config.validate()
    except ValueError as e:
        print(f"\n✗ 配置错误: {e}")
        return
    
    # 处理重建模式
    if rebuild:
        if config.CHROMA_PERSIST_DIR.exists():
            print(f"\n⚠ 重建模式：删除现有索引...")
            shutil.rmtree(config.CHROMA_PERSIST_DIR)
            print(f"✓ 已删除: {config.CHROMA_PERSIST_DIR}")
    else:
        if config.CHROMA_PERSIST_DIR.exists() and any(config.CHROMA_PERSIST_DIR.iterdir()):
            print(f"\n⚠ 检测到现有索引: {config.CHROMA_PERSIST_DIR}")
            print("提示: 使用 --rebuild 参数可以清空重建索引")
    
    # 加载字幕文件
    print(f"\n{'=' * 60}")
    print("步骤 1: 加载字幕文件")
    print(f"{'=' * 60}\n")
    
    documents = load_all_subtitles(data_dir)
    
    if not documents:
        print("\n✗ 没有找到字幕文件，无法构建索引")
        return
    
    # 分块处理
    print(f"\n{'=' * 60}")
    print("步骤 2: 文本分块")
    print(f"{'=' * 60}\n")
    
    chunks = chunk_documents(documents)
    
    if not chunks:
        print("\n✗ 分块失败，无法构建索引")
        return
    
    # 获取嵌入模型
    print(f"\n{'=' * 60}")
    print("步骤 3: 初始化嵌入模型")
    print(f"{'=' * 60}\n")
    
    try:
        embeddings = get_embeddings()
    except Exception as e:
        print(f"\n✗ 嵌入模型初始化失败: {e}")
        print("\n提示: 请检查 API Key 和网络连接")
        return
    
    # 构建向量索引
    print(f"\n{'=' * 60}")
    print("步骤 4: 构建向量索引")
    print(f"{'=' * 60}")
    
    try:
        vectorstore = initialize_vectorstore(chunks, embeddings)
    except Exception as e:
        print(f"\n✗ 索引构建失败: {e}")
        print("\n可能的原因:")
        print("  1. API Key 无效或已过期")
        print("  2. 网络连接问题")
        print("  3. API 配额不足")
        return
    
    # 显示统计信息
    print(f"\n{'=' * 60}")
    print("索引构建完成")
    print(f"{'=' * 60}")
    print(f"\n统计信息:")
    print(f"  原始文档数: {len(documents)}")
    print(f"  文档块数: {len(chunks)}")
    print(f"  向量库目录: {config.CHROMA_PERSIST_DIR}")
    
    # 显示已索引的集数
    episodes = sorted(set(doc.metadata.get('episode', 'Unknown') for doc in documents))
    print(f"\n已索引的集数:")
    for episode in episodes:
        print(f"  - {episode}")
    
    print(f"\n✓ 索引已持久化到磁盘")
    print(f"\n下一步: 运行 'python src/rag_chain.py' 测试检索功能")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="构建《疑犯追踪》字幕向量索引"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="清空现有索引并重新构建"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="字幕数据目录（默认使用配置中的目录）"
    )
    
    args = parser.parse_args()
    
    build_index(rebuild=args.rebuild, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
