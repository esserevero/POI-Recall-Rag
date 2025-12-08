"""
文本分块模块
使用 RecursiveCharacterTextSplitter 进行智能分块
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .config import get_config
except ImportError:
    from config import get_config


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    创建文本分割器
    
    基于字幕特征配置分隔符：
    - 空行（段落分隔）
    - 换行符
    - 空格
    
    Returns:
        配置好的 RecursiveCharacterTextSplitter
    """
    config = get_config()
    
    # 定义分隔符优先级
    # 优先按空行分割（字幕中的段落），然后是换行，最后是空格
    separators = [
        "\n\n",  # 双换行（段落）
        "\n",    # 单换行（对话行）
        " ",     # 空格
        ""       # 字符级别（最后手段）
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    return splitter


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    对文档列表进行分块
    
    Args:
        documents: Document 对象列表
    
    Returns:
        分块后的 Document 列表，每个块保留原始 metadata 并添加 chunk_id
    """
    if not documents:
        print("警告: 没有文档需要分块")
        return []
    
    splitter = create_text_splitter()
    
    all_chunks = []
    
    for doc in documents:
        # 分割文档
        chunks = splitter.split_documents([doc])
        
        # 为每个块添加 chunk_id
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            all_chunks.append(chunk)
        
        episode = doc.metadata.get("episode", "Unknown")
        print(f"✓ {episode}: 分块完成，共 {len(chunks)} 个块")
    
    print(f"\n总计生成 {len(all_chunks)} 个文本块")
    return all_chunks


def chunk_single_document(document: Document) -> List[Document]:
    """
    对单个文档进行分块
    
    Args:
        document: Document 对象
    
    Returns:
        分块后的 Document 列表
    """
    return chunk_documents([document])


if __name__ == "__main__":
    # 测试分块功能
    print("=== 测试文本分块模块 ===\n")
    
    try:
        from .data_loader import load_all_subtitles
    except ImportError:
        from data_loader import load_all_subtitles
    
    # 加载字幕
    print("加载字幕文件...\n")
    docs = load_all_subtitles()
    
    if not docs:
        print("没有找到字幕文件，无法测试分块功能")
    else:
        print(f"\n开始分块处理...\n")
        chunks = chunk_documents(docs)
        
        if chunks:
            print(f"\n分块统计:")
            print(f"  原始文档数: {len(docs)}")
            print(f"  分块总数: {len(chunks)}")
            print(f"  平均每个文档: {len(chunks) / len(docs):.1f} 个块")
            
            # 显示第一个块的示例
            print(f"\n第一个块示例:")
            first_chunk = chunks[0]
            print(f"  集数: {first_chunk.metadata['episode']}")
            print(f"  块 ID: {first_chunk.metadata['chunk_id']}")
            print(f"  长度: {len(first_chunk.page_content)} 字符")
            print(f"  内容预览:\n{first_chunk.page_content[:300]}...")
