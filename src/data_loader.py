"""
字幕加载模块
负责读取字幕文件并封装为 LangChain Document 对象
"""

import re
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

try:
    from .config import get_config
except ImportError:
    from config import get_config


def extract_episode_from_filename(filename: str) -> str:
    """
    从文件名中提取集数信息
    
    Args:
        filename: 文件名，如 "S1E1.txt" 或 "S01E01.txt"
    
    Returns:
        标准化的集数，如 "S01E01"
    """
    # 匹配 S1E1, S01E01, s1e1 等格式
    pattern = r'[Ss](\d+)[Ee](\d+)'
    match = re.search(pattern, filename)
    
    if match:
        season = match.group(1).zfill(2)  # 补齐为两位数
        episode = match.group(2).zfill(2)
        return f"S{season}E{episode}"
    
    # 如果无法匹配，返回原文件名（不含扩展名）
    return Path(filename).stem


def clean_subtitle_text(text: str) -> str:
    """
    清洗字幕文本
    只做非语义的清理：压缩多余空行、去除尾随空格
    不修改文本内容本身
    
    Args:
        text: 原始字幕文本
    
    Returns:
        清洗后的文本
    """
    # 去除每行尾随空格
    lines = [line.rstrip() for line in text.split('\n')]
    
    # 压缩连续空行为单个空行
    cleaned_lines = []
    prev_empty = False
    
    for line in lines:
        is_empty = len(line.strip()) == 0
        
        if is_empty:
            if not prev_empty:
                cleaned_lines.append('')
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    
    return '\n'.join(cleaned_lines)


def load_subtitle_file(file_path: Path) -> Optional[Document]:
    """
    加载单个字幕文件
    
    Args:
        file_path: 字幕文件路径
    
    Returns:
        Document 对象，包含文本和 metadata；如果文件无效则返回 None
    """
    if not file_path.exists():
        print(f"警告: 文件不存在 - {file_path}")
        return None
    
    if not file_path.is_file():
        print(f"警告: 不是文件 - {file_path}")
        return None
    
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # 清洗文本
        cleaned_text = clean_subtitle_text(raw_text)
        
        # 如果清洗后文本为空，跳过
        if not cleaned_text.strip():
            print(f"警告: 文件内容为空 - {file_path}")
            return None
        
        # 提取集数信息
        episode = extract_episode_from_filename(file_path.name)
        
        # 创建 Document 对象
        document = Document(
            page_content=cleaned_text,
            metadata={
                "episode": episode,
                "source": str(file_path),
                "filename": file_path.name
            }
        )
        
        return document
    
    except Exception as e:
        print(f"错误: 读取文件失败 - {file_path}: {e}")
        return None


def load_all_subtitles(directory: Optional[Path] = None) -> List[Document]:
    """
    批量加载目录下所有字幕文件
    
    Args:
        directory: 字幕目录路径，默认使用配置中的 SUBTITLES_DIR
    
    Returns:
        Document 对象列表
    """
    config = get_config()
    
    if directory is None:
        directory = config.SUBTITLES_DIR
    
    if not directory.exists():
        print(f"错误: 目录不存在 - {directory}")
        return []
    
    # 查找所有 .txt 文件
    txt_files = sorted(directory.glob("*.txt"))
    
    if not txt_files:
        print(f"警告: 目录中没有找到 .txt 文件 - {directory}")
        return []
    
    print(f"找到 {len(txt_files)} 个字幕文件")
    
    documents = []
    for file_path in txt_files:
        doc = load_subtitle_file(file_path)
        if doc:
            documents.append(doc)
            print(f"✓ 加载成功: {doc.metadata['episode']} ({file_path.name})")
    
    print(f"\n总计加载 {len(documents)} 个字幕文件")
    return documents


if __name__ == "__main__":
    # 测试加载功能
    print("=== 测试字幕加载模块 ===\n")
    
    # 测试单个文件加载
    config = get_config()
    test_file = config.SUBTITLES_DIR / "S1E1.txt"
    
    if test_file.exists():
        print(f"测试加载单个文件: {test_file}\n")
        doc = load_subtitle_file(test_file)
        if doc:
            print(f"集数: {doc.metadata['episode']}")
            print(f"文本长度: {len(doc.page_content)} 字符")
            print(f"前 200 字符:\n{doc.page_content[:200]}...\n")
    
    # 测试批量加载
    print("\n测试批量加载所有字幕文件:\n")
    docs = load_all_subtitles()
    
    if docs:
        print(f"\n成功加载的集数:")
        for doc in docs:
            print(f"  - {doc.metadata['episode']}: {len(doc.page_content)} 字符")
