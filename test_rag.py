"""
RAG 链路测试脚本
快速测试 RAG 功能
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_chain import query, print_result, load_vectorstore, get_retriever
from config import get_config


def main():
    print("=" * 80)
    print("《疑犯追踪》字幕 RAG 系统 - 快速测试")
    print("=" * 80)
    
    # 验证配置
    config = get_config()
    try:
        config.validate()
        print("\n✓ 配置验证通过")
    except ValueError as e:
        print(f"\n✗ 配置错误: {e}")
        return
    
    # 测试问题
    test_question = "What is the Machine?"
    
    print(f"\n正在初始化 RAG 系统...")
    
    try:
        # 加载向量库
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore)
        
        print(f"✓ RAG 系统初始化完成\n")
        
        # 执行查询
        print(f"测试问题: {test_question}\n")
        result = query(test_question, retriever=retriever)
        
        # 打印结果
        print_result(result)
        
        print("✓ RAG 测试完成！")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
