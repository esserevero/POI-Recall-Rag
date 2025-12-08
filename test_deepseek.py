"""
简单测试脚本 - 清晰显示 LLM 完整回答
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_chain import query, load_vectorstore, get_retriever
from config import get_config


def main():
    print("\n" + "=" * 80)
    print("DeepSeek API 测试 - 查看完整 LLM 回答")
    print("=" * 80 + "\n")
    
    # 验证配置
    config = get_config()
    try:
        config.validate()
        print("✓ 配置验证通过")
        print(f"✓ DeepSeek API Key: {config.DEEPSEEK_API_KEY[:20]}...")
        print(f"✓ LLM 模型: {config.LLM_MODEL}\n")
    except ValueError as e:
        print(f"✗ 配置错误: {e}")
        return
    
    # 测试问题
    test_question = "What is the Machine?"
    
    print("正在初始化 RAG 系统...\n")
    
    try:
        # 加载向量库
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore)
        
        print("✓ RAG 系统初始化完成\n")
        print("=" * 80)
        print(f"测试问题: {test_question}")
        print("=" * 80 + "\n")
        
        # 执行查询
        print("正在调用 DeepSeek API 生成回答...\n")
        result = query(test_question, retriever=retriever)
        
        # 清晰显示完整答案
        print("\n" + "=" * 80)
        print("【DeepSeek 生成的完整回答】")
        print("=" * 80)
        print(result['answer'])
        print("=" * 80 + "\n")
        
        # 显示来源信息
        print(f"来源文档数量: {result['num_sources']}")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] {source['episode']} - chunk {source['chunk_id']}")
        
        print("\n✓ 测试完成！DeepSeek API 工作正常！\n")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
