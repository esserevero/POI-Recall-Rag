"""
依赖测试脚本
验证所有核心依赖是否正确安装
"""

def test_dependencies():
    print("=" * 80)
    print("开始测试依赖...")
    print("=" * 80)
    
    # 测试 ChromaDB
    try:
        import chromadb
        print(f"✅ ChromaDB {chromadb.__version__} - 导入成功")
    except Exception as e:
        print(f"❌ ChromaDB - 导入失败: {e}")
        return False
    
    # 测试 LangChain
    try:
        import langchain
        from langchain_core import __version__ as lc_core_version
        print(f"✅ LangChain Core {lc_core_version} - 导入成功")
    except Exception as e:
        print(f"❌ LangChain - 导入失败: {e}")
        return False
    
    # 测试 LangChain OpenAI
    try:
        from langchain_openai import ChatOpenAI
        print(f"✅ LangChain OpenAI - 导入成功")
    except Exception as e:
        print(f"❌ LangChain OpenAI - 导入失败: {e}")
        return False
    
    # 测试 LangChain Community
    try:
        from langchain_community.vectorstores import Chroma
        print(f"✅ LangChain Community (Chroma) - 导入成功")
    except Exception as e:
        print(f"❌ LangChain Community - 导入失败: {e}")
        return False
    
    # 测试 SentenceTransformers
    try:
        from sentence_transformers import SentenceTransformer
        print(f"✅ SentenceTransformers - 导入成功")
    except Exception as e:
        print(f"❌ SentenceTransformers - 导入失败: {e}")
        return False
    
    # 测试 python-dotenv
    try:
        from dotenv import load_dotenv
        print(f"✅ python-dotenv - 导入成功")
    except Exception as e:
        print(f"❌ python-dotenv - 导入失败: {e}")
        return False
    
    # 测试 pytest
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} - 导入成功")
    except Exception as e:
        print(f"❌ pytest - 导入失败: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ 所有依赖测试通过！")
    print("=" * 80)
    return True


if __name__ == "__main__":
    import sys
    success = test_dependencies()
    sys.exit(0 if success else 1)
