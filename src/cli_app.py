"""
命令行交互界面
提供交互式问答循环
"""

import sys
from pathlib import Path
from typing import Optional

try:
    from .rag_chain import load_vectorstore, get_retriever, query, print_result
    from .config import get_config
except ImportError:
    from rag_chain import load_vectorstore, get_retriever, query, print_result
    from config import get_config


def display_welcome():
    """显示欢迎信息"""
    print("\n" + "=" * 80)
    print("《疑犯追踪》字幕 RAG 问答系统")
    print("Person of Interest Subtitle RAG Q&A System")
    print("=" * 80)
    print("\n欢迎使用！您可以询问任何关于《疑犯追踪》第一季的问题。")
    print("\n使用说明:")
    print("  - 输入您的问题并按 Enter 键")
    print("  - 输入 'quit', 'exit' 或 'q' 退出程序")
    print("  - 输入 'help' 查看帮助信息")
    print("\n" + "-" * 80 + "\n")


def display_help():
    """显示帮助信息"""
    print("\n" + "=" * 80)
    print("帮助信息")
    print("=" * 80)
    print("\n本系统基于《疑犯追踪》第一季字幕构建，可以回答关于剧情、角色、对话等问题。")
    print("\n示例问题:")
    print("  - What is the Machine?")
    print("  - Who is John Reese?")
    print("  - Tell me about Harold Finch")
    print("  - What happened in the first episode?")
    print("  - What is the relationship between Finch and Reese?")
    print("\n系统会:")
    print("  ✓ 检索相关的字幕片段")
    print("  ✓ 生成基于原文的回答")
    print("  ✓ 引用原始英文台词")
    print("  ✓ 标注来源集数")
    print("\n命令:")
    print("  quit, exit, q  - 退出程序")
    print("  help           - 显示此帮助信息")
    print("\n" + "=" * 80 + "\n")


def run_interactive_loop(retriever):
    """
    运行交互式问答循环
    
    Args:
        retriever: 已初始化的检索器实例
    """
    print("系统已就绪！请输入您的问题:\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("问题 > ").strip()
            
            # 处理空输入
            if not user_input:
                continue
            
            # 处理退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n感谢使用！再见！👋\n")
                break
            
            # 处理帮助命令
            if user_input.lower() == 'help':
                display_help()
                continue
            
            # 执行查询
            print("\n正在思考... 🤔\n")
            
            try:
                result = query(user_input, retriever=retriever)
                print_result(result)
            except Exception as e:
                print(f"\n✗ 查询出错: {e}")
                print("请重试或输入 'help' 查看帮助信息。\n")
                continue
            
        except KeyboardInterrupt:
            # 处理 Ctrl+C
            print("\n\n检测到中断信号。正在退出...\n")
            break
        except EOFError:
            # 处理 Ctrl+D (Unix) 或 Ctrl+Z (Windows)
            print("\n\n检测到 EOF。正在退出...\n")
            break
        except Exception as e:
            print(f"\n✗ 发生错误: {e}")
            print("系统将继续运行，请重试。\n")
            continue


def main() -> int:
    """
    主函数
    
    Returns:
        退出状态码 (0 表示成功)
    """
    # 显示欢迎信息
    display_welcome()
    
    # 验证配置
    config = get_config()
    try:
        config.validate()
        print("✓ 配置验证通过\n")
    except ValueError as e:
        print(f"✗ 配置错误: {e}")
        print("\n请检查 .env 文件并确保所有必需的配置项已设置。")
        return 1
    
    # 初始化 RAG 系统
    print("正在初始化 RAG 系统...\n")
    
    try:
        # 加载向量库
        vectorstore = load_vectorstore()
        
        # 创建检索器
        retriever = get_retriever(vectorstore)
        
        print("✓ RAG 系统初始化完成\n")
        print("-" * 80 + "\n")
        
    except FileNotFoundError as e:
        print(f"✗ 初始化失败: {e}")
        print("\n请先运行 'python src/build_index.py' 构建索引。")
        return 1
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 运行交互循环
    try:
        run_interactive_loop(retriever)
    except Exception as e:
        print(f"\n✗ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
