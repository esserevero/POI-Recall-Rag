"""
配置模块
负责环境变量读取和常量定义
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """系统配置类"""
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # DeepSeek API 配置（用于 LLM）
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    
    # 模型配置
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")  # 本地嵌入模型
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
    
    # 路径配置
    CHROMA_PERSIST_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "db")
    SUBTITLES_DIR = PROJECT_ROOT / os.getenv("SUBTITLES_DIR", "data")
    EXAMPLES_DIR = PROJECT_ROOT / os.getenv("EXAMPLES_DIR", "data/examples")
    
    # 分块配置
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    
    # 检索配置
    RETRIEVAL_K = 5  # 检索 top-k 文档
    RETRIEVAL_SCORE_THRESHOLD = 0.3  # 最小相似度阈值
    
    # LLM 配置
    LLM_TEMPERATURE = 0.7  # 生成温度（平衡创造性和准确性）
    LLM_MAX_TOKENS = 1000  # 最大生成 token 数
    
    @classmethod
    def validate(cls):
        """验证配置是否完整"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置。LLM 需要 DeepSeek API Key。"
            )
        
        # 确保目录存在
        cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        cls.SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)
        cls.EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        
        return True


def get_config() -> Config:
    """获取配置对象"""
    return Config


if __name__ == "__main__":
    # 测试配置
    config = get_config()
    print(f"项目根目录: {config.PROJECT_ROOT}")
    print(f"字幕目录: {config.SUBTITLES_DIR}")
    print(f"向量库目录: {config.CHROMA_PERSIST_DIR}")
    print(f"嵌入模型: {config.EMBEDDING_MODEL}")
    print(f"LLM 模型: {config.LLM_MODEL}")
    
    try:
        config.validate()
        print("\n✓ 配置验证通过")
    except ValueError as e:
        print(f"\n✗ 配置验证失败: {e}")
