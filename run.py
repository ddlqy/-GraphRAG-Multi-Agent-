"""运行入口"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphRAG + Multi-Agent 校园智能体系统")
    parser.add_argument("--mode", choices=["streamlit", "api", "test"], default="streamlit",
                       help="运行模式")
    parser.add_argument("--port", type=int, default=8501, help="端口号")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    
    args = parser.parse_args()
    
    if args.mode == "streamlit":
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/app/streamlit_app.py",
            "--server.port", str(args.port),
            "--server.address", args.host
        ])
    elif args.mode == "api":
        import uvicorn
        uvicorn.run(
            "src.api.app:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level="info",
        )
    elif args.mode == "test":
        import pytest
        pytest.main(["-v", "tests/"])


if __name__ == "__main__":
    main()
