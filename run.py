import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    # 检查是否存在.env文件
    env_file = project_root / ".env"
    if not env_file.exists():
        print("警告: 未找到.env文件，使用默认配置")
        print("请复制env.example为.env并根据需要修改配置")

    # 设置MLflow环境变量（如果在DagsHub环境中）
    # 用户可以在.env文件中设置MLFLOW_TRACKING_URI指向他们的DagsHub仓库
    if os.getenv("MLFLOW_TRACKING_URI"):
        print(f"使用MLflow跟踪URI: {os.getenv('MLFLOW_TRACKING_URI')}")

    # 导入并运行主应用
    try:
        from app import main

        main()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"运行错误: {e}")
        sys.exit(1)