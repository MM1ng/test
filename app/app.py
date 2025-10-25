import os
from typing import Tuple

import joblib
import mlflow
import numpy as np
from dotenv import load_dotenv
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载环境变量
load_dotenv()


class KNNModel:
    """KNN分类器模型类"""

    def __init__(self, n_neighbors: int = 3):
        """
        初始化KNN模型

        Args:
            n_neighbors: 邻居数量，默认为3
        """
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练模型

        Args:
            X: 特征数据
            y: 标签数据
        """
        self.model.fit(X, y)
        self.is_trained = True
        print(f"模型训练完成，使用 {self.n_neighbors} 个邻居")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        进行预测

        Args:
            X: 待预测的特征数据

        Returns:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 待预测的特征数据

        Returns:
            预测概率
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        return self.model.predict_proba(X)

    def save_model(self, filepath: str) -> None:
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        joblib.dump(self.model, filepath)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        加载模型

        Args:
            filepath: 模型文件路径
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"模型已从 {filepath} 加载")


def load_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    加载示例数据集

    Returns:
        特征数据和标签数据
    """
    # 使用iris数据集作为示例
    iris = load_iris()
    return iris.data, iris.target


def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 4,
    n_classes: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建合成数据集

    Args:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量

    Returns:
        特征数据和标签数据
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=0,
        n_informative=n_features,
        random_state=random_state,
    )
    return X, y


def main():
    """主函数"""
    print("=== KNN机器学习应用 ===")

    # 从环境变量获取配置
    n_neighbors = int(os.getenv("KNN_NEIGHBORS", "3"))
    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    use_synthetic = os.getenv("USE_SYNTHETIC_DATA", "false").lower() == "true"

    # 启动MLflow运行
    with mlflow.start_run() as run:
        # 记录参数
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("use_synthetic", use_synthetic)

        print("配置参数:")
        print(f"- 邻居数量: {n_neighbors}")
        print(f"- 测试集比例: {test_size}")
        print(f"- 使用合成数据: {use_synthetic}")

        # 加载数据
        if use_synthetic:
            print("\n使用合成数据集...")
            X, y = create_synthetic_data()
        else:
            print("\n使用Iris数据集...")
            X, y = load_sample_data()

        print(f"数据集形状: {X.shape}")
        print(f"类别数量: {len(np.unique(y))}")

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")

        # 创建和训练模型
        model = KNNModel(n_neighbors=n_neighbors)
        model.train(X_train, y_train)

        # 进行预测
        y_pred = model.predict(X_test)

        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n模型准确率: {accuracy:.4f}")

        # 记录指标
        mlflow.log_metric("accuracy", accuracy)

        # 显示详细分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        # 准备输入示例和模型签名
        input_example = X_train[:5]  # 使用前5个训练样本作为输入示例
        
        # 记录模型到MLflow，包含输入示例以自动生成签名
        mlflow.sklearn.log_model(
            sk_model=model.model,
            artifact_path="model",
            input_example=input_example,
            conda_env={
                "name": "knn-env",
                "channels": ["conda-forge", "defaults"],
                "dependencies": [
                    "python=3.11",
                    "pip",
                    {
                        "pip": [
                            "scikit-learn==1.3.0",
                            "numpy==1.24.3",
                            "mlflow==3.5.1"
                        ]
                    }
                ]
            }
        )

        # 保存模型到本地
        model_path = os.getenv("MODEL_PATH", "model.pkl")
        model.save_model(model_path)

        # 示例预测
        print("\n示例预测:")
        sample_idx = 0
        sample_features = X_test[sample_idx : sample_idx + 1]
        sample_pred = model.predict(sample_features)[0]
        sample_proba = model.predict_proba(sample_features)[0]

        print(f"样本特征: {sample_features[0]}")
        print(f"预测类别: {sample_pred}")
        print(f"预测概率: {sample_proba}")

        # 记录一些额外的指标
        mlflow.log_metric("train_samples", X_train.shape[0])
        mlflow.log_metric("test_samples", X_test.shape[0])
        mlflow.log_metric("n_features", X_train.shape[1])


if __name__ == "__main__":
    # 设置MLflow跟踪URI（如果环境变量中指定了DagsHub URI则使用它）
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    
    # 禁用环境变量记录警告
    os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "false"
    
    main()