import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 将父目录添加到sys.path中
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E501

from app import KNNModel, create_synthetic_data, load_sample_data  # noqa: E402


class TestKNNModel:
    """KNN模型测试类"""

    def test_model_initialization(self):
        """测试模型初始化"""
        model = KNNModel(n_neighbors=5)
        assert model.n_neighbors == 5
        assert not model.is_trained

    def test_model_initialization_default(self):
        """测试模型默认初始化"""
        model = KNNModel()
        assert model.n_neighbors == 3
        assert not model.is_trained

    def test_train_model(self):
        """测试模型训练"""
        model = KNNModel(n_neighbors=3)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        model.train(X, y)
        assert model.is_trained

    def test_predict_without_training(self):
        """测试未训练模型预测时抛出异常"""
        model = KNNModel()
        X = np.array([[1, 2]])

        with pytest.raises(ValueError, match="模型尚未训练"):
            model.predict(X)

    def test_predict_proba_without_training(self):
        """测试未训练模型预测概率时抛出异常"""
        model = KNNModel()
        X = np.array([[1, 2]])

        with pytest.raises(ValueError, match="模型尚未训练"):
            model.predict_proba(X)

    def test_save_model_without_training(self):
        """测试未训练模型保存时抛出异常"""
        model = KNNModel()

        with pytest.raises(ValueError, match="模型尚未训练"):
            model.save_model("test_model.pkl")

    def test_full_training_and_prediction_workflow(self):
        """测试完整的训练和预测工作流"""
        # 创建测试数据
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        # 训练模型
        model = KNNModel(n_neighbors=2)
        model.train(X, y)

        # 进行预测
        test_X = np.array([[2, 3], [6, 7]])
        predictions = model.predict(test_X)
        probabilities = model.predict_proba(test_X)

        # 验证预测结果
        assert len(predictions) == 2
        assert predictions.shape == (2,)
        assert probabilities.shape == (2, 2)  # 2个样本，2个类别
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # 概率和为1

    @patch("joblib.dump")
    def test_save_model(self, mock_dump):
        """测试模型保存"""
        model = KNNModel()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        model.train(X, y)
        model.save_model("test_model.pkl")

        mock_dump.assert_called_once()

    @patch("joblib.load")
    def test_load_model(self, mock_load):
        """测试模型加载"""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        model = KNNModel()
        model.load_model("test_model.pkl")

        assert model.is_trained
        mock_load.assert_called_once_with("test_model.pkl")


class TestDataFunctions:
    """数据加载函数测试类"""

    def test_load_sample_data(self):
        """测试加载示例数据"""
        X, y = load_sample_data()

        assert X.shape[1] == 4  # iris数据集有4个特征
        assert len(np.unique(y)) == 3  # iris数据集有3个类别
        assert X.shape[0] == y.shape[0]  # 特征和标签数量一致

    def test_create_synthetic_data_default(self):
        """测试创建默认合成数据"""
        X, y = create_synthetic_data()

        assert X.shape[0] == 1000  # 默认1000个样本
        assert X.shape[1] == 4  # 默认4个特征
        assert len(np.unique(y)) == 3  # 默认3个类别

    def test_create_synthetic_data_custom(self):
        """测试创建自定义合成数据"""
        X, y = create_synthetic_data(n_samples=500, n_features=6, n_classes=4)

        assert X.shape[0] == 500
        assert X.shape[1] == 6
        assert len(np.unique(y)) == 4

    def test_synthetic_data_consistency(self):
        """测试合成数据的一致性（相同参数应产生相同数据）"""
        X1, y1 = create_synthetic_data(n_samples=100, random_state=42)
        X2, y2 = create_synthetic_data(n_samples=100, random_state=42)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestIntegration:
    """集成测试类"""

    @patch("os.getenv")
    def test_main_function_with_env_vars(self, mock_getenv):
        """测试主函数使用环境变量"""
        # 模拟环境变量
        mock_getenv.side_effect = lambda key, default=None: {
            "KNN_NEIGHBORS": "5",
            "TEST_SIZE": "0.3",
            "USE_SYNTHETIC_DATA": "true",
            "MODEL_PATH": "test_model.pkl",
        }.get(key, default)

        pass

    def test_model_performance_on_iris(self):
        """测试模型在iris数据集上的性能"""
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        # 加载iris数据
        iris = load_iris()
        X, y = iris.data, iris.target

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练模型
        model = KNNModel(n_neighbors=3)
        model.train(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # 验证准确率（iris数据集上KNN通常表现很好）
        assert accuracy > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
