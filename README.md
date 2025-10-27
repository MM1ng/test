# KNN: 基于K-最近邻算法的机器学习应用

基于scikit-learn的K-最近邻(KNN)分类器，支持模型训练、预测、评估和持久化。

## 🌟 项目特性

- **KNN算法实现**: 使用scikit-learn实现K-最近邻分类器
- **环境配置**: 通过.env文件进行配置管理
- **代码质量**: Black、isort、Flake8、MyPy代码检查
- **测试覆盖**: Pytest单元测试和集成测试
- **Docker支持**: Docker容器化部署
- **CI/CD**: GitHub Actions自动化流水线
- **MLflow集成**: 实验跟踪和模型管理
- **DagsHub支持**: 与DagsHub平台集成

## 📁 项目结构

```
github-workflow-exercize/
├── app/
│   ├── __init__.py
│   └── app.py              # KNN模型和主要功能
├── tests/
│   └── test_app.py         # 单元测试
├── .github/
│   └── workflows/
│       ├── ci.yml          # CI/CD流水线
│       └── security.yml    # 安全检查流水线
├── .env.example            # 环境变量示例
├── .gitignore              # Git忽略文件
├── .dockerignore           # Docker忽略文件
├── .flake8                 # Flake8配置
├── pyproject.toml          # 项目配置
├── requirements.txt        # Python依赖
├── Dockerfile              # Docker配置
├── Dockerfile.dagshub      # DagsHub Docker配置
└── README.md               # 项目说明文档
```

## ⚙️ 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `KNN_NEIGHBORS` | 3 | KNN邻居数量 |
| `TEST_SIZE` | 0.2 | 测试集比例 |
| `USE_SYNTHETIC_DATA` | false | 是否使用合成数据 |
| `MODEL_PATH` | model.pkl | 模型保存路径 |
| `DEBUG` | false | 调试模式 |
| `LOGLEVEL` | INFO | 日志级别 |
| `MLFLOW_TRACKING_URI` | mlruns | MLflow跟踪URI |

## 🚀 快速开始

### 1. **克隆项目**
```bash
git clone https://github.com/MM1ng/Final-Project.git
cd github
```

### 2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 3. **安装依赖**
```bash
pip install -r requirements.txt
```

### 4. **配置环境变量**
```bash
cp .env.example .env
# 编辑.env文件以修改配置
```

### 5. **运行应用**
```bash
# 方式1: 直接运行
python run.py

# 方式2: 模块方式运行
python -m app.app

# 方式3: 安装为包后运行
pip install -e .
knn-ml-app
```

## 🐳 Docker支持

### 1. **构建镜像**
```bash
docker build -t knn-ml-app .
```

### 2. **运行容器**
```bash
run --name knn-app knn-ml-app
```

## 🧪 测试

### 运行测试
```bash
pytest
```

### 生成测试覆盖率报告
```bash
pytest --cov=app --cov-report=html
```

## 🎨 代码质量

### 格式化代码
```bash
# 使用Black格式化
black .

# 使用isort整理导入
isort .
```

### 代码检查
```bash
# 使用Flake8检查
flake8 .
```

### 静态类型检查
```bash
# 使用MyPy进行类型检查
mypy .
```

## 🔒 安全检查

### 依赖安全检查
```bash
# 使用Safety检查
safety check

# 使用Bandit检查
bandit -r app/
```

## 📊 MLflow与DagsHub集成

本项目已集成MLflow用于实验跟踪和模型管理，并支持与DagsHub平台集成。

### 配置DagsHub

1. 在[DagsHub](https://dagshub.com/)上创建一个新项目
2. 获取您的DagsHub MLflow跟踪URI，格式为: `https://dagshub.com/用户名/项目名.mlflow`
3. 在您的.env文件中设置:
   ```
   MLFLOW_TRACKING_URI=https://dagshub.com/用户名/项目名.mlflow
   MLFLOW_TRACKING_USERNAME=您的用户名
   MLFLOW_TRACKING_PASSWORD=您的DagsHub访问令牌
   ```

### 使用MLflow跟踪实验

运行应用时，MLflow会自动记录实验参数和指标。您可以使用以下命令查看跟踪结果:

```bash
# 启动MLflow UI
mlflow ui
```

然后在浏览器中访问 http://localhost:5000 查看实验结果。

### DagsHub Docker支持

项目包含一个专门用于DagsHub的Dockerfile (Dockerfile.dagshub)，其中包含了git支持，以便与DagsHub协作。

## 🔄 CI/CD流水线

GitHub Actions自动化流水线包括:

### 1. CI/CD Pipeline (`.github/workflows/ci.yml`)
- **代码检查**: Black、isort、Flake8、MyPy
- **测试运行**: Python 3.8-3.11
- **Docker构建**: 构建和测试Docker镜像
- **安全检查**: Safety、Bandit

### 2. 安全检查流水线 (`.github/workflows/security.yml`)
- **依赖安全检查**: 使用Safety进行依赖安全检查
- **代码安全检查**: Bandit和Semgrep代码安全检查
- **SAFETY SARIF报告**: 安全检查的SARIF格式报告

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。