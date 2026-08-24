# 项目提交说明

> 项目提交是所选项目的正式验收包；它与可以包含多个任务的实习经历记录分开。Git/CI 会由导师从远端记录核对，学生不需要在这里复制原始日志或任何凭据。

## 项目

- 项目 ID：M01
- Assignment ID（只填 pseudonymous ID）：待填写（请填自己的 pseudonymous ID）

## 我完成的 Core 内容

- 复现 baseline：多数类预测器，测试集 accuracy = 0.50。
- 在 `model.py` 中实现自建学生模型 `StudentLogisticRegression`：用 NumPy 从零实现逻辑回归，不使用 sklearn 的现成模型。
- 使用相同 5 折交叉验证划分对比 baseline 与 student：baseline 平均 accuracy = 0.40，student 平均 accuracy = 1.00。
- 完成错误分析：baseline 共 7 条错误样本，归纳为 2 类错误类型。
- 公开测试全部通过。

核心模型代码如下：

```python
class StudentLogisticRegression:
 

    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000, seed: int = 0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights_ = None
        self.mean_ = None
        self.std_ = None

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -20, 20)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, features, labels):
        X = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float).reshape(-1)
        if X.shape[0] == 0:
            raise ValueError("cannot fit on empty features")

        # Standardize using training-set statistics only.
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_

        # Add a bias term.
        Xs = np.column_stack([np.ones(Xs.shape[0]), Xs])

        rng = np.random.default_rng(self.seed)
        self.weights_ = rng.normal(0, 0.01, size=Xs.shape[1])

        for _ in range(self.epochs):
            probabilities = self._sigmoid(Xs @ self.weights_)
            gradient = Xs.T @ (probabilities - y) / len(y)
            self.weights_ -= self.learning_rate * gradient

        return self

    def predict(self, features):
        if self.weights_ is None:
            raise RuntimeError("fit must be called before predict")

        X = np.asarray(features, dtype=float)
        Xs = (X - self.mean_) / self.std_
        Xs = np.column_stack([np.ones(Xs.shape[0]), Xs])
        probabilities = self._sigmoid(Xs @ self.weights_)
        return (probabilities >= 0.5).astype(int).tolist()


def build_student_model():
    """Student Core: return a hand-written logistic regression model."""
    return StudentLogisticRegression(learning_rate=0.1, epochs=1000, seed=0)
```
## 运行方法

```text
cd starter-projects/intermediate/M01-tabular-ml
$env:PYTHONPATH = "src"
python -m tabular_ml.run --model baseline
python -m tabular_ml.run --model student
python -m unittest discover -s tests -v
```

## 验证结果

### Baseline（测试集）

```json
{
  "accuracy": 0.5,
  "confusion_matrix": [[0, 2], [0, 2]],
  "count": 4
}
```

### Student（测试集）

```json
{
  "accuracy": 1.0,
  "confusion_matrix": [[2, 0], [0, 2]],
  "count": 4
}
```

### 公开测试

```text
Ran 2 tests in 0.003s

OK
```

### 5 折交叉验证

- 划分方式：`KFold(n_splits=5, shuffle=True, random_state=0)`
- baseline 平均 accuracy：0.40
- student 平均 accuracy：1.00
- baseline 错误样本：7 条
- student 错误样本：0 条

## 项目提交证据

- 代码入口：`starter-projects/intermediate/M01-tabular-ml/src/tabular_ml/`
  - `data.py`：数据加载与字段校验
  - `model.py`：baseline 与自建 `StudentLogisticRegression`
  - `evaluate.py`：accuracy + 混淆矩阵
  - `run.py`：命令行训练与评估入口
- 测试命令：
  ```text
  $env:PYTHONPATH = "src"
  python -m tabular_ml.run --model baseline
  python -m tabular_ml.run --model student
  python -m unittest discover -s tests -v
  ```
- Notebook 或实验入口：`notebooks/student/intermediate/M01-tabular-ml.ipynb`
- Pull Request URL（如有）：待填写
- CI 状态或运行 URL（如有）：待填写
- 与本次提交关联的实习经历 Task ID：待填写

## 已知问题

1. 数据集非常小：训练集 13 条、测试集 4 条，student 的 accuracy = 1.0 不能说明模型具有普适性。
2. 交叉验证脚本目前是桌面临时文件（`C:\Users\33556\Desktop\cv_temp.py`），尚未迁移到项目内；正式提交前建议迁移到 `experiments/cv_baseline_student.py`，以保证实验可复现。
3. student 在测试集和交叉验证中都没有错误样本，因此错误分析主要基于 baseline 的错误样本。

## 我学到的内容

- 先复现 baseline，再通过受控对比实验验证改进是否真实有效。
- 模型比较必须使用相同的数据划分和相同的评估指标。
- 从零实现逻辑回归时，需要自己处理特征标准化、bias 项、梯度下降和预测阈值。
- 预处理统计量只能在训练集上拟合，否则会造成数据泄漏。
- 小样本下的高 accuracy 可能来自数据规律清晰，不能直接推断真实场景表现。

## 如果再给我一天

- 将交叉验证脚本从桌面迁移到项目内，并整理成可复现的实验入口。
- 补充至少三个单元测试：正常场景、边界场景、失败场景。
- 把交叉验证结果和错误分析整理进 Notebook 或实验报告。
- 尝试增加交叉验证折数（如留一法）验证稳定性。

## 给导师的复盘

以下内容由学生先填写，导师审核后才可能进入受限的导师记录；不要填写分数、岗位、人才库、留用或其他内部评价。

### 学员收获

- 完成了“复现 baseline → 自建模型 → 受控对比 → 错误分析”的完整实验闭环。
- 更清楚小样本机器学习实验的局限性和可复现实验的重要性。

### 建议

- 当前数据量较小，建议后续增加更多样本或使用留一法进一步验证模型稳定性。

### 推荐

- 课程 / 书籍 / 爱好：待填写
- 推荐理由：待填写
