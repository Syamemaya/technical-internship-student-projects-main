# 项目提交说明

> 项目提交是所选项目的正式验收包；它与可以包含多个任务的实习经历记录分开。Git/CI 会由导师从远端记录核对，学生不需要在这里复制原始日志或任何凭据。

## 项目

- 项目 ID：M01
- Assignment ID（只填 pseudonymous ID）：M01

## 我完成的 Core 内容
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
$env:PYTHONPATH = "src"
>> python -m tabular_ml.run --model student
>> python -m unittest discover -s tests -v
```

## 验证结果

## 项目提交证据

- 代码入口：
- 测试命令：
- Notebook 或实验入口：
- Pull Request URL（如有）：
- CI 状态或运行 URL（如有）：
- 与本次提交关联的实习经历 Task ID：

## 已知问题

## 我学到的内容

## 如果再给我一天

## 给导师的复盘

以下内容由学生先填写，导师审核后才可能进入受限的导师记录；不要填写分数、岗位、人才库、留用或其他内部评价。

### 学员收获

-

### 建议

-

### 推荐

- 课程 / 书籍 / 爱好：
- 推荐理由：
