from collections import Counter

import numpy as np


class MajorityClassifier:
    """A deliberately simple baseline for a classification task."""

    def fit(self, y):
        counts = Counter(int(value) for value in y)
        if not counts:
            raise ValueError("cannot fit on empty labels")
        self.label_ = counts.most_common(1)[0][0]
        return self

    def predict(self, rows):
        if not hasattr(self, "label_"):
            raise RuntimeError("fit must be called before predict")
        return [self.label_] * len(rows)


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
