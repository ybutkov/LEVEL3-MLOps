import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.logreg import (
    accuracy,
    binary_cross_entropy,
    logistic_regression_gd,
    predict,
    sigmoid,
)


class TestSigmoid:
    def test_zero_input(self):
        assert sigmoid(np.array([0.0])) == pytest.approx(0.5)

    def test_large_positive(self):
        assert sigmoid(np.array([1000.0])) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert sigmoid(np.array([-1000.0])) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        # sigmoid(1) = e / (1 + e)
        assert sigmoid(np.array([1.0])) == pytest.approx(np.e / (1 + np.e), rel=1e-6)

    def test_output_range(self):
        z = np.linspace(-10, 10, 100)
        out = sigmoid(z)
        assert np.all(out > 0) and np.all(out < 1)

    def test_clipping_does_not_raise(self):
        z = np.array([-1e9, 0.0, 1e9])
        out = sigmoid(z)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


class TestBinaryCrossEntropy:
    def test_perfect_predictions(self):
        y = np.array([[1.0], [0.0]])
        y_hat = np.array([[1.0], [0.0]])
        assert binary_cross_entropy(y, y_hat) == pytest.approx(0.0, abs=1e-6)

    def test_worst_predictions(self):
        # predicting opposite of truth should yield high loss
        y = np.array([[1.0], [0.0]])
        y_hat = np.array([[0.0], [1.0]])
        assert binary_cross_entropy(y, y_hat) > 10.0

    def test_uniform_half(self):
        # predicting 0.5 for all gives -log(0.5) ≈ 0.693
        y = np.array([[1.0], [0.0]])
        y_hat = np.array([[0.5], [0.5]])
        assert binary_cross_entropy(y, y_hat) == pytest.approx(np.log(2), rel=1e-6)

    def test_always_non_negative(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=(50, 1)).astype(float)
        y_hat = rng.uniform(0.01, 0.99, size=(50, 1))
        assert binary_cross_entropy(y, y_hat) >= 0.0

    def test_numerical_stability_at_zero(self):
        # y_hat = 0 should not produce nan/inf due to clipping
        y = np.array([[1.0]])
        y_hat = np.array([[0.0]])
        loss = binary_cross_entropy(y, y_hat)
        assert not np.isnan(loss) and not np.isinf(loss)


class TestLogisticRegressionGD:
    def _make_linearly_separable(self):
        """Two clearly separated clusters."""
        rng = np.random.default_rng(42)
        X0 = rng.normal(loc=-2.0, scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=+2.0, scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.vstack([np.zeros((50, 1)), np.ones((50, 1))])
        return X, y

    def test_loss_decreases(self):
        X, y = self._make_linearly_separable()
        _, _, history = logistic_regression_gd(X, y, lr=0.1, max_iter=200)
        assert history[-1] < history[0]

    def test_returns_correct_shapes(self):
        X, y = self._make_linearly_separable()
        w, b, history = logistic_regression_gd(X, y, lr=0.1, max_iter=50)
        assert w.shape == (2, 1)
        assert isinstance(b, float)
        assert len(history) > 0

    def test_high_accuracy_on_separable_data(self):
        X, y = self._make_linearly_separable()
        w, b, _ = logistic_regression_gd(X, y, lr=0.1, max_iter=500)
        preds = predict(X, w, b)
        assert accuracy(y, preds) > 0.95

    def test_converges_early_with_tight_tol(self):
        X, y = self._make_linearly_separable()
        _, _, history_tight = logistic_regression_gd(X, y, lr=0.1, max_iter=1000, tol=1e-3)
        _, _, history_loose = logistic_regression_gd(X, y, lr=0.1, max_iter=1000, tol=1e-9)
        assert len(history_tight) < len(history_loose)


class TestPredict:
    def test_threshold_at_half(self):
        # w=0, b=0 → sigmoid(0)=0.5 → predict 1
        X = np.array([[1.0]])
        w = np.zeros((1, 1))
        assert predict(X, w, 0.0)[0, 0] == 1

    def test_positive_weight_predicts_one(self):
        X = np.array([[10.0]])
        w = np.array([[1.0]])
        assert predict(X, w, 0.0)[0, 0] == 1

    def test_negative_weight_predicts_zero(self):
        X = np.array([[10.0]])
        w = np.array([[-1.0]])
        assert predict(X, w, 0.0)[0, 0] == 0

    def test_output_is_binary(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(20, 3))
        w = rng.normal(size=(3, 1))
        preds = predict(X, w, 0.0)
        assert set(preds.flatten().tolist()).issubset({0, 1})


class TestAccuracy:
    def test_perfect(self):
        y = np.array([[1], [0], [1]])
        assert accuracy(y, y) == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = np.array([[1], [1], [0]])
        y_pred = np.array([[0], [0], [1]])
        assert accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_half_correct(self):
        y_true = np.array([[1], [1], [0], [0]])
        y_pred = np.array([[1], [0], [1], [0]])
        assert accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_flattens_shapes(self):
        # column vector vs row vector should still work
        y_true = np.array([[1], [0], [1]])
        y_pred = np.array([1, 0, 1])
        assert accuracy(y_true, y_pred) == pytest.approx(1.0)
