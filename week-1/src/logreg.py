import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """A numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def binary_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Compute the mean binary cross-entropy loss.  """
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


def logistic_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float, list[float]]:
    """Train logistic regression via gradient descent.

    Returns (weights, bias, loss_history).
    """
    n_samples, n_features = X.shape
    rev_n_samples = 1 / n_samples
    w = np.zeros((n_features, 1))
    b = 0.0
    loss_history = []
    X_T = X.T
    every_n_iters_print = max_iter // 10

    for iter in range(max_iter):
        Z = np.dot(X, w) + b
        A = sigmoid(Z)

        errors = A - y
        dw = rev_n_samples * np.dot(X_T, errors)
        db = rev_n_samples * np.sum(errors)

        w = w - lr * dw
        b = b - lr * db

        loss = binary_cross_entropy(y, A)
        loss_history.append(loss)

        if iter % every_n_iters_print == 0:
            print(f"Iter {iter:4d}: loss={loss:.6f}")
        # if iter > 0 and abs(loss_history[-2] - loss_history[-1]) < tol:
        #     break
        if iter > 0:
            diff = loss_history[-1] - loss_history[-2]
            if diff > 0:
                print(f"Warning: loss increased at iter {iter}. Reduce learning rate.")
                break
            if abs(diff) < tol:
                print(f"Converged at iteration {iter}")
                break

    return w, b, loss_history


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict class labels (0 or 1)."""
    return (sigmoid(np.dot(X, w) + b) >= 0.5).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    return np.mean(y_true.flatten() == y_pred.flatten())
