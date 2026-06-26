"""Scratch: LinearRegression baseline vs Ridge (improvement #1) on validation."""
import numpy as np
import runpy

# reuse dataset+split+preprocess from split3_proto by executing it and grabbing globals
g = runpy.run_path("../_scratch/split3_proto.py")
X_train, y_train, X_val, y_val = g["X_train"], g["y_train"], g["X_val"], g["y_val"]
pre, evaluate = g["pre"], g["evaluate"]

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


def make(est, log=False):
    pipe = Pipeline([("pre", pre), ("model", est)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1) if log else pipe


print("\n=== BASELINE LinearRegression ===")
for name, log in [("raw", False), ("log1p", True)]:
    m = make(LinearRegression(), log); m.fit(X_train, y_train)
    print(f"LinReg {name:6}", evaluate(y_val, np.clip(m.predict(X_val), 0, None)))

print("\n=== improvement #1: Ridge vs LinearRegression (raw) ===")
for name, est in [("LinearRegression", LinearRegression()), ("Ridge(alpha=1)", Ridge(alpha=1.0)),
                  ("Ridge(alpha=10)", Ridge(alpha=10.0))]:
    m = make(est); m.fit(X_train, y_train)
    print(f"{name:18}", evaluate(y_val, np.clip(m.predict(X_val), 0, None)))
