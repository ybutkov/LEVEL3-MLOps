"""Estimator wrappers that bake serving constraints into the model itself.

Keeping a constraint (here: non-negativity) *inside* the fitted estimator means
the saved/served Pipeline is self-contained — no caller needs to re-apply it.
Same principle as putting cyclic encoding in the Pipeline rather than upstream.
"""

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone


class NonNegativeRegressor(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    """Wrap a regressor so its predictions are clipped at zero.

    The target (rental counts) is non-negative, but an unconstrained regressor
    can predict below zero for low-demand hours. Wrapping bakes the clip into
    the model, so a saved/served pipeline never returns negatives — FastAPI and
    batch scoring stay correct without re-clipping.

    Model-agnostic: works for ``LinearRegression``, ``RandomForestRegressor``,
    ``HistGradientBoostingRegressor`` alike.

    Parameters
    ----------
    estimator : BaseEstimator
        The regressor to wrap. Cloned on ``fit`` so the passed instance is
        left untouched; the fitted copy lives in ``estimator_``.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return np.clip(self.estimator_.predict(X), 0, None)
