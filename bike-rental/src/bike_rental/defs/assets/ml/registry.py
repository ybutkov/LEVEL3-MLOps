"""Model-selection domain: a scored registry candidate and the promotion policy.

Pure domain objects with no MLflow dependency. Callers (the week-4 notebook or a
Dagster asset) build :class:`Candidate` instances from MLflow runs and apply a
:class:`PromotionPolicy` to decide which registered version becomes champion.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A registered model version together with its evaluation metrics.

    Champion and challenger are both ``Candidate`` instances: the role is an
    MLflow alias, not a separate type.

    Parameters
    ----------
    version : str
        Registry version number of the model.
    model_type : str
        Short model tag (e.g. ``"linear"``, ``"rf"``, ``"hgb"``).
    metrics : dict[str, float]
        Evaluation metrics keyed by name (must contain the policy metric).
    run_id : str | None, default=None
        Source MLflow run id, for provenance.
    """

    version: str
    model_type: str
    metrics: dict[str, float]
    run_id: str | None = None


@dataclass(frozen=True)
class PromotionPolicy:
    """Decide whether one candidate should replace another.

    Single-metric by default. Multi-metric, weighted, or tie-break rules are
    grown by overriding :meth:`delta` / :meth:`is_better` only — neither
    :class:`Candidate` nor the promotion gate changes.

    Parameters
    ----------
    metric : str, default="rmse"
        Metric key compared (must exist in ``Candidate.metrics``).
    higher_is_better : bool, default=False
        Direction of the metric (``True`` for r2, ``False`` for rmse/mae).
    margin : float, default=0.0
        Minimum improvement a challenger must show to unseat the champion;
        guards against re-crowning on noise.
    """

    metric: str = "rmse"
    higher_is_better: bool = False
    margin: float = 0.0

    def delta(self, a: Candidate, b: Candidate) -> float:
        """Signed improvement of ``a`` over ``b``; ``> 0`` iff ``a`` is better.

        Parameters
        ----------
        a, b : Candidate
            Candidates to compare.

        Returns
        -------
        float
            Direction-adjusted difference of the policy metric.
        """
        d = a.metrics[self.metric] - b.metrics[self.metric]
        return d if self.higher_is_better else -d

    def is_better(self, a: Candidate, b: Candidate) -> tuple[bool, float]:
        """Whether ``a`` should replace ``b``: improvement strictly above margin.

        Parameters
        ----------
        a, b : Candidate
            Challenger and incumbent, respectively.

        Returns
        -------
        tuple[bool, float]
            ``(verdict, delta)`` — the decision and the signed improvement.
        """
        d = self.delta(a, b)
        return d > self.margin, d

    def best(self, candidates: list[Candidate]) -> Candidate:
        """Best of a batch by relative order (``margin`` not applied within a batch).

        Parameters
        ----------
        candidates : list[Candidate]
            Non-empty list of trained candidates.

        Returns
        -------
        Candidate
            The single best candidate under this policy.
        """
        winner = candidates[0]
        for c in candidates[1:]:
            if self.delta(c, winner) > 0:
                winner = c
        return winner
