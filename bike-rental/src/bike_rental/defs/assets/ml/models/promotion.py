"""Champion promotion: the challenger-vs-champion gate as a Dagster asset.

Consumes the model assets' ``Candidate`` outputs directly (by lineage), so
challenger discovery never queries the shared registry — immune to other writers
and to run scoping. Picks the best challenger by the validation metric and
promotes it to the champion alias only if it beats the incumbent by more than
``margin`` (a bootstrap promotes the first ever champion). Selection is
validation-only — the test split is never consulted here, to avoid leaking it
into model selection.
"""

import dagster as dg

from bike_rental.defs.assets.ml.registry import Candidate, PromotionPolicy
from bike_rental.defs.resources.experiment_tracker import ExperimentTracker


class PromotionConfig(dg.Config):
    """Promotion policy knobs (editable per-run in the Launchpad)."""

    metric: str = "rmse"
    higher_is_better: bool = False
    margin: float = 0.0


@dg.asset(group_name="models", kinds={"mlflow"})
def champion(
    linear_hourly: Candidate,
    rf_hourly: Candidate,
    hgb_hourly: Candidate,
    experiment_tracker: ExperimentTracker,
    config: PromotionConfig,
) -> dg.MaterializeResult:
    """Promote the best challenger to champion/production if it beats the incumbent."""
    promotion_policy = PromotionPolicy(
        metric=config.metric,
        higher_is_better=config.higher_is_better,
        margin=config.margin,
    )
    candidates = [linear_hourly, rf_hourly, hgb_hourly]
    challenger = promotion_policy.best(candidates)
    incumbent_champion = experiment_tracker.load_champion()

    def promote(version: str) -> None:
        # champion = the new incumbent (next comparison); production = what the API serves
        experiment_tracker.set_champion(version)
        experiment_tracker.set_production(version)

    if incumbent_champion is None:
        promote(challenger.version)
        decision, detail = "bootstrap", f"champion set to v{challenger.version}"
    else:
        won, delta = promotion_policy.is_better(challenger, incumbent_champion)
        if won:
            promote(challenger.version)
            decision = "promoted"
            detail = (
                f"v{challenger.version} over v{incumbent_champion.version} "
                f"(Δ{promotion_policy.metric}={delta:+.4f})"
            )
        else:
            decision = "kept"
            detail = (
                f"champion v{incumbent_champion.version} "
                f"(Δ{promotion_policy.metric}={delta:+.4f} ≤ margin {promotion_policy.margin})"
            )

    return dg.MaterializeResult(
        metadata={
            "decision": dg.MetadataValue.text(decision),
            "detail": dg.MetadataValue.text(detail),
            "challenger": dg.MetadataValue.text(
                f"{challenger.model_type} v{challenger.version} "
                f"({promotion_policy.metric}={challenger.metrics.get(promotion_policy.metric)})"
            ),
            "candidates": dg.MetadataValue.json(
                [
                    {"model_type": c.model_type, "version": c.version,
                     promotion_policy.metric: c.metrics.get(promotion_policy.metric)}
                    for c in candidates
                ]
            ),
        },
    )
