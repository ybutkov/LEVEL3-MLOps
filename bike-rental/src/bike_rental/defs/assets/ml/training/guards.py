"""Runtime validation guards for model training assets."""

import dagster as dg

from bike_rental.defs import schemas


def assert_no_target_leak(features: list[str], target: str) -> None:
    """Fail if a target-related column other than `target` is used as a feature.

    `total_rentals = registered_rentals + direct_pickups`, so whenever one of the
    three is the target, the other two leak it and must not be features. The set
    of forbidden columns therefore depends on the chosen target.
    """
    target_related = {schemas.TARGET, *schemas.TARGET_COMPONENTS}
    leaked = sorted((target_related - {target}) & set(features))
    if leaked:
        raise dg.Failure(
            description=(
                f"Target leak: {leaked} are components of/derived from target "
                f"'{target}' and must not appear in features."
            ),
            metadata={
                "target": dg.MetadataValue.text(target),
                "leaked_features": dg.MetadataValue.text(", ".join(leaked)),
            },
        )
