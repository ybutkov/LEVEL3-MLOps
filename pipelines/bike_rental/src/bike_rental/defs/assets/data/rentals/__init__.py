"""Rental assets: row typing and hourly counts per location."""

from bike_rental.defs.assets.data.rentals.intermediate import rentals_split
from bike_rental.defs.assets.data.rentals.primary import hourly_rentals

__all__ = [
    "rentals_split",
    "hourly_rentals",
]
