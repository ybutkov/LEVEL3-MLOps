"""Holiday lookup for the serving layer — a stub standing in for a real source.

``is_holiday`` is a model feature, but the API caller shouldn't have to know the
holiday calendar. This repository hides that lookup behind one method, so the
FastAPI layer asks "is this date a holiday?" without caring how it's answered.
The in-memory set here is a placeholder; a production impl would query a holiday
service or DB (e.g. by subclassing and overriding ``is_holiday``) — callers
don't change.
"""

from datetime import date


class HolidayRepository:
    """Answer whether a date is a public holiday. Stub: hardcoded sample set."""

    # Placeholder calendar (a few US holidays), incl. 2013 so future-dated
    # forecasts resolve. Replace with a DB/service lookup; the contract stays.
    _HOLIDAYS: frozenset[date] = frozenset({
        date(2011, 1, 1), date(2011, 7, 4), date(2011, 11, 24), date(2011, 12, 25),
        date(2012, 1, 1), date(2012, 7, 4), date(2012, 11, 22), date(2012, 12, 25),
        date(2013, 1, 1), date(2013, 7, 4), date(2013, 11, 28), date(2013, 12, 25),
    })

    def is_holiday(self, day: date) -> bool:
        """Return True if ``day`` is a public holiday."""
        return day in self._HOLIDAYS
