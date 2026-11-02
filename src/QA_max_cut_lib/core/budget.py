"""Track one strict shot budget for a complete solver run."""

from __future__ import annotations

from numbers import Integral

__all__ = ["ShotBudget", "ShotBudgetExceeded"]


class ShotBudgetExceeded(RuntimeError):
    """Report a requested shot amount that does not fit the remaining budget.

    Parameters
    ----------
    requested : int
        Number of shots requested by the component.
    available : int or None
        Number of shots still available before the request.  None means
        that no total limit exists.
    """

    def __init__(self, requested, available):
        self.requested = int(requested)
        self.available = available
        super().__init__(
            f"Shot budget exceeded: requested {self.requested}, "
            f"available {self.available}."
        )


class ShotBudget:
    """Track strict shot use for one root solver run.

    Parameters
    ----------
    total : int or None, default None
        Maximum number of shots.  None means no total limit.

    Attributes
    ----------
    total : int or None
        The configured limit.
    used : int
        Shots already spent by completed measurements.

    Notes
    -----
    A caller can reserve shots by passing ``reserved`` to ``can_spend``.
    A reservation is not a charge.  The caller charges the reserved
    amount only after it performs the measurement.  This keeps training
    from consuming shots that a final readout or correlation sample needs.
    """

    def __init__(self, total=None):
        if total is not None:
            if isinstance(total, bool) or not isinstance(total, Integral):
                raise TypeError("Shot budget total must be an integer or None.")
            total = int(total)
            if total < 0:
                raise ValueError("Shot budget total must be nonnegative.")
        self._total = None if total is None else int(total)
        self._used = 0

    @property
    def total(self):
        """Return the configured limit, or None."""
        return self._total

    @property
    def used(self):
        """Return the number of spent shots."""
        return self._used

    @property
    def remaining(self):
        """Return total - used, or None when no limit exists."""
        return None if self._total is None else self._total - self._used

    def available(self, reserved=0):
        """Return shots available after a temporary reservation.

        Parameters
        ----------
        reserved : int, default 0
            Shots that must stay available for a later measurement.

        Returns
        -------
        int or None
            Available shots, or None when no total limit exists.
        """
        reserved = self._check_amount(reserved, "reserved")
        if self._total is None:
            return None
        return self._total - self._used - reserved

    def can_spend(self, amount, reserved=0):
        """Return True if a measurement of ``amount`` fits the budget."""
        amount = self._check_amount(amount, "amount")
        reserved = self._check_amount(reserved, "reserved")
        if self._total is None:
            return True
        return self._used + amount + reserved <= self._total

    def spend(self, amount, reserved=0):
        """Charge one completed measurement and return the new used count.

        Raises
        ------
        ShotBudgetExceeded
            The charge would leave less than the reservation.
        """
        amount = self._check_amount(amount, "amount")
        reserved = self._check_amount(reserved, "reserved")
        if not self.can_spend(amount, reserved):
            available = self.available(reserved)
            raise ShotBudgetExceeded(amount, available)
        self._used += amount
        return self._used

    def snapshot(self):
        """Return a JSON-compatible budget record."""
        return {
            "total": self._total,
            "used": self._used,
            "remaining": self.remaining,
        }

    @staticmethod
    def _check_amount(value, name):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer.")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be nonnegative.")
        return value

    def __repr__(self):
        return f"ShotBudget(total={self._total!r}, used={self._used!r})"
