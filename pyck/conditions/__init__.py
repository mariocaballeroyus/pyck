"""Boundary and load conditions."""

from pyck.conditions.condition import Condition
from pyck.conditions.dirichlet import assign_scalar, assign_zeros
from pyck.conditions.load_condition import LoadCondition, create_load_condition


__all__ = [
    "Condition",
    "LoadCondition",
    "create_load_condition",
    "assign_scalar",
    "assign_zeros",
]
