"""Boundary and load conditions."""

from pyck.conditions.condition import Condition
from pyck.conditions.load_condition import LoadCondition, create_load_condition


__all__ = [
    "LoadCondition",
    "create_load_condition",
]
