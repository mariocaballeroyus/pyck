"""Quadrature rules for numerical integration."""

from pyck.quadrature.quadrature import QuadratureRule
from pyck.quadrature.gauss_legendre import GaussLegendre

__all__ = [
    "GaussLegendre",
    "QuadratureRule",
]
