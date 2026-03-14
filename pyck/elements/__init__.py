"""Finite element formulations."""

from pyck.elements.element import Element
from pyck.elements.euler_bernoulli_beam import EulerBernoulliBeam, create_euler_bernoulli_beam
from pyck.elements.timoshenko_beam_1p import TimoshenkoBeam1P
from .timoshenko_beam_2p import TimoshenkoBeam2P, create_timoshenko_beam

def create_rotation_free_timoshenko_beam(
    E: float, A: float, I: float, G: float, k: float = 5.0 / 6.0
) -> TimoshenkoBeam1P:
    return TimoshenkoBeam1P(E=float(E), A=float(A), I=float(I), G=float(G), k=float(k))



__all__ = [
    "Element",
    "EulerBernoulliBeam",
    "TimoshenkoBeam1P",
    "TimoshenkoBeam2P",
    "create_euler_bernoulli_beam",
    "create_timoshenko_beam",
    "create_rotation_free_timoshenko_beam",
]
