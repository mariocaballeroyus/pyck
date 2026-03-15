"""Finite element formulations."""

from pyck.elements.element import Element
from pyck.elements.euler_bernoulli_beam import EulerBernoulliBeam, create_euler_bernoulli_beam
from pyck.elements.timoshenko_beam_1p import TimoshenkoBeam1p
from .timoshenko_beam_2p import TimoshenkoBeam2p, create_timoshenko_beam
from pyck.elements.kirchhoff_love_plate import KirchhoffLovePlate

from pyck.materials import SlenderBeam1d, PlaneStress2d

def create_rotation_free_timoshenko_beam(
    E: float, nu: float, A: float, I: float, k: float = 5.0 / 6.0
) -> TimoshenkoBeam1p:
    material = SlenderBeam1d(E, nu, A, I, k)
    return TimoshenkoBeam1p(material)


def create_kirchhoff_love_plate(
    E: float, nu: float, thickness: float
) -> KirchhoffLovePlate:
    material = PlaneStress2d(E, nu, thickness)
    return KirchhoffLovePlate(material)


__all__ = [
    "Element",
    "EulerBernoulliBeam",
    "KirchhoffLovePlate",
    "TimoshenkoBeam1p",
    "TimoshenkoBeam2p",
    "create_euler_bernoulli_beam",
    "create_kirchhoff_love_plate",
    "create_timoshenko_beam",
    "create_rotation_free_timoshenko_beam",
]
