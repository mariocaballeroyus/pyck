"""Finite element formulations."""

from pyck.elements.element import Element
from pyck.elements.beams import (
    BeamEulerBernoulli1p,
    BeamTimoshenko1p,
    BeamTimoshenko2p,
    create_euler_bernoulli_beam,
    create_timoshenko_beam,
)
from pyck.elements.plates import (
    PlateKirchhoffLove1p,
    PlateReissnerMindlin1p,
    PlateReissnerMindlin3p,
    create_kirchhoff_love_plate,
    create_reissner_mindlin_plate,
)

from pyck.materials import SlenderBeam1d, PlaneStress2d


def create_rotation_free_timoshenko_beam(
    E: float, nu: float, A: float, I: float, k: float = 5.0 / 6.0
) -> BeamTimoshenko1p:
    """Factory function for a rotation-free Timoshenko beam element."""
    material = SlenderBeam1d(E, nu, A, I, k)
    return BeamTimoshenko1p(material)


def create_rotation_free_reissner_mindlin_plate(E, nu, h, ks=5.0/6.0):
    """Factory function for a single-variable (rotation-free) Reissner-Mindlin plate element."""
    material = PlaneStress2d(E, nu, h, ks)
    return PlateReissnerMindlin1p(material)


__all__ = [
    "Element",
    # Beam elements
    "BeamEulerBernoulli1p",
    "BeamTimoshenko1p",
    "BeamTimoshenko2p",
    # Plate elements
    "PlateKirchhoffLove1p",
    "PlateReissnerMindlin1p",
    "PlateReissnerMindlin3p",
    # Factory functions
    "create_euler_bernoulli_beam",
    "create_timoshenko_beam",
    "create_rotation_free_timoshenko_beam",
    "create_kirchhoff_love_plate",
    "create_reissner_mindlin_plate",
    "create_rotation_free_reissner_mindlin_plate",
]
