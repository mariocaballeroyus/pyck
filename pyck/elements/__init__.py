"""Finite element formulations."""

from pyck.elements.element import Element
from pyck.elements.beam_euler_bernoulli_1p import BeamEulerBernoulli1p, create_euler_bernoulli_beam
from pyck.elements.beam_timoshenko_1p import BeamTimoshenko1p
from .beam_timoshenko_2p import BeamTimoshenko2p, create_timoshenko_beam
from pyck.elements.plate_kirchhoff_love_1p import PlateKirchhoffLove1p
from pyck.elements.plate_reissner_mindlin_3p import PlateReissnerMindlin3p

from pyck.materials import SlenderBeam1d, PlaneStress2d

def create_rotation_free_timoshenko_beam(
    E: float, nu: float, A: float, I: float, k: float = 5.0 / 6.0
) -> BeamTimoshenko1p:
    material = SlenderBeam1d(E, nu, A, I, k)
    return BeamTimoshenko1p(material)


def create_kirchhoff_love_plate(
    E: float, nu: float, thickness: float
) -> PlateKirchhoffLove1p:
    material = PlaneStress2d(E, nu, thickness)
    return PlateKirchhoffLove1p(material)


def create_reissner_mindlin_plate(E, nu, h, ks=5.0/6.0):
    """
    Factory function for a Reissner-Mindlin plate element.
    """
    material = PlaneStress2d(E, nu, h, ks)
    return PlateReissnerMindlin3p(material)


__all__ = [
    "Element",
    "BeamEulerBernoulli1p",
    "PlateKirchhoffLove1p",
    "PlateReissnerMindlin3p",
    "BeamTimoshenko1p",
    "BeamTimoshenko2p",
    "create_euler_bernoulli_beam",
    "create_kirchhoff_love_plate",
    "create_reissner_mindlin_plate",
    "create_timoshenko_beam",
    "create_rotation_free_timoshenko_beam",
]
