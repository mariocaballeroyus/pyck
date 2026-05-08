"""Material models for structural elements."""

from pyck.materials.material import Material
from pyck.materials.slender_beam_1d import SlenderBeam1d
from pyck.materials.plane_stress_2d import PlaneStress2d
from pyck.materials.plane_stress_shell import PlaneStressShell


def create_plane_stress(
    E: float, nu: float = 0.3, h: float = 1.0, k: float = 5.0 / 6.0
) -> PlaneStress2d:
    """Create a linear elastic isotropic plane-stress material."""
    return PlaneStress2d(E, nu, h, k)


def create_plane_stress_2d(
    E: float, nu: float = 0.3, h: float = 1.0, k: float = 5.0 / 6.0
) -> PlaneStress2d:
    """Create a linear elastic isotropic plane-stress material."""
    return create_plane_stress(E, nu, h, k)


def create_slender_beam(
    E: float, nu: float, A: float, I: float, k: float = 5.0 / 6.0
) -> SlenderBeam1d:
    """Create a material and section model for 1D beam elements."""
    return SlenderBeam1d(E, nu, A, I, k)


def create_slender_beam_1d(
    E: float, nu: float, A: float, I: float, k: float = 5.0 / 6.0
) -> SlenderBeam1d:
    """Create a material and section model for 1D beam elements."""
    return create_slender_beam(E, nu, A, I, k)


__all__ = [
    "Material",
    "SlenderBeam1d",
    "PlaneStress2d",
    "PlaneStressShell",
    "create_plane_stress",
    "create_plane_stress_2d",
    "create_slender_beam",
    "create_slender_beam_1d",
]
