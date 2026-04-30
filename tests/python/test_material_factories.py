import pyck as ck


def test_create_plane_stress_factory():
    material = ck.material.create_plane_stress(10.92, 0.3, 0.25)

    assert isinstance(material, ck.PlaneStress2d)
    assert material.youngs_modulus == 10.92
    assert material.poisson_ratio == 0.3
    assert material.thickness == 0.25


def test_create_plane_stress_factory_is_exported_at_top_level():
    material = ck.create_plane_stress(10.92, 0.3, 0.25)

    assert isinstance(material, ck.PlaneStress2d)
    assert material.thickness == 0.25


def test_create_slender_beam_factory():
    material = ck.material.create_slender_beam(210.0e9, 0.3, 0.02, 1.0e-6)

    assert isinstance(material, ck.SlenderBeam1d)
    assert material.youngs_modulus == 210.0e9
    assert material.section_area == 0.02
    assert material.moment_inertia == 1.0e-6
