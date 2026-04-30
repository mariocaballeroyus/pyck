import numpy as np
import pyck as ck


def test_export_rm_displ_2p_all_fields(tmp_path):
    patch = ck.geometry.create_rectangle(nu=4, nv=4, deg=3, l=1.0, w=1.0)
    material = ck.material.create_plane_stress(10.92, 0.3, 0.25)
    element = ck.PlateReissnerMindlinDispl2p(material)
    n_dofs = patch.num_control_pts * element._cpp_object.num_node_dofs()
    solution = np.linspace(0.0, 1.0, n_dofs)

    path = ck.io.export_field_vtk(
        tmp_path / "rm_displ_2p_all.vtk",
        patch,
        element,
        solution,
        samples_per_span=3,
        fields=("all",),
    )

    text = path.read_text()
    assert "SCALARS w float 1" in text
    assert "SCALARS wb float 1" in text
    assert "SCALARS shear_displacement float 1" in text
    assert "SCALARS psi float 1" in text
    assert "SCALARS curl_psi float 1" in text
    assert "VECTORS curl_psi_vector float" in text
    assert "VECTORS rotation float" in text
    assert "VECTORS kappa float" in text
    assert "VECTORS gamma float" in text
