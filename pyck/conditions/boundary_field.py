"""Boundary field operators for weak boundary conditions."""

from __future__ import annotations

from typing import Literal

import pyck._pyck as _pyck

BoundaryField = _pyck.BoundaryField
TransverseDisplacement = _pyck.TransverseDisplacement
ForceTraction = _pyck.ForceTraction
NormalRotation = _pyck.NormalRotation
TangentialRotation = _pyck.TangentialRotation
NormalTransverseShear = _pyck.NormalTransverseShear
NormalBendingMoment = _pyck.NormalBendingMoment
TwistingMoment = _pyck.TwistingMoment

# Generic basis-DOF fields, parameterised by `dof_index` (which slot of the
# per-node DOF block to read).
BasisValue = _pyck.BasisValue
BasisNormalSlope = _pyck.BasisNormalSlope
BasisNormalCurvature = _pyck.BasisNormalCurvature

# Displacement-side fields: w (transverse displacement), rot_n (normal rotation),
# rot_s (tangential rotation).
# Traction-side fields: q_n (normal transverse shear), m_n (normal bending
# moment), m_ns (twisting moment).
FieldName = Literal["w", "rot_n", "rot_s", "q_n", "m_n", "m_ns"]

_FIELD_FACTORIES: dict[str, type] = {
    "w": _pyck.TransverseDisplacement,
    "rot_n": _pyck.NormalRotation,
    "rot_s": _pyck.TangentialRotation,
    "q_n": _pyck.NormalTransverseShear,
    "m_n": _pyck.NormalBendingMoment,
    "m_ns": _pyck.TwistingMoment,
}


def resolve_field(field: FieldName | _pyck.BoundaryField) -> _pyck.BoundaryField:
    """Map user-facing field names to a BoundaryField instance."""
    if isinstance(field, _pyck.BoundaryField):
        return field
    if not isinstance(field, str):
        raise TypeError("field must be a pyck.BoundaryField or a field name string.")

    try:
        return _FIELD_FACTORIES[field.lower()]()
    except KeyError as exc:
        names = ", ".join(repr(k) for k in _FIELD_FACTORIES)
        raise ValueError(
            f"Unknown boundary field {field!r}. Expected one of: {names}."
        ) from exc
