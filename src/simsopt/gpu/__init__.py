"""Device-resident numerical kernels for coil optimization.

This package is intentionally functional: arrays cross the boundary once and
the objective can then be differentiated and compiled as one JAX program.
The API is experimental while the GPU-native implementation is developed.
"""

from .adapters import MinimalCoilData, minimal_coil_data
from .biot_savart import (
    biot_savart_field,
    biot_savart_field_custom_vjp,
    biot_savart_field_reference,
)
from .config import GpuConfig, backend_report
from .curves import (
    coefficients_to_dofs,
    curve_lengths,
    dofs_to_coefficients,
    evaluate_cartesian_fourier,
    evaluate_cartesian_fourier_derivatives,
    expand_by_symmetry,
    fourier_basis,
    fourier_basis_set,
    symmetry_transforms,
)
from .flux import normalized_flux, quadratic_flux
from .objective import minimal_coil_objective
from .scipy import ScipyObjectiveBridge

__all__ = [
    "GpuConfig",
    "MinimalCoilData",
    "ScipyObjectiveBridge",
    "backend_report",
    "biot_savart_field",
    "biot_savart_field_custom_vjp",
    "biot_savart_field_reference",
    "coefficients_to_dofs",
    "curve_lengths",
    "dofs_to_coefficients",
    "evaluate_cartesian_fourier",
    "evaluate_cartesian_fourier_derivatives",
    "expand_by_symmetry",
    "fourier_basis",
    "fourier_basis_set",
    "minimal_coil_data",
    "minimal_coil_objective",
    "normalized_flux",
    "quadratic_flux",
    "symmetry_transforms",
]
