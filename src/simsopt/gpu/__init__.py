"""Device-resident numerical kernels for coil optimization.

This package is intentionally functional: arrays cross the boundary once and
the objective can then be differentiated and compiled as one JAX program.
The API is experimental while the GPU-native implementation is developed.
"""

from .adapters import MinimalCoilData, minimal_coil_data
from .augmented_lagrangian import (
    AugmentedLagrangianResult,
    ScipyAugmentedLagrangianBridge,
    equality_augmented_lagrangian,
    minimize_equality_augmented_lagrangian,
    smooth_abs_constraints,
    smooth_sqrt_constraints,
)
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
from .device_augmented_lagrangian import (
    DeviceAugmentedLagrangian,
    DeviceAugmentedLagrangianConfig,
    DeviceAugmentedLagrangianResult,
)
from .distances import (
    coil_coil_distance,
    coil_coil_distance_residuals,
    coil_surface_distance,
    coil_surface_distance_residuals,
    curve_pair_indices,
)
from .flux import normalized_flux, quadratic_flux
from .lbfgs import (
    DeviceLBFGS,
    DeviceLBFGSConfig,
    DeviceLBFGSResult,
    TargetAwareCheckpointRecorder,
    TargetAwareConfig,
)
from .objective import (
    LOCAL_RESIDUAL_FAMILIES,
    local_engineering_residual_layout,
    minimal_coil_augmented_lagrangian_terms,
    minimal_coil_local_residual_terms,
    minimal_coil_objective,
)
from .regularizers import (
    arclength_variation,
    curve_curvature_residuals,
    curve_curvatures,
    lp_curve_curvature_penalty,
    mean_squared_curvature,
    mean_squared_curvature_residuals,
)
from .scipy import ScipyCoilObjectiveBridge, ScipyObjectiveBridge

__all__ = [
    "LOCAL_RESIDUAL_FAMILIES",
    "AugmentedLagrangianResult",
    "DeviceAugmentedLagrangian",
    "DeviceAugmentedLagrangianConfig",
    "DeviceAugmentedLagrangianResult",
    "DeviceLBFGS",
    "DeviceLBFGSConfig",
    "DeviceLBFGSResult",
    "GpuConfig",
    "MinimalCoilData",
    "ScipyAugmentedLagrangianBridge",
    "ScipyCoilObjectiveBridge",
    "ScipyObjectiveBridge",
    "TargetAwareCheckpointRecorder",
    "TargetAwareConfig",
    "arclength_variation",
    "backend_report",
    "biot_savart_field",
    "biot_savart_field_custom_vjp",
    "biot_savart_field_reference",
    "coefficients_to_dofs",
    "coil_coil_distance",
    "coil_coil_distance_residuals",
    "coil_surface_distance",
    "coil_surface_distance_residuals",
    "curve_curvature_residuals",
    "curve_curvatures",
    "curve_lengths",
    "curve_pair_indices",
    "dofs_to_coefficients",
    "equality_augmented_lagrangian",
    "evaluate_cartesian_fourier",
    "evaluate_cartesian_fourier_derivatives",
    "expand_by_symmetry",
    "fourier_basis",
    "fourier_basis_set",
    "local_engineering_residual_layout",
    "lp_curve_curvature_penalty",
    "mean_squared_curvature",
    "mean_squared_curvature_residuals",
    "minimal_coil_augmented_lagrangian_terms",
    "minimal_coil_data",
    "minimal_coil_local_residual_terms",
    "minimal_coil_objective",
    "minimize_equality_augmented_lagrangian",
    "normalized_flux",
    "quadratic_flux",
    "smooth_abs_constraints",
    "smooth_sqrt_constraints",
    "symmetry_transforms",
]
