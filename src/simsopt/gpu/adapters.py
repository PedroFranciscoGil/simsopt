"""Host-side adapters from SIMSOPT objects to immutable array data."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .config import GpuConfig
from .curves import fourier_basis_set, symmetry_transforms
from .objective import minimal_coil_objective


@dataclass(frozen=True)
class MinimalCoilData:
    """Array representation of the supported minimal stage-two problem."""

    curve_dofs: np.ndarray
    base_currents: np.ndarray
    bases: jnp.ndarray
    transforms: jnp.ndarray
    current_signs: jnp.ndarray
    surface_points: np.ndarray
    surface_normal: np.ndarray
    target_normal_field: np.ndarray
    order: int
    nfp: int
    stellsym: bool

    def objective(
        self,
        curve_dofs,
        base_currents,
        *,
        length_target,
        length_weight,
        flux_definition="quadratic flux",
        curvature_threshold=5.0,
        curvature_weight=0.0,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=0.0,
        arclength_variation_weight=0.0,
        config: GpuConfig = None,
    ):
        """Evaluate the device objective using this problem's static data."""
        if config is None:
            config = GpuConfig()
        return minimal_coil_objective(
            curve_dofs,
            base_currents,
            self.bases,
            self.transforms,
            self.current_signs,
            self.surface_points,
            self.surface_normal,
            self.target_normal_field,
            length_target=length_target,
            length_weight=length_weight,
            flux_definition=flux_definition,
            curvature_threshold=curvature_threshold,
            curvature_weight=curvature_weight,
            mean_squared_curvature_threshold=mean_squared_curvature_threshold,
            mean_squared_curvature_weight=mean_squared_curvature_weight,
            arclength_variation_weight=arclength_variation_weight,
            target_tile_size=config.target_tile_size,
            source_tile_size=config.source_tile_size,
            vjp_mode=config.vjp_mode,
        )


def minimal_coil_data(
    surface,
    base_curves,
    base_currents,
    nfp: int,
    stellsym: bool,
    target_normal_field=None,
    *,
    config: GpuConfig = None,
) -> MinimalCoilData:
    """Copy a supported SIMSOPT problem into a flat array representation."""
    from simsopt.geo import CurveXYZFourier

    if config is None:
        config = GpuConfig()
    if not base_curves:
        raise ValueError("at least one base curve is required")
    if len(base_curves) != len(base_currents):
        raise ValueError("base_curves and base_currents must have equal length")
    if not all(isinstance(curve, CurveXYZFourier) for curve in base_curves):
        raise TypeError("all base curves must be CurveXYZFourier instances")

    curve_dofs = np.stack(
        [np.asarray(curve.get_dofs(), dtype=config.dtype) for curve in base_curves]
    )
    coefficients_per_axis, remainder = divmod(curve_dofs.shape[1], 3)
    if remainder or coefficients_per_axis % 2 != 1:
        raise TypeError("base curves are not compatible with CurveXYZFourier")
    order = (coefficients_per_axis - 1) // 2

    quadpoints = np.asarray(base_curves[0].quadpoints, dtype=config.dtype)
    for curve in base_curves[1:]:
        candidate = np.asarray(curve.quadpoints, dtype=config.dtype)
        if candidate.shape != quadpoints.shape or not np.array_equal(
            candidate, quadpoints
        ):
            raise ValueError("all base curves must use identical quadrature points")

    current_values = np.asarray(
        [current.get_value() for current in base_currents], dtype=config.dtype
    )
    surface_points = np.asarray(surface.gamma(), dtype=config.dtype).reshape((-1, 3))
    surface_normal = np.asarray(surface.normal(), dtype=config.dtype).reshape((-1, 3))
    if target_normal_field is None:
        target = np.zeros(surface_points.shape[0], dtype=config.dtype)
    else:
        target = np.asarray(target_normal_field, dtype=config.dtype)
        if target.size != surface_points.shape[0]:
            raise ValueError(
                "target_normal_field must have one value per surface point"
            )
        target = target.reshape((-1,))

    bases = fourier_basis_set(
        quadpoints, order, max_derivative=3, dtype=config.jax_dtype
    )
    transforms, current_signs = symmetry_transforms(
        nfp, stellsym, dtype=config.jax_dtype
    )
    return MinimalCoilData(
        curve_dofs=curve_dofs,
        base_currents=current_values,
        bases=bases,
        transforms=transforms,
        current_signs=current_signs,
        surface_points=surface_points,
        surface_normal=surface_normal,
        target_normal_field=target,
        order=order,
        nfp=nfp,
        stellsym=stellsym,
    )
