"""Configuration and diagnostics for the device-resident backend."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Scientific parity is defined in double precision. This selects precision,
# not a device; JAX remains free to use the configured CPU, GPU, or TPU backend.
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class GpuConfig:
    """Static configuration for compiled coil kernels."""

    dtype: str = "float64"
    target_tile_size: int = 128
    source_tile_size: int = 256
    vjp_mode: str = "custom"

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.target_tile_size <= 0:
            raise ValueError("target_tile_size must be positive")
        if self.source_tile_size <= 0:
            raise ValueError("source_tile_size must be positive")
        if self.vjp_mode not in ("autodiff", "custom"):
            raise ValueError("vjp_mode must be 'autodiff' or 'custom'")

    @property
    def jax_dtype(self):
        """Return the JAX scalar type selected by dtype."""
        return jnp.float64 if self.dtype == "float64" else jnp.float32


def backend_report() -> dict:
    """Return machine-readable information about the active JAX backend."""
    devices = [
        {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "id": str(device.id),
        }
        for device in jax.devices()
    ]
    return {
        "default_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "devices": devices,
    }
