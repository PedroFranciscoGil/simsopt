"""A narrow host bridge for SciPy optimizers."""

import jax
import jax.numpy as jnp
import numpy as np

from .config import GpuConfig


class ScipyObjectiveBridge:
    """Compile a flat JAX objective and expose SciPy's value-gradient protocol."""

    def __init__(self, objective, initial_x):
        initial_x = np.asarray(initial_x)
        if initial_x.ndim != 1:
            raise ValueError("initial_x must be one-dimensional")
        if not np.issubdtype(initial_x.dtype, np.floating):
            raise TypeError("initial_x must have a floating-point dtype")
        self._objective = objective
        self._value_and_grad = jax.jit(jax.value_and_grad(objective))
        self._compiled = None
        self._shape = initial_x.shape
        self._dtype = initial_x.dtype
        self.evaluations = 0

    @property
    def is_compiled(self):
        """Whether the executable has been explicitly compiled."""
        return self._compiled is not None

    def compile(self, example_x):
        """Compile for the configured shape and dtype, outside timed regions."""
        x = self._coerce(example_x)
        device_x = jnp.asarray(x)
        self._compiled = self._value_and_grad.lower(device_x).compile()
        value, gradient = self._compiled(device_x)
        value.block_until_ready()
        gradient.block_until_ready()
        return self

    def _coerce(self, x):
        x = np.asarray(x, dtype=self._dtype)
        if x.shape != self._shape:
            raise ValueError(f"x must have shape {self._shape}, got {x.shape}")
        return x

    def __call__(self, x):
        """Return a Python float and NumPy gradient for scipy.optimize.minimize."""
        host_x = self._coerce(x)
        if self._compiled is None:
            self.compile(host_x)
        value, gradient = self._compiled(jnp.asarray(host_x))
        value.block_until_ready()
        self.evaluations += 1
        return float(value), np.asarray(gradient)


class ScipyCoilObjectiveBridge:
    """Expose a :class:`MinimalCoilData` objective in SIMSOPT dof order.

    SIMSOPT's stage-two objective orders free current degrees of freedom before
    the Fourier-curve coefficients. Fixed currents remain immutable device
    constants. This adapter preserves that order so SciPy sees the same vector
    and gradient as the CPU ``Optimizable`` graph.
    """

    def __init__(
        self,
        data,
        *,
        free_current_indices,
        objective_kwargs,
        config: GpuConfig = None,
    ):
        if config is None:
            config = GpuConfig()
        indices = np.asarray(free_current_indices, dtype=np.int32)
        if indices.ndim != 1:
            raise ValueError("free_current_indices must be one-dimensional")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("free_current_indices must not contain duplicates")
        if np.any(indices < 0) or np.any(indices >= data.base_currents.size):
            raise ValueError("free_current_indices contains an out-of-range index")

        self.data = data
        self.free_current_indices = indices
        self.objective_kwargs = dict(objective_kwargs)
        self.config = config
        self._current_count = indices.size
        self.initial_x = np.concatenate(
            (
                np.asarray(data.base_currents)[indices],
                np.asarray(data.curve_dofs).reshape((-1,)),
            )
        )

        def objective(x):
            curve_dofs, currents = self.unpack(x)
            return data.objective(
                curve_dofs,
                currents,
                **self.objective_kwargs,
                config=config,
            )

        self._bridge = ScipyObjectiveBridge(objective, self.initial_x)

    @property
    def is_compiled(self):
        return self._bridge.is_compiled

    @property
    def evaluations(self):
        return self._bridge.evaluations

    def unpack(self, x):
        """Return curve dofs and physical currents represented by flat ``x``."""
        x = jnp.asarray(x)
        if x.ndim != 1 or x.shape != self.initial_x.shape:
            raise ValueError(f"x must have shape {self.initial_x.shape}, got {x.shape}")
        currents = jnp.asarray(self.data.base_currents).at[
            self.free_current_indices
        ].set(x[: self._current_count])
        curve_dofs = x[self._current_count :].reshape(self.data.curve_dofs.shape)
        return curve_dofs, currents

    def compile(self):
        """Compile and warm the complete value-and-gradient executable."""
        self._bridge.compile(self.initial_x)
        return self

    def __call__(self, x):
        return self._bridge(x)
