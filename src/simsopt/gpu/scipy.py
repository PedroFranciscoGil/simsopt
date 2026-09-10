"""A narrow host bridge for SciPy optimizers."""

import jax
import jax.numpy as jnp
import numpy as np


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
