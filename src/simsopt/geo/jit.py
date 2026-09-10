from jax import jit as jaxjit

from .config import parameters


def jit(fun, **args):
    if parameters["jit"]:
        return jaxjit(fun, **args)
    else:
        return fun
