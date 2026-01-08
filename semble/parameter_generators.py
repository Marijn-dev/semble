from typing import Any
from numpy.typing import NDArray
from numpy.random import Generator

import numpy as np

Args = dict[str, Any]


class ParameterGenerator:
    dim: int

    def __init__(self, dim: int):
        self.dim = dim  # dimension of the parameter

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def sample(self, rng: Generator) -> NDArray:
        return self._sample_impl(rng)

    def _sample_impl(self, rng: Generator) -> NDArray:
        del rng
        raise NotImplementedError


class Product(ParameterGenerator):
    def __init__(self, par_gens: list[ParameterGenerator]):
        super().__init__(len(par_gens))
        self._par_gens = par_gens

    def _sample_impl(self, rng):
        samples = tuple(g.sample(rng) for g in self._par_gens)

        return np.hstack(samples)


class Uniform(ParameterGenerator):
    def __init__(self, low, high, dim=1):
        super().__init__(dim)

        self._low = low
        self._high = high

    def _sample_impl(self, rng):
        parameter = rng.uniform(low=self._low, high=self._high, size=self.dim)
        return parameter

class Delta(ParameterGenerator):
    def __init__(self, value, dim=1):
        super().__init__(dim)

        self._value = value
    
    def _sample_impl(self, rng):
        parameter = np.full(shape=(self.dim,), fill_value=self._value, dtype=float)
        return parameter
    
# Distributions
_pargen_names = {
    "Uniform": Uniform,
    "Delta": Delta,
}


def get_parameter_generator(name: str, args: dict[str, Any]) -> ParameterGenerator:
    if name == "Product":
        if not isinstance(args, list):
            raise TypeError(
                "To construct a Product, args should be a list "
                "of (name, args) for each coordinate."
            )

        components = [get_parameter_generator(el["name"], el["args"]) for el in args]
        return Product(components)

    else:
        if not isinstance(args, dict):
            raise TypeError(
                f"args should be a dictionary of arguments to {name}.__init__."
            )

        return _pargen_names[name](**args)
