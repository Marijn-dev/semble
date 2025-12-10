from typing import Any

from numpy.typing import NDArray
from numpy.random import Generator


Args = dict[str, Any]

class ParameterGenerator:
    dim: int

    def __init__(self, dim: int):
        self.dim = dim # dimension of the parameter

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def sample(self, rng: Generator) -> NDArray:
        return self._sample_impl(rng) 
    
    def _sample_impl(self, rng: Generator) -> NDArray:
        del rng
        raise NotImplementedError 
    
class Uniform(ParameterGenerator):
    def __init__(self, low, high, dim):
        super().__init__(dim)

        self._low = low
        self._high = high
        
    def _sample_impl(self, rng) -> NDArray:
        parameter = rng.uniform(low=self._low, high=self._high, size=self.dim)
        return parameter



# Distributions     
_parameter_names = {
    "Uniform": Uniform,
    }   


def get_parameter_generator(name: str, args: dict[str, Any]) -> ParameterGenerator:
    return _parameter_names[name](**args)