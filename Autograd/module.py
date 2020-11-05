from Autograd import Tensor
from Autograd import Parameter
import inspect
from typing import Iterator

# collections of parameters that has a forward method
# similar to PyTorch's that contains layers and a method forward(input) that returns the output
class Module:
    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self): # get all the methods/parameters of the object
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self):
        for parameters in self.parameters():
            parameters.zero_grad()
