import numpy as np
from typing import List, NamedTuple, Callable, Union, Optional

# each dependency points back to some other tensor and it has some function
# that takes my grad_fn and calculates gradient for backward
# For eg: if 2 tensors are added, the new tensor knows what backward function was
# and points back to them to calculate gradient wrt of each tensor
class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray] ## read union again video


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    """
    Can pass normal array in the Tensor and not necessarily
    :param arrayable:
    :return:
    """
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)

class Tensor:
    def __init__(self,
                 data: Arrayable,
                 require_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self.data = ensure_array(data)
        self.require_grad = require_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.require_grad:
            self.zero_grad()

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data))

    def __repr__(self) -> str:
        return "tensor {}, requires_grad={}".format(self.data, self.require_grad)

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.require_grad, "called backward on non-require-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError('grad must be specified for non-0-tensor ')

        self.grad.data += grad.data
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:
    """
    Sums all elements of tensor and return 0-D tensor
    :param t:
    :return:
    """
    data = np.sum(t.data)
    requires_grad = t.require_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily is 0-tensor,
            :param grad:
            :return:
            """
            return grad * np.ones_like(t.data)
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data,
                  requires_grad,
                  depends_on)