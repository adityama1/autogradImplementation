from Autograd import Tensor, Dependency
import numpy as np
from typing import List


def tanh(tensor: Tensor) -> Tensor:
    data = np.tanh(tensor.data)
    requires_grad = tensor.require_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on: List[Dependency] = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)