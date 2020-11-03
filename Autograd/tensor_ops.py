from Autograd.tensor import Tensor, Dependency
import numpy as np
from typing import List

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


# _add to represent that method is internal/private
def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    require_grad: bool = t1.require_grad or t2.require_grad

    depends_on: List[Dependency] = []

    if t1.require_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily is 0-tensor,
            :param grad:
            :return:
            """
            # handle broadcasting
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # sum across broadcasted but non-added dims
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad * np.ones_like(t1.data)
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.require_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily is 0-tensor,
            :param grad:
            :return:
            """
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # sum across broadcasted but non-added dims
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad * np.ones_like(t2.data)
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  require_grad,
                  depends_on)


# Internal/private method
def _mul(t1: Tensor, t2:Tensor) -> Tensor:
    data = t1.data * t2.data
    require_grad: bool = t1.require_grad or t2.require_grad

    depends_on: List[Dependency] = []

    if t1.require_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily is 0-tensor,
            :param grad:
            :return:
            """
            grad = grad * t2.data
            # handle broadcasting
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # sum across broadcasted but non-added dims
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.require_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily is 0-tensor,
            :param grad:
            :return:
            """
            grad *= t1.data
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # sum across broadcasted but non-added dims
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  require_grad,
                  depends_on)


def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.require_grad

    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []
    return Tensor(data,
                  requires_grad,
                  depends_on)


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2