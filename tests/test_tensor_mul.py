import unittest

from Autograd.tensor import Tensor
import numpy as np

class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self):
        t1 = Tensor([1,2,3], require_grad=True)
        t2 = Tensor([4,5,6], require_grad=True)

        t3 = t1 * t2  # overriden in tensor class
        t3.backward(Tensor([-1,-2,-3]))
        assert t1.grad.data.tolist() == [-4, -10, -18]
        assert t2.grad.data.tolist() == [-1, -4, -9]

        t1 *= 2
        assert t1.data.tolist() == [2, 4, 6]
        t1 *= 0.1
        np.testing.assert_array_almost_equal(t1.data, [0.2, 0.4, 0.6])

    def test_broadcast_mul(self):
        t1 = Tensor([[1, 2, 3], [4,5,6]], require_grad=True) # dim = 2,3
        t2 = Tensor([7,8,9], require_grad=True) # dim = (3,)

        t3 = t1 * t2
        t3.backward(Tensor([[1,1,1],[1,1,1]]))

        assert t1.grad.data.tolist() == [[7,8,9],[7,8,9]]
        assert t2.grad.data.tolist() == [5,7,9]

    def test_broadcast_mul2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True) # dim = 2,3
        t2 = Tensor([[7, 8, 9]], require_grad=True) # dim = (1,3)

        t3 = t1 * t2
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[7,8,9],[7,8,9]]
        assert t2.grad.data.tolist() == [[5,7,9]]
