import unittest

from Autograd.tensor import Tensor
import numpy as np


class TestTensorMatMul(unittest.TestCase):
    def test_simple_mul(self):
        t1 = Tensor([[1,2],[3,4],[5,6]], require_grad=True) # 3x2
        t2 = Tensor([[10],[20]], require_grad=True) # 2x1

        t3 = t1 @ t2  # overriden in tensor class
        assert t3.data.tolist() == [[50], [110], [170]]

        grad = Tensor([[-1], [-2], [-3]])
        t3.backward(grad)

        np.testing.assert_array_almost_equal(t1.grad.data,
                                             grad.data @ t2.data.T)
        np.testing.assert_array_almost_equal(t2.grad.data,
                                             t1.data.T @ grad.data)

    # def test_broadcast_mul(self):
    #     t1 = Tensor([[1, 2, 3], [4,5,6]], require_grad=True) # dim = 2,3
    #     t2 = Tensor([7,8,9], require_grad=True) # dim = (3,)
    #
    #     t3 = t1 * t2
    #     t3.backward(Tensor([[1,1,1],[1,1,1]]))
    #
    #     assert t1.grad.data.tolist() == [[7,8,9],[7,8,9]]
    #     assert t2.grad.data.tolist() == [5,7,9]
    #
    # def test_broadcast_mul2(self):
    #     t1 = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True) # dim = 2,3
    #     t2 = Tensor([[7, 8, 9]], require_grad=True) # dim = (1,3)
    #
    #     t3 = t1 * t2
    #     t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
    #
    #     assert t1.grad.data.tolist() == [[7,8,9],[7,8,9]]
    #     assert t2.grad.data.tolist() == [[5,7,9]]
