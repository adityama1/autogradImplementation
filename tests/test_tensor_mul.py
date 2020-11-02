import unittest

from Autograd.tensor import Tensor, mul


class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self):
        t1 = Tensor([1,2,3], require_grad=True)
        t2 = Tensor([4,5,6], require_grad=True)

        t3 = mul(t1, t2)
        t3.backward(Tensor([-1,-2,-3]))

        assert t1.grad.data.tolist() == [-4,-10,-18]
        assert t2.grad.data.tolist() == [-1,-4,-9]

    def test_broadcast_mul(self):
        t1 = Tensor([[1, 2, 3], [4,5,6]], require_grad=True) # dim = 2,3
        t2 = Tensor([7,8,9], require_grad=True) # dim = (3,)

        t3 = mul(t1, t2)
        t3.backward(Tensor([[1,1,1],[1,1,1]]))

        t1.grad.data.tolist() == [[7,8,9],[7,8,9]]
        t2.grad.data.tolist() == [5,7,9]

    def test_broadcast_mul2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True) # dim = 2,3
        t2 = Tensor([[7, 8, 9]], require_grad=True) # dim = (1,3)

        t3 = mul(t1, t2)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        t1.grad.data.tolist() == [[7,8,9],[7,8,9]]
        t2.grad.data.tolist() == [[5,7,9]]
