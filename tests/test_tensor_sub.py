import unittest

from Autograd.tensor import Tensor


class TestTensorSub(unittest.TestCase):
    def test_simple_sum(self):
        t1 = Tensor([1,2,3], require_grad=True)
        t2 = Tensor([4,5,6], require_grad=True)

        t3 = t1 - t2
        t3.backward(Tensor([-1,-2,-3]))

        assert t1.grad.data.tolist() == [-1,-2,-3]
        assert t2.grad.data.tolist() == [1,2,3]

        t1 -= 0.1
        assert t1.data.tolist() == [0.9, 1.9, 2.9]

    def test_broadcast_sub(self):
        t1 = Tensor([[1, 2, 3], [4,5,6]], require_grad=True) # dim = 2,3
        t2 = Tensor([7,8,9], require_grad=True) # dim = (3,)

        t3 = t1 - t2
        t3.backward(Tensor([[1,1,1],[1,1,1]]))

        assert t1.grad.data.tolist() == [[1,1,1],[1,1,1]]
        assert t2.grad.data.tolist() == [-2, -2, -2]

    def test_broadcast_sub2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], require_grad=True) # dim = 2,3
        t2 = Tensor([[7, 8, 9]], require_grad=True) # dim = (1,3)

        t3 = t1 - t2
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [[-2, -2, -2]]
