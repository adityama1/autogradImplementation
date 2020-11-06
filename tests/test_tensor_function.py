import unittest
from Autograd import Tensor, tanh, relu
import numpy as np


class TestTensorFunction(unittest.TestCase):
    def test_tanh_computation(self):
        t1 = Tensor([1,2,3], require_grad=True)
        t2 = tanh(t1)

        np.testing.assert_array_almost_equal(t2.data, np.tanh(t1.data))

        t2.backward(Tensor([1,1,1]))
        # gradient of tanh is 1-tanh**2
        truth_data = 1 - np.tanh(t1.data)**2
        np.testing.assert_array_almost_equal(t1.grad.data, truth_data)

    def test_relu_computation(self):
        t1 = Tensor([1, -2, 3, -1], require_grad=True)
        t2 = relu(t1)

        assert t2.data.tolist() == [1, 0, 3, 0]

        t3 = Tensor(np.random.randn(10,3))
        t4 = relu(t3)
        max_t3 = [max(t3.data[row][col], 0) for row in range(len(t3.data)) for col in range(len(t3.data[0]))]

        np.testing.assert_array_almost_equal(max_t3, t4.data.flatten())

    def test_relu_grad(self):
        t1 = Tensor([1, -2, 3, -1], require_grad=True)
        t2 = relu(t1)

        t2.backward(Tensor([1,1,1,1]))

        assert t1.grad.data.tolist() == [1, 0, 1, 0]

        t3 = Tensor(np.random.randn(10, 3), require_grad=True)
        t4 = relu(t3)

        gradient = Tensor(np.random.randn(10,3))
        t4.backward(gradient)
        gradient.data[t3.data <= 0] = 0
        np.testing.assert_array_almost_equal(t3.grad.data, gradient.data)