# This will minimize the sum of squared error loss function

from Autograd.tensor import Tensor

# create a random tensor
x = Tensor([10, 40, -5, 6, 7, 1, 4, -10], require_grad=True)

for i in range(100):
    # calculate x**2
    x.zero_grad()
    sum_sq_error = (x * x).sum()

    # set l.r. -> 0.1
    sum_sq_error.backward()

    # calc. delta
    delta_x = 0.1 * x.grad
    # x = Tensor(x.data - delta_x.data, require_grad=True)
    x -= delta_x

    print(i, sum_sq_error)

