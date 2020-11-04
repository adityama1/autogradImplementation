import numpy as np

from Autograd.tensor import Tensor

x_data = Tensor(np.random.rand(100, 3))
coef = Tensor(np.array([-1, 3, -2], dtype=float))
y_data = x_data @ coef + 5# + np.random.randint(-2,2, size=(100,))  # matrix multiplication @

w = Tensor(np.random.randn(3), require_grad=True)
b = Tensor(np.random.randn(), require_grad=True)
batch_size = 32
for epoch in range(100):
    epoch_loss = 0.0
    for start in range(0, 100, batch_size):
        end = start + batch_size
        w.zero_grad()
        b.zero_grad()

        inputs = x_data[start:end]
        predicted = inputs @ w + b
        actuals = y_data[start:end]
        errors = predicted - actuals
        loss = (errors*errors).sum()

        loss.backward()
        epoch_loss += loss.data

        l_r = 0.001
        w -= w.grad*l_r
        b -= b.grad*l_r

    print(epoch, epoch_loss)

print('Final: ', w)