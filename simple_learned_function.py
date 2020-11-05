import numpy as np

from Autograd import Tensor, Parameter, Module
from Autograd import SGD

x_data = Tensor(np.random.rand(100, 3))
coef = Tensor(np.array([-1, 3, -2], dtype=float))
y_data = x_data @ coef + 5# + np.random.randint(-2,2, size=(100,))  # matrix multiplication @


class Model(Module):
    def __init__(self) -> None:
        #w = Tensor(np.random.randn(3), require_grad=True)
        #b = Tensor(np.random.randn(), require_grad=True)
        self.w = Parameter(3)
        self.b = Parameter()

    def predict(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b


model = Model()
batch_size = 32
l_r = 0.001
optimizer = SGD(l_r)

for epoch in range(1000):
    epoch_loss = 0.0
    for start in range(0, 100, batch_size):
        end = start + batch_size

        model.zero_grad()

        inputs = x_data[start:end]
        predicted = model.predict(inputs)
        actuals = y_data[start:end]
        errors = predicted - actuals
        loss = (errors*errors).sum()

        loss.backward()
        epoch_loss += loss.data

        # model.w -= model.w.grad*l_r
        # model.b -= model.b.grad*l_r
        optimizer.step(model)

    print(epoch, epoch_loss)

print('Final: ', model.w)