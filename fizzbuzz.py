"""
Print the numbers 1 to 100, except
* if the number is divisible by 3, print "fizz"
* if the number is divisible by 5, print "buzz"
* if the number is divisible by 15, print "fizzbuzz"
"""

import numpy as np
from typing import List
from Autograd import Tensor, Parameter, Module
from Autograd import SGD
from Autograd import tanh


def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]


def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


x_train = Tensor([binary_encode(x) for x in range(101, 1024)])
y_train = Tensor([fizz_buzz_encode(x) for x in range(101, 1024)])


class FizzBuzzModel(Module):
    def __init__(self, num_hidden: int = 10) -> None:
        self.w1 = Parameter(10, num_hidden)
        self.b1 = Parameter(num_hidden)

        self.w2 = Parameter(num_hidden, 4)
        self.b2 = Parameter(4)

    def predict(self, inputs: Tensor) -> Tensor:
        # inputs will be batch size x10
        x1 = inputs @ self.w1 + self.b1 # (batchsize, num_hidden)
        x2 = tanh(x1)
        x3 = x2 @ self.w2 + self.b2 # numhidden, 4
        return x3


model = FizzBuzzModel()
batch_size = 32
l_r = 0.001
optimizer = SGD(l_r)
starts = np.arange(0, x_train.shape[0], batch_size)
for epoch in range(10000):
    epoch_loss = 0.0

    np.random.shuffle(starts)
    for start in starts:
        end = start + batch_size

        model.zero_grad()

        inputs = x_train[start:end]
        predicted = model.predict(inputs)
        actuals = y_train[start:end]
        errors = predicted - actuals
        loss = (errors*errors).sum()

        loss.backward()
        epoch_loss += loss.data

        # model.w -= model.w.grad*l_r
        # model.b -= model.b.grad*l_r
        optimizer.step(model)

    print(epoch, epoch_loss)

n_correct = 0
for x in range(1, 101):
    inputs = Tensor(binary_encode(x))
    predicted = model.predict(inputs)[0]
    predicted_idx = np.argmax(predicted.data)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), 'fizz', 'buzz', 'fizzbuzz']

    if actual_idx == predicted_idx:
        n_correct += 1
    #print(x, labels[actual_idx], labels[predicted_idx])

print(n_correct,'/100')