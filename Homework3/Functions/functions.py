import numpy as np
import random

### activation functions:

def softmax(values, pos):
    elem_exp = np.exp(values[pos])

    distribution = np.sum(np.exp(values))
    
    return elem_exp / distribution

def sigmoid(values, pos):
    return 1 / (1 + np.exp(-values[pos]))

def sigmoid_derivative(values, pos, is_output = False):
    if is_output:
        return values[pos] * (1 - values[pos])

    sigmoid_value = sigmoid(values, pos)

    return sigmoid_value * (1 - sigmoid_value)


### cost functions:

def cross_entropy_derivative(error, shape):
    # print(error.shape, ":", shape)
    ced = np.zeros(shape)

    for i in range(shape[0]):
        ced[i] = np.full(shape[1], error[i])

    # print(ced)
    
    return ced

def quadratic_derivative(error, out, shape, inp=None):
    qd = np.zeros(shape)

    for i in range(shape[0]):
        qd[i] = np.full(shape[1], (error[i]) * sigmoid_derivative(out, i, is_output = True))

        if inp is not None:
            np.multiply(qd[i],inp)

    return qd


# initialize random weights from a normal distribution with given standard deviation:
def smart_init_weights(n, m):
    stddev = 1 / (np.sqrt(m))

    return np.random.normal(scale=stddev, size=(n,m))

