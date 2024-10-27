import numpy as np

from typing import Callable

SimpleFn = Callable[[np.ndarray], np.ndarray]
MultiFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class Activation:
    def __init__(self, func: SimpleFn, prime: SimpleFn) -> None:
        self.func = func
        self.prime = prime

    def __call__(self, x: np.ndarray, prime: bool = False) -> np.ndarray:
        return self.prime(x) if prime else self.func(x)


class ErrorFn:
    def __init__(self, func: MultiFn, prime: MultiFn) -> None:
        self.func = func
        self.prime = prime

    def __call__(self, x: np.ndarray, y: np.ndarray, prime: bool = False) -> np.ndarray:
        return self.prime(x, y) if prime else self.func(x, y)


class Layer:
    def __init__(self, ninps: int, nouts: int, activation: Activation) -> None:
        self.weights = np.random.random(size=(nouts, ninps))
        self.bias = np.random.random(size=(nouts, 1))
        self.activation = activation

        self.z = np.zeros_like(self.bias)
        self.a = np.zeros_like(self.bias)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.z = self.weights @ inputs + self.bias
        self.a = self.activation(self.z)
        return self.a


class NeuralNetwork:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return self.forward(inputs)

    def forward(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        outs = []
        for inp in inputs:
            inp = inp.T
            for layer in self.layers:
                inp = layer(inp)
            outs.append(inp)
        return list(np.transpose(outs))

    def backward(self, y: np.ndarray, error: ErrorFn, lr: float = 0.05) -> None:
        layer = self.layers[-1]

        err_da = error(layer.a, y, prime=True)
        da_dz = layer.activation(layer.z, prime=True)
        dz_dw = self.layers[-2].a

        layer.weights -= (lr * err_da * da_dz * dz_dw).T

np.random.seed(sum(map(ord, 'minecraft')))

relu = Activation(lambda x: np.maximum(0, x), lambda x: x > 0)
tanh = Activation(lambda x: np.tanh(x), lambda x: 1 - np.tanh(x) ** 2)
linear = Activation(lambda x: x, lambda x: np.ones_like(x))

mse = ErrorFn(lambda x, y: (x - y) ** 2, lambda x, y: y - x)

nn = NeuralNetwork([
    Layer(2, 4, relu),
    Layer(4, 4, relu),
    Layer(4, 1, tanh)
])

samples = 1
features = 2

xs: np.ndarray = np.random.randint(0, 2, size=(samples, features))
ys = np.apply_along_axis(lambda x: x[0] ^ x[1], 1, xs)

out = nn([xs])
print(xs, '->', out,'~=', ys, '+-', mse(out, ys))

for _ in range(1000):
    nn([xs])
    nn.backward(ys, mse)

out = nn([xs])
print(xs, '->', out,'~=', ys, '+-', mse(out, ys))
