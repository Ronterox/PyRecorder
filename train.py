from utils import time_import, timeit, print_computation_times, read_and_parse

time_import('numpy')
import numpy as np

time_import('tqdm')
from tqdm import trange

FILENAME = 'osu'
SEED = "minecraft"

# TODO: Maybe normalize the image, by putting pixels together lol

xs, ys = read_and_parse(FILENAME)


def get_weights(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.random.random(size=(inp.shape[-1], out.shape[-1]))


def forward(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    # print(x.shape, w.shape)
    return x @ w


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def cost(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    err = ((y_hat - y) ** 2).mean(keepdims=True, axis=0)
    # print(err)
    return err


def backward(ws: np.ndarray, cst: np.ndarray, lr: float = 0.005) -> np.ndarray:
    return ws - lr * cst


@timeit
def setup(xs: np.ndarray, ys: np.ndarray, normalize: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(sum(map(ord, SEED)))
    print(xs.shape, ys.shape)

    if normalize:
        xmean, xstd = xs.mean(keepdims=True, axis=-1), xs.std(keepdims=True, axis=-1)
        ymean, ystd = ys.mean(keepdims=True, axis=-1), ys.std(keepdims=True, axis=-1)

        # Z-score normalization
        xs = (xs - xmean) / xstd
        ys = (ys - ymean) / ystd

    ws = get_weights(xs, ys)
    print(ws.shape)

    return xs, ys, ws


@timeit
def train(xs: np.ndarray, ys: np.ndarray, ws: np.ndarray, epochs: int = 100, lr: float = 0.005) -> None:
    for i in trange(epochs):
        fw = relu(forward(xs, ws))
        cst = cost(fw, ys)
        ws = backward(ws, cst, lr)

        if i % 10 == 0:
            print(f'\tEpoch: {i}, Cost: {cst.mean()}')

    cst = cost(forward(xs, ws), ys)
    print(f'\tFinal Cost: {cst.mean()}')


xs, ys, ws = setup(xs, ys, normalize=True)
train(xs, ys, ws, epochs=1000, lr=0.0005)

print_computation_times()
