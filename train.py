from utils import time_import, timeit, print_computation_times, read_and_parse

time_import('numpy')
import numpy as np

time_import('tqdm')
from tqdm import trange

FILENAME = 'osu'
SEED = 'minecraft'


def get_weights(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.random.random(size=(inp.shape[-1], out.shape[-1]))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def cost(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    err = ((y_hat - y) ** 2).mean(keepdims=True, axis=0)
    # print(err)
    return err


def backward(ws: np.ndarray, cst: np.ndarray, lr: float = 0.005) -> np.ndarray:
    return ws - lr * cst


@timeit
def setup(xs: np.ndarray, ys: np.ndarray, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(sum(map(ord, SEED)))
    print(xs.shape, ys.shape)

    xs.shape = (len(xs), -1)
    if normalize:
        xmean, xstd = xs.mean(keepdims=True, axis=-1), xs.std(keepdims=True, axis=-1)
        ymean, ystd = ys.mean(keepdims=True, axis=-1), ys.std(keepdims=True, axis=-1)

        # Z-score normalization
        xs = (xs - xmean) / xstd
        ys = (ys - ymean) / ystd

    return xs, ys


@timeit
def train(xs: np.ndarray, ys: np.ndarray, epochs: int = 100, lr: float = 0.005) -> None:
    ws = get_weights(xs, ys)
    for i in trange(epochs):
        fw = relu(xs @ ws)
        cst = cost(fw, ys)
        ws = backward(ws, cst, lr)

        if i % 10 == 0:
            print(f'\tEpoch: {i}, Cost: {cst.mean()}')

    cst = cost(relu(xs @ ws), ys)
    print(f'\tFinal Cost: {cst.mean()}')


if __name__ == '__main__':
    # TODO: We can generate new data by using the last X pos

    xs, ys = read_and_parse(FILENAME, 16)

    xs, ys = setup(xs, ys, normalize=True)
    train(xs, ys, epochs=1_000, lr=0.005)

    print_computation_times()
