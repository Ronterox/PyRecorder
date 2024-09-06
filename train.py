from utils import check_path_exists, time_import, timeit, print_computation_times

time_import('pandas')
import pandas as pd

time_import('numpy')
import numpy as np

time_import('tqdm')
from tqdm import trange

FILENAME = 'osu'
SEED = "minecraft"


def parse_params(params: list[str]) -> int | list[int]:
    if ',' in params[0]:
        return list(map(int, params[0].split(',')))
    return int(params[0].strip('\n ') == 'right') + 1


@timeit
def read_and_parse(filename: str) -> pd.DataFrame:
    check_path_exists(f'{filename}.pkl', f"File '{filename}.pkl' does not exist!")

    dtype = np.float32

    df: pd.DataFrame = pd.read_pickle(f'{filename}.pkl')
    df = df[['bytes', 'action', 'params']]
    df.bytes = df.bytes.apply(lambda x: x.astype(dtype))
    df.action = df.action.map({'Move': 1, 'ButtonUp': 2, 'ButtonDown': 3})
    df.params = df.params.map(parse_params).where(df.action == 1).ffill()
    df['output'] = df.apply(lambda row: np.array(row.params + [row.action], dtype), axis=1)

    return df[['bytes', 'output']]


# TODO: Maybe normalize the image, by putting pixels together lol

train_data = read_and_parse(FILENAME)

xs = np.stack(train_data.bytes)
ys = np.stack(train_data.output)

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
