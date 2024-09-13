import importlib
import time

total_import_time = 0
total_computation_time = 0


def time_import(name: str):
    global total_import_time

    print(f"Importing {name}...")
    start = time.perf_counter()
    importlib.import_module(name)
    import_time = time.perf_counter() - start
    print(f"""Time elapsed to import {name}: {import_time}""")
    total_import_time += import_time


time_import('numpy')
import numpy as np

time_import('pandas')
import pandas as pd

time_import('os')
import os


def timeit(func):
    def wrapper(*args, **kwargs):
        global total_computation_time

        start = time.perf_counter()
        result = func(*args, **kwargs)
        computation_time = time.perf_counter() - start
        print(f"Time elapsed to run {func.__name__}: {computation_time}")
        total_computation_time += computation_time
        return result

    return wrapper


@timeit
def check_path_exists(path: str, err_message: str, fail_on_error=True) -> bool:
    exists = os.path.exists(path)
    if not exists and fail_on_error:
        print(err_message)
        exit(1)
    return exists


def parse_params(params: list[str]) -> int | list[int]:
    if ',' in params[0]:
        return list(map(int, params[0].split(',')))
    return int(params[0].strip('\n ') == 'right') + 1


@timeit
def read_and_parse(filename: str, down_sample: int = 1) -> tuple[np.ndarray, np.ndarray]:
    check_path_exists(f'{filename}.pkl', f"File '{filename}.pkl' does not exist!")

    dtype = np.float32

    df: pd.DataFrame = pd.read_pickle(f'{filename}.pkl')
    df = df[['rgb', 'action', 'params']]
    df.rgb = df.rgb.apply(lambda x: x[::down_sample, ::down_sample].astype(dtype))
    df.action = df.action.map({'Move': 1, 'ButtonUp': 2, 'ButtonDown': 3})
    df.params = df.params.map(parse_params).where(df.action == 1).ffill()
    df['output'] = df.apply(lambda row: np.array([int(x) // down_sample for x in row.params] + [row.action], dtype), axis=1)

    return np.stack(df.rgb), np.stack(df.output)


def print_computation_times():
    print("-" * 100)
    total_time = total_import_time + total_computation_time
    print(f"Import time: {total_import_time} ({total_import_time / total_time:%} of the total time)")
    print(f"Computation time: {total_computation_time} ({total_computation_time / total_time:%} of the total time)")
