from utils import time_import, timeit, check_path_exists, print_computation_times

time_import('numpy')
import numpy as np

time_import('math')
import math

time_import('os')
import os

time_import('replay')
from replay import parse_file_steps, TimedInputs, Second, MSecond, Action

time_import('typing')
from typing import NamedTuple

FILENAME = 'osu'

class TrainData(NamedTuple):
    sec: Second
    ms: MSecond
    bytes: np.ndarray
    action: Action
    params: list[str]

@timeit
def parse_steps(filename: str) -> TimedInputs:
    return parse_file_steps(filename)

@timeit
def parse_data(images_dir: str, filename: str = f'{FILENAME}.steps') -> list[TrainData]:
    check_path_exists(images_dir, f"Directory '{images_dir}' does not exist!")
    check_path_exists(filename, f"File '{filename}' does not exist!")

    print("Parsing steps...", filename)
    inputs = parse_file_steps(filename)

    print("Listing directory...")
    data: list[TrainData] = []
    for filename in sorted(os.listdir(images_dir), key=lambda f: f[:-5]):
        max_bytes = 0
        print("Parsing file...", filename)
        with open(os.path.join(images_dir, filename), 'rb') as image:
            b: np.ndarray = np.fromfile(image, dtype=np.uint8)
            max_bytes = max(max_bytes, b.size)

            left, right = filename.split("_")
            sec = int(left)
            ms = float(f"0.{right[:-4]}")

            (ms, action, params) = next(x for x in inputs[sec] if x.ms == ms)
            data.append(TrainData(sec, ms, b, action, params))

    print("Padding data...")
    target_bytes = 2 ** round(math.log2(max_bytes))
    for i in range(len(data)):
        b = data[i].bytes
        data[i]._replace(bytes=np.pad(b, (0, target_bytes - b.size)))

    return data

@timeit
def process_data(data_list: list[TrainData]):
    print("Processing data to dataframe...")
    df = pd.DataFrame(data_list, columns=['sec', 'ms', 'bytes', 'action', 'params'])
    print(df.head())
    return df

if __name__ == '__main__':
    train_data_raw = parse_data('screenshots')

    time_import('pandas')
    import pandas as pd

    train_df = process_data(train_data_raw)
    train_df.to_pickle(f'{FILENAME}.pkl')

    print_computation_times()