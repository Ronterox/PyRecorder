from utils import time_import, timeit, check_path_exists, print_computation_times

time_import('numpy')
import numpy as np

time_import('os')
import os

time_import('replay')
from replay import parse_file_steps, Second, MSecond, Action

time_import('typing')
from typing import NamedTuple

time_import('PIL')
from PIL import Image

class TrainData(NamedTuple):
    sec: Second
    ms: MSecond
    rgb: np.ndarray
    action: Action
    params: list[str]

@timeit
def parse_data(images_dir: str, filepath: str) -> list[TrainData]:
    check_path_exists(images_dir, f"Directory '{images_dir}' does not exist!")
    check_path_exists(filepath, f"File '{filepath}' does not exist!")

    print("Parsing steps...", filepath)
    inputs = parse_file_steps(filepath)

    print("Listing directory...")
    images_paths = sorted(os.listdir(images_dir), key=lambda f: f[:-5])
    data: list[TrainData | None] = [None] * len(images_paths)
    for i, filename in enumerate(images_paths):
        print("Parsing file...", filename)
        rgb = np.asarray(Image.open(os.path.join(images_dir, filename)))

        left, right = filename.split("_")
        sec = int(left)
        ms = float(f"0.{right[:-4]}")

        (ms, action, params) = next(x for x in inputs[sec] if x.ms == ms)
        data[i] = TrainData(sec, ms, rgb, action, params)

    return data

@timeit
def process_data(data_list: list[TrainData]):
    print("Processing data to dataframe...")
    df = pd.DataFrame(data_list, columns=['sec', 'ms', 'rgb', 'action', 'params'])
    print(df.head())
    return df

if __name__ == '__main__':
    filename = 'osu'

    train_data_raw = parse_data('screenshots', f'{filename}.steps')

    time_import('pandas')
    import pandas as pd

    train_df = process_data(train_data_raw)
    train_df.to_pickle(f'{filename}.pkl')

    print_computation_times()