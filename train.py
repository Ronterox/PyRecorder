import importlib
import time

total_import_time = 0
total_computation_time = 0

def time_import(name):
    global total_import_time

    print(f"Importing {name}...")
    start = time.perf_counter()
    importlib.import_module(name)
    import_time = time.perf_counter() - start
    print(f"""Time elapsed to import {name}: {import_time}""")
    total_import_time += import_time

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

time_import('numpy')
import numpy as np

time_import('math')
import math

time_import('os')
import os

@timeit
def parse_data(base_dir):
    print("Running script...")
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' does not exist!")
        exit(1)

    print("Listing directory")
    data = []
    for filename in sorted(os.listdir(base_dir), key=lambda f: f[:-5]):
        max_bytes = 0
        with open(os.path.join(base_dir, filename), 'rb') as image:
            b = np.fromfile(image, dtype=np.uint8)
            max_bytes = max(max_bytes, b.size)

            left, right = filename.split("_")
            sec = int(left)
            ms = float(f"0.{right[:-4]}")

            data.append((sec, ms, b))

    target_bytes = 2 ** round(math.log2(max_bytes))

    for i in range(len(data)):
        (sec, ms, b) = data[i]
        b = np.pad(b, (0, target_bytes - b.size))
        data[i] = (sec, ms, b)

    return data

data = parse_data('screenshots')

time_import('pandas')
import pandas as pd

def process_data(dict_data: dict):
    df = pd.DataFrame(dict_data, columns=['sec', 'ms', 'bytes'])
    print(df)
    return df

process_data(data)

print("-"*100)
total_time = total_import_time + total_computation_time
print(f"Import time: {total_import_time} ({total_import_time/total_time:%} of the total time)")
print(f"Computation time: {total_computation_time} ({total_computation_time/total_time:%} of the total time)")