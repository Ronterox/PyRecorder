import importlib
import time
import os

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
def check_path_exists(path: str, err_message: str, fail_on_error = True) -> bool:
    exists = os.path.exists(path)
    if not exists and fail_on_error:
        print(err_message)
        exit(1)
    return exists

def print_computation_times():
    print("-"*100)
    total_time = total_import_time + total_computation_time
    print(f"Import time: {total_import_time} ({total_import_time/total_time:%} of the total time)")
    print(f"Computation time: {total_computation_time} ({total_computation_time/total_time:%} of the total time)")
