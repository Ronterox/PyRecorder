from utils import check_path_exists, time_import, timeit, print_computation_times

time_import('pandas')
import pandas as pd

# time_import('numpy')
# import numpy as np

FILENAME = 'osu'


def parse_params(params: list[str]) -> int | list[int]:
    if ',' in params[0]:
        return list(map(int, params[0].split(',')))
    return int(params[0].strip('\n ') == 'right') + 1


@timeit
def read_and_parse(filename: str) -> pd.DataFrame:
    check_path_exists(f'{filename}.pkl', f"File '{filename}.pkl' does not exist!")

    df: pd.DataFrame = pd.read_pickle(f'{filename}.pkl')
    df = df[['bytes', 'action', 'params']]
    df.action = df.action.map({'Move': 1, 'ButtonUp': 2, 'ButtonDown': 3})
    df.params = df.params.map(parse_params).where(df.action == 1).ffill()
    df['output'] = df.apply(lambda row: row.params + [row.action], axis=1)

    return df[['bytes', 'output']]


train_data = read_and_parse(FILENAME)
print(train_data.head())

x = train_data.bytes
y = train_data.output

print_computation_times()
