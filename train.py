import pandas as pd

from  utils import check_path_exists

FILENAME = 'osu'

check_path_exists(f'{FILENAME}.pkl', f"File '{FILENAME}.pkl' does not exist!")

train_data: pd.DataFrame = pd.read_pickle(f'{FILENAME}.pkl')

print(train_data.info())