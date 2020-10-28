import pandas as pd
from pathlib import Path

data_folder = Path("../MLP_Win_Prediction/Data")

base_data_file = data_folder / "base_data.xlsx"

pd.read_excel(base_data_file)



