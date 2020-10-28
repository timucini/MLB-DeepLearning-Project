import xlrd
from pathlib import Path

data_folder = Path("./Data")

base_data_file = data_folder / "base_data.xlsx"

base_data_workbook = xlrd.open_workbook(base_data_file)

base_data_worksheet = base_data_workbook.sheet_by_name("Teams")

print(base_data_worksheet.cell_value(2, 3))
