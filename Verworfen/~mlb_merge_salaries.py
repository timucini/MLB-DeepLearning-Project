import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime as dt

def mergeSalaries(IDColumn, gameLogs, salaries):
    merged = pd.merge(gameLogs[['row','yearID',IDColumn]], salaries, how="left", left_on=['yearID',IDColumn], right_on=['yearID','playerID'])
    newCol = IDColumn.replace(" ID"," salary")
    merged = merged.rename(columns={"salary":newCol})
    return merged[['row',newCol]]

path = Path
salaries = pd.read_csv(path+r'\Filtered\_mlb_filtered_Salaries.csv', index_col=False)
gameLogs = pd.read_csv(path+r'\Filtered\_mlb_filtered_GameLogs.csv', index_col=False)

gameLogs['yearID'] = pd.DatetimeIndex(pd.to_datetime(gameLogs['Date'])).year-1
salaries = salaries.groupby(['yearID','playerID'], as_index=False)['salary'].mean()

allSalaries = []
for idC in gameLogs.columns:
    if idC.find(" ID")>-1:
        allSalaries.append(mergeSalaries(idC, gameLogs, salaries))

mergedSalaries = gameLogs['row']
for merSal in allSalaries:
    mergedSalaries = pd.merge(mergedSalaries, merSal, how="left", on='row')
mergedSalaries.to_csv(path+r'\Merged\_mlb_merged_Salaries.csv', index = False)