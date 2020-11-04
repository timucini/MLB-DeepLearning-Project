import pandas as pd
import numpy as np
from datetime import datetime as dt

def mergePeople(IDColumn, gameLogs, people):
    merged = pd.merge(gameLogs[['row','Date',IDColumn]], people, how="left", left_on=[IDColumn], right_on=['playerID'])
    merged['age'] = (pd.to_datetime(merged['Date']) - pd.to_datetime(merged['birthdate'])) / np.timedelta64(1, 'Y')
    newColumns = {"age":IDColumn.replace(" ID"," "+" age")}
    for column in people.drop(columns=['playerID','birthdate']).columns:
        newColumns[column] = IDColumn.replace(" ID"," "+str(column))
    merged = merged.rename(columns=newColumns)
    return merged[['row']+list(newColumns.values())]

path = r'C:\Users\DonBrezz\Documents\GitHub\MLB-DeepLearning-Project'
people = pd.read_csv(path+r'\Filtered\_mlb_filtered_People.csv', index_col=False)
gameLogs = pd.read_csv(path+r'\Filtered\_mlb_filtered_GameLogs.csv', index_col=False)

allPeople = []
for idC in gameLogs.columns:
    if idC.find("starting")>-1:
        allPeople.append(mergePeople(idC, gameLogs, people))

mergedPeople = gameLogs['row']
for merSal in allPeople:
    mergedPeople = pd.merge(mergedPeople, merSal, how="left", on='row')
mergedPeople.to_csv(path+r'\Merged\_mlb_merged_People.csv', index = False)