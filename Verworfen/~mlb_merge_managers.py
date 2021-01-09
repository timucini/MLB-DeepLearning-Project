import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime as dt

def mergeVisitingManagers(managers, gameLogs):
    visitingManagers            = gameLogs[['row','Date','Visiting team manager ID']]
    visitingManagers['yearID']  = pd.DatetimeIndex(pd.to_datetime(visitingManagers['Date'])).year-1
    visitingManagers            = pd.merge(visitingManagers, managers, left_on=['yearID','Visiting team manager ID'], right_on=['yearID','playerID'], how="left")
    return visitingManagers

def mergeHomeManagers(managers, gameLogs):
    homeManagers            = gameLogs[['row','Date','Home team manager ID']]
    homeManagers['yearID']  = pd.DatetimeIndex(pd.to_datetime(homeManagers['Date'])).year-1
    homeManagers            = pd.merge(homeManagers, managers, left_on=['yearID','Home team manager ID'], right_on=['yearID','playerID'], how="left")
    return homeManagers

def mergeManagers(visitingManagers, homeManagers):
    homes          = homeManagers[['row','Games','Wins','Losses']]
    visitings      = visitingManagers[['row','Games','Wins','Losses']]
    mergedManagers = pd.merge(homes, visitings, on='row', suffixes=(' home manager',' visiting manager'))
    return mergedManagers


path = Path
managers = pd.read_csv(path+r'\Filtered\_mlb_filtered_Managers.csv', index_col=False)
gameLogs = pd.read_csv(path+r'\Filtered\_mlb_filtered_GameLogs.csv', index_col=False)
managers = managers.groupby(['yearID','playerID'], as_index=False)['Games','Wins','Losses'].sum()
mergeManagers(mergeVisitingManagers(managers, gameLogs),mergeHomeManagers(managers, gameLogs)).to_csv(path+r'\Merged\_mlb_merged_Managers.csv', index = False)