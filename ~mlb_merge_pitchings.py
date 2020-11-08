import pandas as pd
import numpy as np
from datetime import datetime as dt

def mergeVisitingPitchers(pitchers, gameLogs):
    visitingPitchers            = gameLogs[['row','Date','Visiting starting pitcher ID']]
    visitingPitchers['yearID']  = pd.DatetimeIndex(pd.to_datetime(visitingPitchers['Date'])).year-1
    visitingPitchers            = pd.merge(visitingPitchers, pitchers, left_on=['yearID','Visiting starting pitcher ID'], right_on=['yearID','playerID'], how="left")
    return visitingPitchers

def mergeHomePitchers(pitchers, gameLogs):
    homePitchers            = gameLogs[['row','Date','Home starting pitcher ID']]
    homePitchers['yearID']  = pd.DatetimeIndex(pd.to_datetime(homePitchers['Date'])).year-1
    homePitchers            = pd.merge(homePitchers, pitchers, left_on=['yearID','Home starting pitcher ID'], right_on=['yearID','playerID'], how="left")
    return homePitchers

def mergePitchers(visitingPitchers, homePitchers):
    homes          = homePitchers.drop(columns=['yearID','Home starting pitcher ID','playerID','Date'])
    visitings      = visitingPitchers.drop(columns=['yearID','Visiting starting pitcher ID','playerID','Date'])
    mergedPitchers = pd.merge(homes, visitings, on='row', suffixes=(' home pitcher',' visiting pitcher'))
    return mergedPitchers


path = r'/Users/sewerynkozlowski/Desktop/HTW_2_Semester/Analytische Anwendungen/MLB-DeepLearning-Project/MLB-DeepLearning-Project'
pitchers = pd.read_csv(path+r'/Filtered/_mlb_filtered_Pitching.csv', index_col=False)
gameLogs = pd.read_csv(path+r'/Filtered/_mlb_filtered_GameLogs.csv', index_col=False)

aggregators = {}
for column in pitchers.drop(columns=['yearID','playerID']).columns:
    if column.find("average")>-1:
        aggregators[column] = 'mean'
    else:
        aggregators[column] = 'sum'

pitchers = pitchers.groupby(['yearID','playerID'], as_index=False).agg(aggregators)
mergePitchers(mergeVisitingPitchers(pitchers, gameLogs),mergeHomePitchers(pitchers, gameLogs)).to_csv(path+r'\Merged\_mlb_merged_Pitchers.csv', index = False)