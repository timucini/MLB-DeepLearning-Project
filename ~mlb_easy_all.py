import pandas as pd
import numpy as np

path = r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Projekt'
stats       = pd.read_csv(path+r'\Remerged\_mlb_game_stats.csv', index_col=False)
people      = pd.read_csv(path+r'\Remerged\_mlb_remerged_People.csv', index_col=False)
teams       = pd.read_csv(path+r'\Remerged\_mlb_remerged_Teams.csv', index_col=False)
managers    = pd.read_csv(path+r'\Remerged\_mlb_remerged_Managers.csv', index_col=False)
pitchings   = pd.read_csv(path+r'\Remerged\_mlb_remerged_Pitchers.csv', index_col=False)
battings    = pd.read_csv(path+r'\Remerged\_mlb_remerged_Batting.csv', index_col=False)
fieldings   = pd.read_csv(path+r'\Remerged\_mlb_remerged_Fielding.csv', index_col=False)
mlbData     = [teams, managers, people, pitchings, battings, fieldings]
mlbAll      = stats

for data in mlbData:
    mlbAll = pd.merge(mlbAll, data, on='row', how="left")
mlbAll.to_csv(path+r'\_mlb_remerged_all.csv', index = False)

generalColumns = ['Date','Visiting team','Home team','Visiting score','Home score','Home team win']
mlbAll      = mlbAll.dropna().reset_index(drop=True).drop(columns='row')
targets     = mlbAll[generalColumns]
predictors  = mlbAll.drop(columns=generalColumns)
targets.to_csv(path+r'\_mlb_targets.csv', index = False)
predictors.to_csv(path+r'\_mlb_predictors.csv', index = False)