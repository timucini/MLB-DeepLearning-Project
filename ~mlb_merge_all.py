import pandas as pd
import numpy as np

path = r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Projekt'
scorings    = pd.read_csv(path+r'\Merged\_mlb_created_scoring.csv', index_col=False)
people      = pd.read_csv(path+r'\Merged\_mlb_merged_People.csv', index_col=False)
teams       = pd.read_csv(path+r'\Merged\_mlb_merged_Teams.csv', index_col=False)
managers    = pd.read_csv(path+r'\Merged\_mlb_merged_Managers.csv', index_col=False)
pitchings   = pd.read_csv(path+r'\Merged\_mlb_merged_Pitchers.csv', index_col=False)
battings    = pd.read_csv(path+r'\Merged\_mlb_merged_Batting.csv', index_col=False)
fieldings   = pd.read_csv(path+r'\Merged\_mlb_merged_Fielding.csv', index_col=False)
mlbData     = [scorings, people, teams, managers, pitchings, battings, fieldings]

pd.concat(mlbData, axis=1).dropna().to_csv(path+r'\_mlb_all_dropped.csv', index = False)