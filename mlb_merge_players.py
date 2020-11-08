import pandas as pd
import numpy as np
from datetime import datetime as dt

def mergeStartingPlayers(player, gameLogs, homeOrVisiting, playerNumber):

    playerNumberValue = homeOrVisiting + ' ' + 'starting player' + ' ' + str(playerNumber) + ' ' + 'ID'
    visitingPlayer = gameLogs[['row', 'Date', playerNumberValue]]
    visitingPlayer['yearID'] = pd.DatetimeIndex(pd.to_datetime(visitingPlayer['Date'])).year-1
    visitingPlayer = pd.merge(visitingPlayer, player, left_on=['yearID' , playerNumberValue], right_on=['yearID', 'playerID'], how="left")
    print(playerNumberValue)
    return visitingPlayer

def mergePlayers(visitingPlayers, homePlayers, playerNumber):

    homes = homePlayers.drop(columns=['yearID', 'Home starting player' + ' ' + str(playerNumber) + ' ' + 'ID', 'playerID', 'Date'])
    visitings = visitingPlayers.drop(columns=['yearID', 'Visiting starting player' + ' ' + str(playerNumber) + ' ' + 'ID', 'playerID', 'Date'])
    mergedPlayers = pd.merge(homes, visitings, on='row', suffixes=(' home player ' + str(playerNumber), ' visiting player ' + str(playerNumber)))
    return mergedPlayers


# mergeStartingPlayers(1, 'Visiting', 2, 3)

path = r'/Users/sewerynkozlowski/Desktop/HTW_2_Semester/Analytische Anwendungen/MLB-DeepLearning-Project/MLB-DeepLearning-Project'
fieldings = pd.read_csv(path+r'/Filtered/_mlb_filtered_Fielding.csv', index_col=False)
gameLogs = pd.read_csv(path+r'/Filtered/_mlb_filtered_GameLogs.csv', index_col=False)

aggregators = {}
for column in fieldings.drop(columns=['yearID','playerID']).columns:
    if column.find("average")>-1:
        aggregators[column] = 'mean'
    else:
        aggregators[column] = 'sum'

fieldings = fieldings.groupby(['yearID', 'playerID'], as_index=False).agg(aggregators)
playerNumber = 4
mergePlayers(mergeStartingPlayers(fieldings, gameLogs, 'Visiting', playerNumber), mergeStartingPlayers(fieldings, gameLogs, 'Home', playerNumber), playerNumber).to_csv(path+r'\Merged\_mlb_merged_Fieldings.csv', index = False)
