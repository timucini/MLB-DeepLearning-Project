import pandas as pd

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
    mergedPlayers = pd.merge(homes, visitings, on='row',  suffixes=(' Visiting starting player ' + str(playerNumber), ' Home starting player ' + str(playerNumber)))
    return mergedPlayers

def mergeThePlayers(playerNumber):
    homePlayers = mergeStartingPlayers(fieldings, gameLogs, 'Home', playerNumber)
    visitingPlayers = mergeStartingPlayers(fieldings, gameLogs, 'Visiting', playerNumber)
    homePlayers2 = mergeStartingPlayers(fieldings, gameLogs, 'Home', playerNumber+1)
    visitingPlayers2 = mergeStartingPlayers(fieldings, gameLogs, 'Visiting', playerNumber+1)
    merge1 = mergePlayers(visitingPlayers, homePlayers, playerNumber)
    merge2 = mergePlayers(visitingPlayers2, homePlayers2, playerNumber+1)
    merge12 = pd.merge(merge1, merge2, how="left", on="row")
    return merge12

def mergeAllFrames(fieldings, gameLogs):
    merge12 = mergeThePlayers(1)
    merge34 = mergeThePlayers(3)
    merge1234 = pd.merge(merge12, merge34, how="left", on="row")
    merge56 = mergeThePlayers(5)
    merge78 = mergeThePlayers(7)
    merge5678 = pd.merge(merge56, merge78, how="left", on="row")
    merge18 = pd.merge(merge1234, merge5678, how="left", on="row")
    homePlayers9 = mergeStartingPlayers(fieldings, gameLogs, 'Home', 9)
    visitingPlayers9 = mergeStartingPlayers(fieldings, gameLogs, 'Visiting', 9)
    merge9 = mergePlayers(visitingPlayers9, homePlayers9, 9)
    merge19 = pd.merge(merge18, merge9, how="left", on="row")
    merge19.to_csv(path + r'\Merged\_mlb_merged_Batting.csv', index=False)

path = r'/Users/sewerynkozlowski/Desktop/HTW_2_Semester/Analytische Anwendungen/MLB-DeepLearning-Project/MLB-DeepLearning-Project'
fieldings = pd.read_csv(path+r'/Filtered/_mlb_filtered_Batting.csv', index_col=False)
gameLogs = pd.read_csv(path+r'/Filtered/_mlb_filtered_GameLogs.csv', index_col=False)

aggregators = {}
for column in fieldings.drop(columns=['yearID','playerID']).columns:
    if column.find("average")>-1:
        aggregators[column] = 'mean'
    else:
        aggregators[column] = 'sum'

battings = fieldings.groupby(['yearID', 'playerID'], as_index=False).agg(aggregators)

mergeAllFrames(battings, gameLogs)
