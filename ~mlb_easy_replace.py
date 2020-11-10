import pandas as pd
import numpy as np
import datetime

def replaceNAsManagers(managers):
    managers    = managers.groupby(['yearID','playerID'], as_index=False)['Games','Wins','Losses'].sum()
    players     = managers['playerID'].unique()
    years       = managers['yearID'].unique()
    checksum    = players.size*years.size
    fullMans    = pd.DataFrame(np.array(np.meshgrid(years, players)).T.reshape(-1,2), columns=['yearID','playerID'])
    fullMans    = pd.merge(fullMans, managers, on=['yearID','playerID'], how="left")
    fullMans    = pd.merge(fullMans[['yearID','playerID']], fullMans.groupby(['playerID']).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
    fullMans    = fullMans.fillna(0)
    print("Replaced NAs from Managers. Checksum: ",checksum==fullMans.index.size, checksum, fullMans.index.size)
    return fullMans

def replaceNAsFielding(fieldings):
    fieldings   = fieldings.groupby(['yearID','playerID'], as_index=False).sum()
    players     = fieldings['playerID'].unique()
    years       = fieldings['yearID'].unique()
    checksum    = players.size*years.size
    fullField   = pd.DataFrame(np.array(np.meshgrid(years, players)).T.reshape(-1,2), columns=['yearID','playerID'])
    fullField   = pd.merge(fullField, fieldings, on=['yearID','playerID'], how="left")
    fullField   = pd.merge(fullField[['yearID','playerID']], fullField.groupby(['playerID']).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
    fullField   = fullField.fillna(0)
    print("Replaced NAs from Fieldings. Checksum: ",checksum==fullField.index.size, checksum, fullField.index.size)
    return fullField

def replaceNAsBatting(battings):
    battings   = battings.groupby(['yearID','playerID'], as_index=False).sum()
    players     = battings['playerID'].unique()
    years       = battings['yearID'].unique()
    checksum    = players.size*years.size
    fullBatts   = pd.DataFrame(np.array(np.meshgrid(years, players)).T.reshape(-1,2), columns=['yearID','playerID'])
    fullBatts   = pd.merge(fullBatts, battings, on=['yearID','playerID'], how="left")
    fullBatts   = pd.merge(fullBatts[['yearID','playerID']], fullBatts.groupby(['playerID']).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
    fullBatts   = fullBatts.fillna(0)
    print("Replaced NAs from Battings. Checksum: ",checksum==fullBatts.index.size, checksum, fullBatts.index.size)
    return fullBatts

def replaceNAsPitching(pitchings):
    aggregators = {}
    for column in pitchings.drop(columns=['yearID','playerID']).columns:
        if column.find("average")>-1:
            aggregators[column] = 'mean'
        else:
            aggregators[column] = 'sum'
    pitchings   = pitchings.groupby(['yearID','playerID'], as_index=False).agg(aggregators)
    players     = pitchings['playerID'].unique()
    years       = pitchings['yearID'].unique()
    checksum    = players.size*years.size
    fullPitch   = pd.DataFrame(np.array(np.meshgrid(years, players)).T.reshape(-1,2), columns=['yearID','playerID'])
    fullPitch   = pd.merge(fullPitch, pitchings, on=['yearID','playerID'], how="left")
    fullPitch   = pd.merge(fullPitch[['yearID','playerID']], fullPitch.groupby(['playerID']).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
    fullPitch   = fullPitch.fillna(0)
    print("Replaced NAs from Pitchings. Checksum: ",checksum==fullPitch.index.size, checksum, fullPitch.index.size)
    return fullPitch

def replaceNAsTeams(teams):
    teams['division C'] = (teams['Division']=="C")
    teams['division E'] = (teams['Division']=="E")
    teams['division W'] = (teams['Division']=="W")
    teams['Division winner']    = (teams['Division winner']=='Y')
    teams['League winner']      = (teams['League winner']=='Y')
    teams['World series winner']= (teams['World series winner']=='Y')
    aggregators = {}
    for column in teams.drop(columns=['yearID','teamID']).columns:
        if (column.find("average")>-1) or (column.find("percentage")>-1):
            aggregators[column] = 'mean'
        elif (column.find("winner")>-1) or (column.find("division")>-1):
            aggregators[column] = 'max'
        else:
            aggregators[column] = 'sum'
    teams       = teams.groupby(['yearID','teamID'], as_index=False).agg(aggregators)
    players     = teams['teamID'].unique()
    years       = teams['yearID'].unique()
    checksum    = players.size*years.size
    fullTeams   = pd.DataFrame(np.array(np.meshgrid(years, players)).T.reshape(-1,2), columns=['yearID','teamID'])
    fullTeams   = pd.merge(fullTeams, teams, on=['yearID','teamID'], how="left")
    fullTeams   = pd.merge(fullTeams[['yearID','teamID']], fullTeams.groupby(['teamID']).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
    fullTeams['division C'] = fullTeams['division C'].fillna(False)
    fullTeams['division E'] = fullTeams['division E'].fillna(False)
    fullTeams['division W'] = fullTeams['division W'].fillna(False)
    fullTeams['Division winner']    = fullTeams['Division winner'].fillna(False)
    fullTeams['League winner']      = fullTeams['League winner'].fillna(False)
    fullTeams['World series winner']= fullTeams['World series winner'].fillna(False)
    fullTeams   = fullTeams.drop(columns=['Division'])
    fullTeams   = fullTeams.fillna(0)
    print("Replaced NAs from Teams. Checksum: ",checksum==fullTeams.index.size, checksum, fullTeams.index.size)
    return fullTeams

def encodePeople(people):
    people['bats right'] = (people['bats']=="R") | (people['bats']=="B")
    people['bats left'] = (people['bats']=="L") | (people['bats']=="B")
    people['throws right'] = people['throws']=="R"
    people = people.drop(columns=['bats','throws'])
    print("People encoded")
    return people

def encodeGames(gameLogs):
    gameLogs['Visiting league AL'] = gameLogs['Visiting league']=="AL"
    gameLogs['Home league AL']     = gameLogs['Home league']=="AL"
    gameLogs = gameLogs.drop(columns=['Visiting league','Home league'])
    print("GameLogs encoded")
    return gameLogs

path = r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Projekt'
gameLogs    = pd.read_csv(path+r'\Filtered\_mlb_filtered_GameLogs.csv', index_col=False)
people      = pd.read_csv(path+r'\Filtered\_mlb_filtered_People.csv', index_col=False)
teams       = pd.read_csv(path+r'\Filtered\_mlb_filtered_Teams.csv', index_col=False)
managers    = pd.read_csv(path+r'\Filtered\_mlb_filtered_Managers.csv', index_col=False)
pitchings   = pd.read_csv(path+r'\Filtered\_mlb_filtered_Pitching.csv', index_col=False)
battings    = pd.read_csv(path+r'\Filtered\_mlb_filtered_Batting.csv', index_col=False)
fieldings   = pd.read_csv(path+r'\Filtered\_mlb_filtered_Fielding.csv', index_col=False)

encodeGames(gameLogs).to_csv(path+r'\Replaced\_mlb_encoded_GameLogs.csv', index = False)
encodePeople(people).to_csv(path+r'\Replaced\_mlb_encoded_People.csv', index = False)
replaceNAsTeams(teams).to_csv(path+r'\Replaced\_mlb_replaced_Teams.csv', index = False)
replaceNAsPitching(pitchings).to_csv(path+r'\Replaced\_mlb_replaced_Pitching.csv', index = False)
replaceNAsBatting(battings).to_csv(path+r'\Replaced\_mlb_replaced_Batting.csv', index = False)
replaceNAsFielding(fieldings).to_csv(path+r'\Replaced\_mlb_replaced_Fielding.csv', index = False)
replaceNAsManagers(managers).to_csv(path+r'\Replaced\_mlb_replaced_Managers.csv', index = False)