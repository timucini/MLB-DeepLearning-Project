import pandas as pd
import numpy as np
import datetime

def replaceNAsManagers(managers, gameLogs, default=True):
    onlyMans    = None
    for playerColumn in gameLogs.columns:
        if playerColumn.find("manager")>-1:
            players = gameLogs[['Date',playerColumn]]
            players['yearID'] = pd.DatetimeIndex(pd.to_datetime(players['Date'])).year-1
            players = players.rename(columns={playerColumn:"playerID"})
            onlyMans = pd.concat([onlyMans, players]).drop(columns='Date').drop_duplicates().dropna().reset_index(drop=True)
    managers    = managers.groupby(['yearID','playerID'], as_index=False)['Games','Wins','Losses'].sum()
    players     = managers['playerID'].unique()
    years       = managers['yearID'].unique()
    players     = np.array(list(dict.fromkeys(players.tolist()+onlyMans['playerID'].unique().tolist())))
    years       = np.array(list(dict.fromkeys(years.tolist()+onlyMans['yearID'].unique().tolist())))
    fullMans    = pd.DataFrame(np.array(np.meshgrid(years, players)).T.reshape(-1,2), columns=['yearID','playerID'])
    fullMans['yearID'] = pd.to_numeric(fullMans['yearID'])
    fullMans    = pd.merge(fullMans, managers, on=['yearID','playerID'], how="left")
    fullMans    = pd.merge(fullMans[['yearID','playerID']], fullMans.groupby(['playerID']).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
    if default:
        fullMans    = fullMans.fillna(0)
    fullMans    = pd.merge(onlyMans, fullMans, on=['yearID','playerID'], how="left")
    return fullMans

def replaceNAsFielding(fieldings, gameLogs, default=True):
    onlyField   = None
    for playerColumn in gameLogs.columns:
        if playerColumn.find("player")>-1:
            players = gameLogs[['Date',playerColumn]]
            players['yearID'] = pd.DatetimeIndex(pd.to_datetime(players['Date'])).year-1
            players = players.rename(columns={playerColumn:"playerID"})
            onlyField = pd.concat([onlyField, players]).drop(columns='Date').drop_duplicates().dropna().reset_index(drop=True)
    fieldings   = fieldings.groupby(['yearID','playerID'], as_index=False).sum()
    players     = fieldings['playerID'].unique()
    years       = fieldings['yearID'].unique()
    players     = np.array(list(dict.fromkeys(players.tolist()+onlyField['playerID'].unique().tolist())))
    years       = np.array(list(dict.fromkeys(years.tolist()+onlyField['yearID'].unique().tolist())))
    fullField   = pd.DataFrame(np.array(np.meshgrid(years, players)).T.reshape(-1,2), columns=['yearID','playerID'])
    fullField['yearID'] = pd.to_numeric(fullField['yearID'])
    fullField   = pd.merge(fullField, fieldings, on=['yearID','playerID'], how="left")
    fullField   = pd.merge(fullField[['yearID','playerID']], fullField.groupby(['playerID']).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
    if default:
        fullField   = fullField.fillna(0)
    fullField   = pd.merge(onlyField, fullField, on=['yearID','playerID'], how="left")
    return fullField

def replaceNAsBatting(battings, gameLogs, default=True):
    onlyBatts   = None
    for playerColumn in gameLogs.columns:
        if playerColumn.find("player")>-1:
            players = gameLogs[['Date',playerColumn]]
            players['yearID'] = pd.DatetimeIndex(pd.to_datetime(players['Date'])).year-1
            players = players.rename(columns={playerColumn:"playerID"})
            onlyBatts = pd.concat([onlyBatts, players]).drop(columns='Date').drop_duplicates().dropna().reset_index(drop=True)
    battings   = battings.groupby(['yearID','playerID'], as_index=False).sum()
    players     = battings['playerID'].unique()
    years       = battings['yearID'].unique()
    players     = np.array(list(dict.fromkeys(players.tolist()+onlyBatts['playerID'].unique().tolist())))
    years       = np.array(list(dict.fromkeys(years.tolist()+onlyBatts['yearID'].unique().tolist())))
    fullBatts   = pd.DataFrame(np.array(np.meshgrid(years, players)).T.reshape(-1,2), columns=['yearID','playerID'])
    fullBatts['yearID'] = pd.to_numeric(fullBatts['yearID']) 
    fullBatts   = pd.merge(fullBatts, battings, on=['yearID','playerID'], how="left")
    fullBatts   = pd.merge(fullBatts[['yearID','playerID']], fullBatts.groupby(['playerID']).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
    if default:
        fullBatts   = fullBatts.fillna(0)
    fullBatts   = pd.merge(onlyBatts, fullBatts, on=['yearID','playerID'], how="left")
    return fullBatts

def replaceNAsPitching(pitchings, gameLogs, default=True):
    onlyPitch   = None
    for playerColumn in gameLogs.columns:
        if playerColumn.find("starting pitcher")>-1:
            players = gameLogs[['Date',playerColumn]]
            players['yearID'] = pd.DatetimeIndex(pd.to_datetime(players['Date'])).year-1
            players = players.rename(columns={playerColumn:"playerID"})
            onlyPitch = pd.concat([onlyPitch, players]).drop(columns='Date').drop_duplicates().dropna().reset_index(drop=True)
    aggregators = {}
    for column in pitchings.drop(columns=['yearID','playerID']).columns:
        if column.find("average")>-1:
            aggregators[column] = 'mean'
        else:
            aggregators[column] = 'sum'
    pitchings   = pitchings.groupby(['yearID','playerID'], as_index=False).agg(aggregators)
    players     = pitchings['playerID'].unique()
    years       = pitchings['yearID'].unique()
    players     = np.array(list(dict.fromkeys(players.tolist()+onlyPitch['playerID'].unique().tolist())))
    years       = np.array(list(dict.fromkeys(years.tolist()+onlyPitch['yearID'].unique().tolist())))
    fullPitch   = pd.DataFrame(np.array(np.meshgrid(years, players)).T.reshape(-1,2), columns=['yearID','playerID'])
    fullPitch['yearID'] = pd.to_numeric(fullPitch['yearID'])
    fullPitch   = pd.merge(fullPitch, pitchings, on=['yearID','playerID'], how="left")
    fullPitch   = pd.merge(fullPitch[['yearID','playerID']], fullPitch.groupby(['playerID']).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
    if default:
        fullPitch   = fullPitch.fillna(0)
    fullPitch   = pd.merge(onlyPitch, fullPitch, on=['yearID','playerID'], how="left")
    return fullPitch

def replaceNAsTeams(teams, gameLogs, default=True):
    onlyTeams   = None
    for teamsColumn in ['Visiting team', 'Home team']:
        team = gameLogs[['Date',teamsColumn]]
        team['yearID'] = pd.DatetimeIndex(pd.to_datetime(team['Date'])).year-1
        team = team.rename(columns={teamsColumn:"teamID"})
        onlyTeams = pd.concat([onlyTeams, team]).drop(columns='Date').drop_duplicates().dropna().reset_index(drop=True)
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
    teamids     = teams['teamID'].unique()
    years       = teams['yearID'].unique()
    teamids     = np.array(list(dict.fromkeys(teamids.tolist()+onlyTeams['teamID'].unique().tolist())))
    years       = np.array(list(dict.fromkeys(years.tolist()+onlyTeams['yearID'].unique().tolist())))
    fullTeams   = pd.DataFrame(np.array(np.meshgrid(years, teamids)).T.reshape(-1,2), columns=['yearID','teamID'])
    fullTeams['yearID'] = pd.to_numeric(fullTeams['yearID'])
    fullTeams   = pd.merge(fullTeams, teams, on=['yearID','teamID'], how="left")
    fullTeams   = pd.merge(fullTeams[['yearID','teamID']], fullTeams.groupby(['teamID']).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
    fullTeams['division C'] = fullTeams['division C'].fillna(False)
    fullTeams['division E'] = fullTeams['division E'].fillna(False)
    fullTeams['division W'] = fullTeams['division W'].fillna(False)
    fullTeams['Division winner']    = fullTeams['Division winner'].fillna(False)
    fullTeams['League winner']      = fullTeams['League winner'].fillna(False)
    fullTeams['World series winner']= fullTeams['World series winner'].fillna(False)
    fullTeams   = fullTeams.drop(columns=['Division'])
    fullTeams['Pythagorean expectation'] = (fullTeams['Runs scored']**1.83)/(fullTeams['Runs scored']**1.83+fullTeams['Opponents runs scored']**1.83)
    if default:
        fullTeams   = fullTeams.fillna(0)
    fullTeams    = pd.merge(onlyTeams, fullTeams, on=['yearID','teamID'], how="left")
    return fullTeams

def encodePeople(people, gameLogs):
    onlyPeopl   = []
    for peopleColumn in gameLogs.columns:
        if peopleColumn.find("ID")>-1:
            onlyPeopl = onlyPeopl + gameLogs[peopleColumn].unique().tolist()
    people = people[people['playerID'].isin(onlyPeopl)]
    people['bats right'] = (people['bats']=="R") | (people['bats']=="B")
    people['bats left'] = (people['bats']=="L") | (people['bats']=="B")
    people['throws right'] = people['throws']=="R"
    people = people.drop(columns=['bats','throws'])
    #print("People encoded")
    return people

def encodeGames(gameLogs, minYear):
    gameLogs = gameLogs[pd.to_datetime(gameLogs['Date'])>=datetime.datetime(minYear,1,1)]
    gameLogs['Visiting league AL'] = gameLogs['Visiting league']=="AL"
    gameLogs['Home league AL']     = gameLogs['Home league']=="AL"
    gameLogs = gameLogs.drop(columns=['Visiting league','Home league'])
    gameLogs = gameLogs.dropna().reset_index(drop=True)
    gameLogs['row'] = range(0,gameLogs.index.size)
    gameLogs = gameLogs[(gameLogs.columns[-1:].tolist()+gameLogs.columns[:-1].tolist())]
    return gameLogs

path = r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Projekt'
gameLogs    = pd.read_csv(path+r'\Filtered\_mlb_filtered_GameLogs.csv', index_col=False)
people      = pd.read_csv(path+r'\Filtered\_mlb_filtered_People.csv', index_col=False)
teams       = pd.read_csv(path+r'\Filtered\_mlb_filtered_Teams.csv', index_col=False)
managers    = pd.read_csv(path+r'\Filtered\_mlb_filtered_Managers.csv', index_col=False)
pitchings   = pd.read_csv(path+r'\Filtered\_mlb_filtered_Pitching.csv', index_col=False)
battings    = pd.read_csv(path+r'\Filtered\_mlb_filtered_Batting.csv', index_col=False)
fieldings   = pd.read_csv(path+r'\Filtered\_mlb_filtered_Fielding.csv', index_col=False)

people      = encodePeople(people, gameLogs)
teams       = replaceNAsTeams(teams, gameLogs)
managers    = replaceNAsManagers(managers, gameLogs, False)
pitchings   = replaceNAsPitching(pitchings, gameLogs, False)
battings    = replaceNAsBatting(battings, gameLogs, False)
fieldings   = replaceNAsFielding(fieldings, gameLogs, False)
yearIndicators = [teams, managers, fieldings, pitchings, battings]

minYears = []
for indicator in yearIndicators:
    minYears.append(min(indicator['yearID'].unique()))
minYear = max(minYears)

encodeGames(gameLogs, minYear).to_csv(path+r'\Replaced\_mlb_encoded_GameLogs.csv', index = False)
people.to_csv(path+r'\Replaced\_mlb_encoded_People.csv', index = False)
teams.to_csv(path+r'\Replaced\_mlb_replaced_Teams.csv', index = False)
managers.to_csv(path+r'\Replaced\_mlb_replaced_Managers.csv', index = False)
pitchings.to_csv(path+r'\Replaced\_mlb_replaced_Pitching.csv', index = False)
battings.to_csv(path+r'\Replaced\_mlb_replaced_Batting.csv', index = False)
fieldings.to_csv(path+r'\Replaced\_mlb_replaced_Fielding.csv', index = False)