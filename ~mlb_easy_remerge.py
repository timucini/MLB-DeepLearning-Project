import pandas as pd
import numpy as np
from datetime import datetime as dt

def mergeManagers(managers, gameLogs):
    #Get visiting managers
    visitingManagers            = gameLogs[['row','Date','Visiting team manager ID']]
    visitingManagers['yearID']  = pd.DatetimeIndex(pd.to_datetime(visitingManagers['Date'])).year-1
    visitingManagers            = pd.merge(visitingManagers, managers, left_on=['yearID','Visiting team manager ID'], right_on=['yearID','playerID'], how="left")
    #Get home managers
    homeManagers            = gameLogs[['row','Date','Home team manager ID']]
    homeManagers['yearID']  = pd.DatetimeIndex(pd.to_datetime(homeManagers['Date'])).year-1
    homeManagers            = pd.merge(homeManagers, managers, left_on=['yearID','Home team manager ID'], right_on=['yearID','playerID'], how="left")
    #Merge managers
    homes          = homeManagers[['row','Games','Wins','Losses']]
    visitings      = visitingManagers[['row','Games','Wins','Losses']]
    merged         = pd.merge(homes, visitings, on='row', suffixes=(' home manager',' visiting manager'))
    print("Merged Managers. Checksum: ", gameLogs.index.size==merged.index.size, gameLogs.index.size, merged.index.size)
    return merged

def mergePitchings(pitchers, gameLogs):
    #Get visiting pitchers
    visitingPitchers            = gameLogs[['row','Date','Visiting starting pitcher ID']]
    visitingPitchers['yearID']  = pd.DatetimeIndex(pd.to_datetime(visitingPitchers['Date'])).year-1
    visitingPitchers            = pd.merge(visitingPitchers, pitchers, left_on=['yearID','Visiting starting pitcher ID'], right_on=['yearID','playerID'], how="left")
    #Get home pitchers
    homePitchers            = gameLogs[['row','Date','Home starting pitcher ID']]
    homePitchers['yearID']  = pd.DatetimeIndex(pd.to_datetime(homePitchers['Date'])).year-1
    homePitchers            = pd.merge(homePitchers, pitchers, left_on=['yearID','Home starting pitcher ID'], right_on=['yearID','playerID'], how="left")
    #Merge pitchers
    homes          = homePitchers.drop(columns=['yearID','Home starting pitcher ID','playerID','Date'])
    visitings      = visitingPitchers.drop(columns=['yearID','Visiting starting pitcher ID','playerID','Date'])
    merged         = pd.merge(homes, visitings, on='row', suffixes=(' home pitcher',' visiting pitcher'))
    print("Merged Pitchings. Checksum: ", gameLogs.index.size==merged.index.size, gameLogs.index.size, merged.index.size)
    return merged

def mergePeople(people, gameLogs):
    #Merge people
    allPeople = []
    for IDColumn in gameLogs.columns:
        if IDColumn.find("starting")>-1:
            merged = pd.merge(gameLogs[['row','Date',IDColumn]], people, how="left", left_on=[IDColumn], right_on=['playerID'])
            merged['age'] = (pd.to_datetime(merged['Date']) - pd.to_datetime(merged['birthdate'])) / np.timedelta64(1, 'Y')
            newColumns = {"age":IDColumn.replace(" ID"," "+" age")}
            for column in people.drop(columns=['playerID','birthdate']).columns:
                newColumns[column] = IDColumn.replace(" ID"," "+str(column))
            merged = merged.rename(columns=newColumns)
            allPeople.append(merged[['row']+list(newColumns.values())])
    merged = gameLogs['row']
    for merSal in allPeople:
        merged = pd.merge(merged, merSal, how="left", on='row')
    print("Merged People. Checksum: ", gameLogs.index.size==merged.index.size, gameLogs.index.size, merged.index.size)
    return merged

def mergeTeams(teams, gameLogs):
    #Get visiting teams
    visitingTeams            = gameLogs[['row','Date','Visiting team','Visiting league AL']]
    visitingTeams['yearID']  = pd.DatetimeIndex(pd.to_datetime(visitingTeams['Date'])).year-1
    visitingTeams            = pd.merge(visitingTeams, teams, left_on=['yearID','Visiting team'], right_on=['yearID','teamID'], how="left")
    #Get home teams
    homeTeams            = gameLogs[['row','Date','Home team','Home league AL']]
    homeTeams['yearID']  = pd.DatetimeIndex(pd.to_datetime(homeTeams['Date'])).year-1
    homeTeams            = pd.merge(homeTeams, teams, left_on=['yearID','Home team'], right_on=['yearID','teamID'], how="left")
    #Merge teams
    homes          = homeTeams.drop(columns=['yearID','teamID'])
    visitings      = visitingTeams.drop(columns=['yearID','teamID'])
    merged         = pd.merge(homes, visitings, on='row', suffixes=(' home team',' visiting team'))
    print("Merged Teams. Checksum: ", gameLogs.index.size==merged.index.size, gameLogs.index.size, merged.index.size)
    return merged

def mergeFieldings(fieldings, gameLogs):
    gameLogs['yearID'] = pd.DatetimeIndex(pd.to_datetime(gameLogs['Date'])).year-1
    allPlayers = []
    for playerColumn in gameLogs.columns:
        if playerColumn.find("starting")>-1:
            merged = pd.merge(gameLogs[['row','yearID',playerColumn]], fieldings, how="left", left_on=[playerColumn,'yearID'], right_on=['playerID','yearID'])
            newColumns = {}
            for column in fieldings.drop(columns=['playerID','yearID']).columns:
                newColumns[column] = playerColumn.replace(" ID"," "+str(column))
            merged = merged.rename(columns=newColumns)
            allPlayers.append(merged[['row']+list(newColumns.values())])
    merged = gameLogs['row']
    for playerData in allPlayers:
        merged = pd.merge(merged, playerData, how="left", on='row')
    print("Merged Fieldings. Checksum: ", gameLogs.index.size==merged.index.size, gameLogs.index.size, merged.index.size)
    return merged

def mergeBattings(battings, gameLogs):
    gameLogs['yearID'] = pd.DatetimeIndex(pd.to_datetime(gameLogs['Date'])).year-1
    allPlayers = []
    for playerColumn in gameLogs.columns:
        if playerColumn.find("starting")>-1:
            merged = pd.merge(gameLogs[['row','yearID',playerColumn]], battings, how="left", left_on=[playerColumn,'yearID'], right_on=['playerID','yearID'])
            newColumns = {}
            for column in battings.drop(columns=['playerID','yearID']).columns:
                newColumns[column] = playerColumn.replace(" ID"," "+str(column))
            merged = merged.rename(columns=newColumns)
            allPlayers.append(merged[['row']+list(newColumns.values())])
    merged = gameLogs['row']
    for playerData in allPlayers:
        merged = pd.merge(merged, playerData, how="left", on='row')
    print("Merged Battings. Checksum: ", gameLogs.index.size==merged.index.size, gameLogs.index.size, merged.index.size)
    return merged

def createScorings(gameLogs):
    scoreLogs = gameLogs[['row','Visiting team','Home team','Visiting score','Home score']]
    scoreLogs['Home team win'] = scoreLogs['Home score']>scoreLogs['Visiting score']
    scoreLogs['Home team odd'] = (scoreLogs['Home score'].replace(0,1))/(scoreLogs['Visiting score'].replace(0,1))
    homeTeams = {}
    for team in scoreLogs['Home team'].unique():
        homeTeams[team] = scoreLogs[scoreLogs['Home team']==team]
    vistTeams = {}
    for team in scoreLogs['Visiting team'].unique():
        vistTeams[team] = scoreLogs[scoreLogs['Visiting team']==team]
    homeTVers = {}
    for hTeam in homeTeams:
        homeTeams[hTeam]['Home win ratio']   = homeTeams[hTeam].loc[:,'Home team win'].rolling(10).mean().shift(1)
        homeTeams[hTeam]['Home score ratio'] = homeTeams[hTeam].loc[:,'Home score'].rolling(10).mean().shift(1)
        homeTeams[hTeam]['Home odd ratio']   = homeTeams[hTeam].loc[:,'Home team odd'].rolling(10).mean().shift(1)
        temp = homeTeams[hTeam]
        versus = {}
        for team in temp['Visiting team'].unique():
            versus[team] = temp[temp['Visiting team']==team]
        for vTeam in versus:
            versus[vTeam]['Home versus win ratio']   = versus[vTeam].loc[:,'Home team win'].rolling(5).mean().shift(1)
            versus[vTeam]['Home versus score ratio'] = versus[vTeam].loc[:,'Home score'].rolling(5).mean().shift(1)
            versus[vTeam]['Home versus odd ratio']   = versus[vTeam].loc[:,'Home team odd'].rolling(5).mean().shift(1)
        homeTVers[hTeam] = pd.concat(versus)
    vistTVers = {}
    for vTeam in vistTeams:
        vistTeams[vTeam]['Visiting win ratio']   = (1-vistTeams[vTeam].loc[:,'Home team win']).rolling(10).mean().shift(1)
        vistTeams[vTeam]['Visiting score ratio'] = vistTeams[vTeam].loc[:,'Visiting score'].rolling(10).mean().shift(1)
        vistTeams[vTeam]['Visiting odd ratio']   = (1/vistTeams[vTeam].loc[:,'Home team odd']).rolling(10).mean().shift(1)
        temp = vistTeams[vTeam]
        versus = {}
        for team in temp['Home team'].unique():
            versus[team] = temp[temp['Home team']==team]
        for hTeam in versus:
            versus[hTeam]['Visiting versus win ratio']   = (1-versus[hTeam].loc[:,'Home team win']).rolling(5).mean().shift(1)
            versus[hTeam]['Visiting versus score ratio'] = versus[hTeam].loc[:,'Visiting score'].rolling(5).mean().shift(1)
            versus[hTeam]['Visiting versus odd ratio']   = (1/versus[hTeam].loc[:,'Home team odd']).rolling(5).mean().shift(1)
        vistTVers[vTeam] = pd.concat(versus)
    merged = pd.merge(pd.concat(vistTeams)[['row'
                                       ,'Visiting win ratio'
                                       ,'Visiting score ratio'
                                       ,'Visiting odd ratio']]
                 ,pd.concat(homeTVers)[['row'
                                       ,'Home versus win ratio'
                                       ,'Home versus score ratio'
                                       ,'Home versus odd ratio']]
                 , on='row')
    merged = pd.merge(pd.concat(vistTVers)[['row'
                                       ,'Visiting versus win ratio'
                                       ,'Visiting versus score ratio'
                                       ,'Visiting versus odd ratio']]
                 ,merged, on='row')
    merged = pd.merge(pd.concat(homeTeams)[['row'
                                       ,'Home win ratio'
                                       ,'Home score ratio'
                                       ,'Home odd ratio']]
                 ,merged, on='row')
    merged = pd.merge(scoreLogs[['row','Visiting score','Home score','Home team win','Home team odd']],merged, on='row').fillna(0)
    print("Created Scorings. Checksum: ", gameLogs.index.size==merged.index.size, gameLogs.index.size, merged.index.size)
    return merged

path = r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Projekt'
gameLogs    = pd.read_csv(path+r'\Replaced\_mlb_encoded_GameLogs.csv', index_col=False)
people      = pd.read_csv(path+r'\Replaced\_mlb_encoded_People.csv', index_col=False)
teams       = pd.read_csv(path+r'\Replaced\_mlb_replaced_Teams.csv', index_col=False)
managers    = pd.read_csv(path+r'\Replaced\_mlb_replaced_Managers.csv', index_col=False)
pitchings   = pd.read_csv(path+r'\Replaced\_mlb_replaced_Pitching.csv', index_col=False)
battings    = pd.read_csv(path+r'\Replaced\_mlb_replaced_Batting.csv', index_col=False)
fieldings   = pd.read_csv(path+r'\Replaced\_mlb_replaced_Fielding.csv', index_col=False)

#scorings    = createScorings(gameLogs)
people      = mergePeople(people, gameLogs)
#teams       = mergeTeams(teams, gameLogs)
#managers    = mergeManagers(managers, gameLogs)
#pitchings   = mergePitchings(pitchings, gameLogs)
#battings    = mergeBattings(battings, gameLogs)
#fieldings   = mergeFieldings(fieldings, gameLogs)
#mlbData     = [scorings, people, teams, managers, pitchings, battings, fieldings]

#scorings.to_csv(path+r'\Remerged\_mlb_recreated_Scoring.csv', index = False)
people.to_csv(path+r'\Remerged\_mlb_remerged_People.csv', index = False)
#teams.to_csv(path+r'\Remerged\_mlb_remerged_Teams.csv', index = False)
#managers.to_csv(path+r'\Remerged\_mlb_remerged_Managers.csv', index = False)
#pitchings.to_csv(path+r'\Remerged\_mlb_remerged_Pitchers.csv', index = False)
#battings.to_csv(path+r'\Remerged\_mlb_remerged_Batting.csv', index = False)
#fieldings.to_csv(path+r'\Remerged\_mlb_remerged_Fielding.csv', index = False)