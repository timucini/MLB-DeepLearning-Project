import pandas as pd
import numpy as np
from datetime import datetime as dt

def mergeManagers(managers, gameLogs):
    #Sum up doubled data
    managers = managers.groupby(['yearID','playerID'], as_index=False)['Games','Wins','Losses'].sum()
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
    return pd.merge(homes, visitings, on='row', suffixes=(' home manager',' visiting manager'))

def mergePitchings(pitchers, gameLogs):
    #Get aggregators for doubled data
    aggregators = {}
    for column in pitchers.drop(columns=['yearID','playerID']).columns:
        if column.find("average")>-1:
            aggregators[column] = 'mean'
        else:
            aggregators[column] = 'sum'
    #Aggregate doubled data
    pitchers = pitchers.groupby(['yearID','playerID'], as_index=False).agg(aggregators)
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
    return pd.merge(homes, visitings, on='row', suffixes=(' home pitcher',' visiting pitcher'))

def mergePeople(people, gameLogs):
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
    mergedPeople = gameLogs['row']
    for merSal in allPeople:
        mergedPeople = pd.merge(mergedPeople, merSal, how="left", on='row')
    return mergedPeople

def mergeTeams(teams, gameLogs):
    #Encode team data
    teams.loc[(teams['Division winner'] == 'N'), 'Division winner'] = 0
    teams.loc[(teams['Division winner'] == 'Y'), 'Division winner'] = 1
    teams.loc[(teams['League winner'] == 'N'), 'League winner'] = 0
    teams.loc[(teams['League winner'] == 'Y'), 'League winner'] = 1
    teams.loc[(teams['World series winner'] == 'N'), 'World series winner'] = 0
    teams.loc[(teams['World series winner'] == 'Y'), 'World series winner'] = 1
    teams.loc[(teams['Division'] == 'W'), 'Division'] = 0
    teams.loc[(teams['Division'] == 'E'), 'Division'] = 1
    teams.loc[(teams['Division'] == 'C'), 'Division'] = 2
    teams['Pythagorean_expectation'] = (teams['Runs scored'] ** 1.83) / (teams['Runs scored'] ** 1.83 + teams['Opponents runs scored'] ** 1.83)
    #Merge teams
    mergedTeams = gameLogs[['Date','Visiting team','Home team']]
    mergedTeams['Date'] = pd.to_datetime(mergedTeams['Date']).dt.year-1
    mergedTeams = pd.merge(mergedTeams, teams, left_on=['Date', 'Visiting team'], right_on=['yearID', 'teamID'], how='left')
    mergedTeams = pd.merge(mergedTeams, teams, left_on=['Date', 'Home team'], right_on=['yearID', 'teamID'], how='left', suffixes=[' visiting', ' home'])
    return mergedTeams[['row', 'Division visiting', 'Rank visiting', 'Games visiting', 'Wins visiting', 'Losses visiting', 'Division winner visiting',
               'League winner visiting', 'World series winner visiting', 'Runs scored visiting', 'At bats visiting',
               'Hits by batters visiting', 'Doubles visiting', 'Triples visiting', 'Homeruns visiting', 'Walks visiting', 'Strikeouts visiting',
               'Stolen bases visiting', 'Cought stealing visiting', 'Batters hit by pitch visiting', 'Sacrifice flies visiting',
               'Opponents runs scored visiting', 'Earned runs allowed visiting', 'Earned runs average visiting', 'Shutouts visiting',
               'Saves visiting', 'Hits allowed visiting', 'Homeruns allowed visiting', 'Walks allowed visiting',
               'Strikeouts allowed visiting', 'Errors visiting', 'Double plays visiting', 'Fielding percentage visiting',
               'Pythagorean_expectation visiting', 'Division home', 'Rank home', 'Games home', 'Wins home', 'Losses home',
               'Division winner home', 'League winner home', 'World series winner home', 'Runs scored home',
               'At bats home', 'Hits by batters home', 'Doubles home', 'Triples home', 'Homeruns home',
               'Walks home', 'Strikeouts home', 'Stolen bases home', 'Cought stealing home',
               'Batters hit by pitch home', 'Sacrifice flies home', 'Opponents runs scored home',
               'Earned runs allowed home', 'Earned runs average home', 'Shutouts home', 'Saves home',
               'Hits allowed home', 'Homeruns allowed home', 'Walks allowed home', 'Strikeouts allowed home',
               'Errors home', 'Double plays home', 'Fielding percentage home', 'Pythagorean_expectation home']]

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
        homeTeams[hTeam]['Home win ratio']   = team.loc[:,'Home team win'].rolling(10).mean().shift(1)
        homeTeams[hTeam]['Home score ratio'] = team.loc[:,'Home score'].rolling(10).mean().shift(1)
        homeTeams[hTeam]['Home odd ratio']   = team.loc[:,'Home team odd'].rolling(10).mean().shift(1)
        for vTeam in versus:
            versus[vTeam]['Home versus win ratio']   = team.loc[:,'Home team win'].rolling(5).mean().shift(1)
            versus[vTeam]['Home versus score ratio'] = team.loc[:,'Home score'].rolling(5).mean().shift(1)
            versus[vTeam]['Home versus odd ratio']   = team.loc[:,'Home team odd'].rolling(5).mean().shift(1)
        homeTVers[hTeam] = pd.concat(versus)
    vistTVers = {}
    for vTeam in vistTeams:
        vistTeams[vTeam]['Visiting win ratio']   = (1-team.loc[:,'Home team win']).rolling(10).mean().shift(1)
        vistTeams[vTeam]['Visiting score ratio'] = team.loc[:,'Visiting score'].rolling(10).mean().shift(1)
        vistTeams[vTeam]['Visiting odd ratio']   = (1/team.loc[:,'Home team odd']).rolling(10).mean().shift(1)
        versus = getTeams('Home team', vistTeams[vTeam])
        for hTeam in versus:
            versus[hTeam]['Visiting versus win ratio']   = (1-team.loc[:,'Home team win']).rolling(5).mean().shift(1)
            versus[hTeam]['Visiting versus score ratio'] = team.loc[:,'Visiting score'].rolling(5).mean().shift(1)
            versus[hTeam]['Visiting versus odd ratio']   = (1/team.loc[:,'Home team odd']).rolling(5).mean().shift(1)
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
    return pd.merge(scoreLogs[['row','Visiting score','Home score','Home team win','Home team odd']],merged, on='row')