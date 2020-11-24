import pandas as pd
from pathlib import Path
import numpy as np

def getTeams(teamColumn ,gameLogs):
    teams = {}
    for team in gameLogs[teamColumn].unique():
        teams[team] = gameLogs[gameLogs[teamColumn]==team]
    return teams

def getWinRatio(teamType, team, window=10):
    if teamType=="Home":
        return team.loc[:,'Home team win'].rolling(window).mean().shift(1)
    else:
        return (1-team.loc[:,'Home team win']).rolling(window).mean().shift(1)

def getScoreRatio(teamType, team, window=10):
    if teamType=="Home":
        return team.loc[:,'Home score'].rolling(window).mean().shift(1)
    else:
        return team.loc[:,'Visiting score'].rolling(window).mean().shift(1)

def getOddRatio(teamType, team, window=10):
    if teamType=="Home":
        return team.loc[:,'Home team odd'].rolling(window).mean().shift(1)
    else:
        return (1/team.loc[:,'Home team odd']).rolling(window).mean().shift(1)

path = Path
gameLogs = pd.read_csv(path+r'\Filtered\_mlb_filtered_GameLogs.csv', index_col=False)

gameLogs = gameLogs[['row','Visiting team','Home team','Visiting score','Home score']]
gameLogs['Home team win'] = gameLogs['Home score']>gameLogs['Visiting score']
gameLogs['Home team odd'] = (gameLogs['Home score'].replace(0,1))/(gameLogs['Visiting score'].replace(0,1))

homeTeams = getTeams('Home team', gameLogs)
vistTeams = getTeams('Visiting team', gameLogs)
homeTVers = {}
for hTeam in homeTeams:
    homeTeams[hTeam]['Home win ratio']   = getWinRatio("Home", homeTeams[hTeam])
    homeTeams[hTeam]['Home score ratio'] = getScoreRatio("Home", homeTeams[hTeam])
    homeTeams[hTeam]['Home odd ratio']   = getOddRatio("Home", homeTeams[hTeam])
    versus = getTeams('Visiting team', homeTeams[hTeam])
    for vTeam in versus:
        versus[vTeam]['Home versus win ratio']   = getWinRatio("Home", versus[vTeam], 5)
        versus[vTeam]['Home versus score ratio'] = getScoreRatio("Home", versus[vTeam], 5)
        versus[vTeam]['Home versus odd ratio']   = getOddRatio("Home", versus[vTeam], 5)
    homeTVers[hTeam] = pd.concat(versus)
vistTVers = {}
for vTeam in vistTeams:
    vistTeams[vTeam]['Visiting win ratio']   = getWinRatio("Visiting", vistTeams[vTeam])
    vistTeams[vTeam]['Visiting score ratio'] = getScoreRatio("Visiting", vistTeams[vTeam])
    vistTeams[vTeam]['Visiting odd ratio']   = getOddRatio("Visiting", vistTeams[vTeam])
    versus = getTeams('Home team', vistTeams[vTeam])
    for hTeam in versus:
        versus[hTeam]['Visiting versus win ratio']   = getWinRatio("Visiting", versus[hTeam], 5)
        versus[hTeam]['Visiting versus score ratio'] = getScoreRatio("Visiting", versus[hTeam], 5)
        versus[hTeam]['Visiting versus odd ratio']   = getOddRatio("Visiting", versus[hTeam], 5)
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
merged = pd.merge(gameLogs[['row'
                           ,'Visiting score'
                           ,'Home score'
                           ,'Home team win'
                           ,'Home team odd']]
                 ,merged, on='row')
merged.sort_values(by=['row']).to_csv(path+r'\Merged\_mlb_created_Scoring.csv', index = False)