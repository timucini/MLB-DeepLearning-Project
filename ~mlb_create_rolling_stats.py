import pandas as pd
import numpy as np
from datetime import datetime

def getTeamPart(teamColumn ,fromFrame):
    teams = []
    for team in fromFrame[teamColumn].unique():
        teams.append(fromFrame[fromFrame[teamColumn]==team])
    return teams

def getRollingMean(target ,team, window=3):
    return team.loc[:,target].rolling(window).mean().shift(1)

def getValues(targetTeamType, teams, columns):
    teamColumns = []
    versusColumns = []
    versus = []
    for team in teams:
        for column in columns:
            newColumn = column + ' ratio'
            teamColumns.append(newColumn)
            team[newColumn] = getRollingMean(column, team)
        tempvs = getTeamPart(targetTeamType, team)
        for vsteam in tempvs:
            for column in columns:
                newColumn = column + ' versus ratio'
                versusColumns.append(newColumn)
                vsteam[newColumn] = getRollingMean(column, vsteam)
        versus.append(pd.concat(tempvs))
    return pd.merge(pd.concat(teams)[['row']+list(dict.fromkeys(teamColumns))], pd.concat(versus)[['row']+list(dict.fromkeys(versusColumns))], on='row')

path = r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Projekt'
gameLogs = pd.read_csv(path+r'\Replaced\_mlb_encoded_GameLogs.csv', index_col=False)

gameLogs['Home team win'] = gameLogs['Home score']>gameLogs['Visiting score']
gameLogs['Home team odd'] = (gameLogs['Home score'].replace(0,1))/(gameLogs['Visiting score'].replace(0,1))
gameLogs['Visiting team win'] = gameLogs['Visiting score']>gameLogs['Home score']
gameLogs['Visiting team odd'] = (gameLogs['Visiting score'].replace(0,1))/(gameLogs['Home score'].replace(0,1))

generalColumns = ['row','Date','Visiting team','Home team','Visiting score','Home score']
homeColumns = ['Home team win', 'Home team odd',
    'Home at-bats','Home hits','Home doubles','Home triples','Home homeruns','Home RBI','Home sacrifice hits','Home sacrifice flies',
    'Home hit-by-pitch','Home walks','Home intentional walks','Home strikeouts','Home stolen bases','Home caught stealing','Home grounded into double plays',
    'Home left on base','Home pitchers used','Home individual earned runs','Home team earned runs','Home wild pitches',
    'Home balks','Home putouts','Home assists','Home errors','Home passed balls','Home double plays','Home triple plays']
visitingColumns = ['Visiting team win', 'Visiting team odd',
    'Visiting at-bats','Visiting hits','Visiting doubles','Visiting triples','Visiting homeruns','Visiting RBI','Visiting sacrifice hits','Visiting sacrifice flies',
    'Visiting hit-by-pitch','Visiting walks','Visiting intentional walks','Visiting strikeouts','Visiting stolen bases','Visiting caught stealing','Visiting grounded into double plays',
    'Visiting left on base','Visiting pitchers used','Visiting individual earned runs','Visiting team earned runs','Visiting wild pitches',
    'Visiting balks','Visiting putouts','Visiting assists','Visiting errors','Visiting passed balls','Visiting double plays','Visiting triple plays']

homes = getValues('Visiting team', getTeamPart('Home team', gameLogs[generalColumns+homeColumns]), homeColumns)
visitings = getValues('Home team', getTeamPart('Visiting team', gameLogs[generalColumns+visitingColumns]), visitingColumns)
teams = pd.merge(homes, visitings, on='row')
gameLogs = pd.merge(gameLogs[generalColumns+['Home team win']], teams, on='row').fillna(0)
gameLogs.to_csv(path+r'\Remerged\_mlb_game_stats.csv', index = False)