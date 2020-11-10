import pandas as pd
import numpy as np
from datetime import datetime as dt

def filterGameLogs(gameLogs, dropna=True):
    #Parsing dates
    gameLogs['Date'] = pd.to_datetime(gameLogs['Date'], format="%Y%m%d")
    #Filtering invalid or non-matching games
    gameLogs = gameLogs[gameLogs['Completion information'].isnull()]
    gameLogs = gameLogs[gameLogs['Forfeit information'].isnull()]
    gameLogs = gameLogs[gameLogs['Protest information'].isnull()]
    gameLogs = gameLogs[gameLogs['Additional information'].isnull()]
    gameLogs = gameLogs[(gameLogs['Visiting league']=="NL") | (gameLogs['Visiting league']=="AL")]
    gameLogs = gameLogs[(gameLogs['Home league']=="NL") | (gameLogs['Home league']=="AL")]
    #gameLogs = gameLogs[gameLogs['Acquisition information']!="P"]
    #Dropping columns
    columns =   ['Date','Visiting team','Visiting league','Home team','Home league'
                ,'Visiting score','Home score','Visiting team manager ID','Home team manager ID'
                ,'Visiting starting pitcher ID','Home starting pitcher ID'
                ,'Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID'
                ,'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID'
                ,'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID'
                ,'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID'
                ,'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID'
                ,'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID']
    gameLogs = gameLogs[columns]
    #Returning gameLogs
    if dropna==True: return gameLogs.dropna().reset_index(drop=True)
    return gameLogs.reset_index(drop=True)
    
def filterManagers(managers, dropna=True):
    #Dropping columns
    columns = ['yearID','playerID','G','W','L']
    managers = managers[columns]
    #Renaming columns
    managers = managers.rename(columns={"G":"Games","W":"Wins","L":"Losses"})
    #Returning managers
    if dropna==True: return managers.dropna().reset_index(drop=True)
    return managers.reset_index(drop=True)

def matchPeopleIDs(people, gameLogs, dropna=True):
    #Generating unique identifier
    identifier = people[['playerID','retroID']].drop_duplicates(subset=['retroID']).dropna()
    #Left-outer join per ID-column
    for idC in gameLogs.columns:
        if idC.find(" ID")>-1:
            merged        = pd.merge(gameLogs[idC], identifier, left_on=idC, right_on='retroID', how="left")
            gameLogs[idC] = merged['playerID']
    #Returning gameLogs
    if dropna==True: return gameLogs.dropna().reset_index(drop=True)
    return gameLogs.reset_index(drop=True)

def filterPeople(people, dropna=True):
    #Generating birthdate
    people = people.rename(columns={"birthYear":"year","birthMonth":"month","birthDay":"day"})
    people['birthdate'] = pd.to_datetime(people[['year','month','day']])
    #Transforming non metric values
    people['weight'] = 0.453592*people['weight']
    people['height'] = 0.0254*people['height']
    #Dropping columns
    columns = ['playerID','birthdate','weight','height','bats','throws','finalGame']
    people = people[columns].drop(columns=['finalGame'])
    #Returning people
    if dropna==True: return people.dropna().reset_index(drop=True)
    return people.reset_index(drop=True)

def filterFieldings(fieldings, dropna=True):
    #Dropping columns
    columns = ['yearID','playerID','PO','A','E','DP']
    fieldings = fieldings[columns]
    #Renaming columns
    fieldings = fieldings.rename(columns={"PO":"Putouts","A":"Assists","E":"Error","DP":"Double plays"})
    #Returning fieldings
    if dropna==True: return fieldings.dropna().reset_index(drop=True)
    return fieldings.reset_index(drop=True)

def filterPitchings(pitchings, dropna=True):
    #Dropping columns
    columns = ['yearID','playerID','H','ER','HR','BB','SO','BAOpp','ERA','IBB','WP','HBP','BK','BFP','R','SH','SF','GIDP']
    pitchings = pitchings[columns]
    #Renaming columns
    pitchings = pitchings.rename(columns={"H":"Hits","ER":"Earned Runs","HR":"Homeruns","BB":"Walks","SO":"Strikeouts","BAOpp":"Opponent batting average","ERA":"Earned run average"
                                         ,"IBB":"Intentional walks","WP":"Wild pitches","HBP":"Batters hit by pitch","BK":"Balks","BFP":"Batters faced","R":"Runs allowed","SH":"Batters sacrifices"
                                         ,"SF":"Batters sacrifice flies","GIDP":"Grounded into double plays"})
    #Returning pitchings
    if dropna==True: return pitchings.dropna().reset_index(drop=True)
    return pitchings.reset_index(drop=True)

def filterBattings(battings, dropna=True):
    #Dropping columns
    columns = ['yearID','playerID','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP']
    battings = battings[columns]
    #Renaming columns
    battings = battings.rename(columns={"AB":"At bats","R":"Runs","H":"Hits","2B":"Doubles","3B":"Triples","HR":"Homeruns","RBI":"Runs batted in","SB":"Stolen bases","CS":"Caught stealing"
                                       ,"BB":"Base on balls","SO":"Strikeouts","IBB":"Intentional walks","HBP":"Hit by pitch","SH":"Sacrifice hits","SF":"Sacrifice flies","GIDP":"Grounded into double plays"})
    #Returning battings
    if dropna==True: return battings.dropna().reset_index(drop=True)
    return battings.reset_index(drop=True)

def filterTeams(teams, dropna=True):
    #Dropping columns
    columns = ['yearID','teamIDretro','divID','Rank','G','W','L','DivWin','LgWin','WSWin','R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','SHO','SV'
              ,'HA','HRA','BBA','SOA','E','DP','FP']
    teams = teams[columns]
    #Renaming columns
    teams = teams.rename(columns={"teamIDretro":"teamID","divID":"Division","G":"Games","W":"Wins","L":"Losses","DivWin":"Division winner","LgWin":"League winner","WSWin":"World series winner","R":"Runs scored","AB":"At bats"
                                 ,"H":"Hits by batters","2B":"Doubles","3B":"Triples","HR":"Homeruns","BB":"Walks","SO":"Strikeouts","SB":"Stolen bases","CS":"Cought stealing","HBP":"Batters hit by pitch"
                                 ,"SF":"Sacrifice flies","RA":"Opponents runs scored","ER":"Earned runs allowed","ERA":"Earned runs average","SHO":"Shutouts","SV":"Saves","HA":"Hits allowed"
                                 ,"HRA":"Homeruns allowed","BBA":"Walks allowed","SOA":"Strikeouts allowed","E":"Errors","DP":"Double plays","FP":"Fielding percentage"})
    #Returning teams
    if dropna==True: return teams.dropna().reset_index(drop=True)
    return teams.reset_index(drop=True)
#Loading data
path = r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Projekt'
gameLogs    = pd.read_csv(path+r'\Input\GameLogs.csv', index_col=False)
people      = pd.read_csv(path+r'\Input\People.csv', index_col=False)
teams       = pd.read_csv(path+r'\Input\Teams.csv', index_col=False)
managers    = pd.read_csv(path+r'\Input\Managers.csv', index_col=False)
fieldings   = pd.read_csv(path+r'\Input\Fielding.csv', index_col=False)
pitchings   = pd.read_csv(path+r'\Input\Pitching.csv', index_col=False)
battings    = pd.read_csv(path+r'\Input\Batting.csv', index_col=False)
#Filter gameLogs
gameLogs    = filterGameLogs(gameLogs, False)
#Get matching IDs
gameLogs    = matchPeopleIDs(people, gameLogs, False)
#Filter data
people      = filterPeople(people)
teams       = filterTeams(teams, False)
managers    = filterManagers(managers, False)
fieldings   = filterFieldings(fieldings, False)
pitchings   = filterPitchings(pitchings, False)
battings    = filterBattings(battings, False)
#Creating row IDs
gameLogs['row'] = range(0,gameLogs.index.size)
#Rearange row IDs
gameLogs = gameLogs[(gameLogs.columns[-1:].tolist()+gameLogs.columns[:-1].tolist())]
print(gameLogs)
#Saving data
path = r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Projekt'
gameLogs.to_csv(path+r'\Filtered\_mlb_filtered_GameLogs.csv', index = False)
people.to_csv(path+r'\Filtered\_mlb_filtered_People.csv', index = False)
teams.to_csv(path+r'\Filtered\_mlb_filtered_Teams.csv', index = False)
managers.to_csv(path+r'\Filtered\_mlb_filtered_Managers.csv', index = False)
fieldings.to_csv(path+r'\Filtered\_mlb_filtered_Fielding.csv', index = False)
pitchings.to_csv(path+r'\Filtered\_mlb_filtered_Pitching.csv', index = False)
battings.to_csv(path+r'\Filtered\_mlb_filtered_Batting.csv', index = False)