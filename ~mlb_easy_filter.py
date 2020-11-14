import pandas as pd
import numpy as np
from datetime import datetime as dt

def filterGameLogs(gameLogs, dropna=True):
    #Parsing dates
    gameLogs['Date'] = pd.to_datetime(gameLogs['Date'], format="%Y%m%d")
    #Filtering invalid games
    gameLogs = gameLogs[gameLogs['Forfeit information'].isna()]
    gameLogs = gameLogs[gameLogs['Protest information'].isna()]
    #Dropping columns
    generalColumns = [
        'Date','Visiting team','Visiting league','Home team','Home league','Visiting score','Home score']
    visitingStatsColumns = [
        'Visiting at-bats','Visiting hits','Visiting doubles','Visiting triples','Visiting homeruns','Visiting RBI','Visiting sacrifice hits','Visiting sacrifice flies',
        'Visiting hit-by-pitch','Visiting walks','Visiting intentional walks','Visiting strikeouts','Visiting stolen bases','Visiting caught stealing','Visiting grounded into double plays',
        'Visiting left on base','Visiting pitchers used','Visiting individual earned runs','Visiting team earned runs','Visiting wild pitches',
        'Visiting balks','Visiting putouts','Visiting assists','Visiting errors','Visiting passed balls','Visiting double plays','Visiting triple plays']
    homeStatsColumns = [
        'Home at-bats','Home hits','Home doubles','Home triples','Home homeruns','Home RBI','Home sacrifice hits','Home sacrifice flies',
        'Home hit-by-pitch','Home walks','Home intentional walks','Home strikeouts','Home stolen bases','Home caught stealing','Home grounded into double plays',
        'Home left on base','Home pitchers used','Home individual earned runs','Home team earned runs','Home wild pitches',
        'Home balks','Home putouts','Home assists','Home errors','Home passed balls','Home double plays','Home triple plays']
    visitingIDColumns = [
        'Visiting team manager ID','Visiting starting pitcher ID',
        'Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID',
        'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID',
        'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID']
    homeIDColumns = [
        'Home team manager ID','Home starting pitcher ID',
        'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID',
        'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID',
        'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID']
    #Filtering invalid informations
    gameLogs = gameLogs[generalColumns+visitingStatsColumns+homeStatsColumns+visitingIDColumns+homeIDColumns]
    gameLogs = gameLogs.dropna(subset=generalColumns)
    for column in visitingStatsColumns+homeStatsColumns:
        gameLogs = gameLogs[(gameLogs[column]>=0) | (gameLogs[column].isna())]
    #Returning gameLogs
    return gameLogs.reset_index(drop=True)
    
def filterManagers(managers):
    #Dropping columns
    columns = ['yearID','playerID','G','W','L']
    managers = managers[columns]
    #Renaming columns
    managers = managers.rename(columns={"G":"Games","W":"Wins","L":"Losses"})
    #Returning managers
    return managers.reset_index(drop=True)

def matchPeopleIDs(people, gameLogs):
    #Generating unique identifier
    identifier = people[['playerID','retroID']].drop_duplicates(subset=['retroID']).dropna()
    #Left-outer join per ID-column
    for idC in gameLogs.columns:
        if idC.find(" ID")>-1:
            merged        = pd.merge(gameLogs[idC], identifier, left_on=idC, right_on='retroID', how="left")
            gameLogs[idC] = merged['playerID']
    #Returning gameLogs
    return gameLogs.reset_index(drop=True)

def filterPeople(people):
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
    return people.reset_index(drop=True)

def addCollegePlaying(people, collegePlaying):
    collegePlaying = collegePlaying['playerID'].drop_duplicates()
    people['Played in college'] = people['playerID'].isin(collegePlaying)
    return people

def filterFieldings(fieldings):
    #Dropping columns
    columns = ['yearID','playerID','PO','A','E','DP','PB','WP','SB','CS','ZR']
    fieldings = fieldings[columns]
    #Renaming columns
    fieldings = fieldings.rename(columns={"PO":"Putouts","A":"Assists","E":"Error","DP":"Double plays","PB":"Passed Balls","WP":"Wild Pitches","SB":"Opponent Stolen Bases","CS":"Opponents Caught Stealing","ZR":"Zone Rating"})
    #Returning fieldings
    return fieldings.reset_index(drop=True)

def filterPitchings(pitchings):
    #Dropping columns
    columns = ['yearID','playerID','H','ER','HR','BB','SO','BAOpp','ERA','IBB','WP','HBP','BK','BFP','R','SH','SF','GIDP']
    pitchings = pitchings[columns]
    #Renaming columns
    pitchings = pitchings.rename(columns={"H":"Hits","ER":"Earned Runs","HR":"Homeruns","BB":"Walks","SO":"Strikeouts","BAOpp":"Opponent batting average","ERA":"Earned run average"
                                         ,"IBB":"Intentional walks","WP":"Wild pitches","HBP":"Batters hit by pitch","BK":"Balks","BFP":"Batters faced","R":"Runs allowed","SH":"Batters sacrifices"
                                         ,"SF":"Batters sacrifice flies","GIDP":"Grounded into double plays"})
    #Returning pitchings
    return pitchings.reset_index(drop=True)

def filterBattings(battings):
    #Dropping columns
    columns = ['yearID','playerID','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP']
    battings = battings[columns]
    #Renaming columns
    battings = battings.rename(columns={"AB":"At bats","R":"Runs","H":"Hits","2B":"Doubles","3B":"Triples","HR":"Homeruns","RBI":"Runs batted in","SB":"Stolen bases","CS":"Caught stealing"
                                       ,"BB":"Base on balls","SO":"Strikeouts","IBB":"Intentional walks","HBP":"Hit by pitch","SH":"Sacrifice hits","SF":"Sacrifice flies","GIDP":"Grounded into double plays"})
    #Returning battings
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
collegePlaying = pd.read_csv(path+r'\Input\CollegePlaying.csv', index_col=False)
#Filter gameLogs
gameLogs    = filterGameLogs(gameLogs)
#Get matching IDs
gameLogs    = matchPeopleIDs(people, gameLogs)
#Filter data
people      = addCollegePlaying(filterPeople(people), collegePlaying)
teams       = filterTeams(teams)
managers    = filterManagers(managers)
fieldings   = filterFieldings(fieldings)
pitchings   = filterPitchings(pitchings)
battings    = filterBattings(battings)
#Saving data
gameLogs.to_csv(path+r'\Filtered\_mlb_filtered_GameLogs.csv', index = False)
people.to_csv(path+r'\Filtered\_mlb_filtered_People.csv', index = False)
teams.to_csv(path+r'\Filtered\_mlb_filtered_Teams.csv', index = False)
managers.to_csv(path+r'\Filtered\_mlb_filtered_Managers.csv', index = False)
fieldings.to_csv(path+r'\Filtered\_mlb_filtered_Fielding.csv', index = False)
pitchings.to_csv(path+r'\Filtered\_mlb_filtered_Pitching.csv', index = False)
battings.to_csv(path+r'\Filtered\_mlb_filtered_Batting.csv', index = False)