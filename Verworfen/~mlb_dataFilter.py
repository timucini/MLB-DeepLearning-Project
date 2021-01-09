import numpy as np
import pandas as pd
import datetime
import glob
import os

def mlb_gameLogsLoader(path=r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Daten\Baseball Data up to 2020\GameLogs'):
    col_names = ["Date","Number of game","Day of week","Visiting team","Visiting league","Visiting team game number","Home team"
                ,"Home league","Home team game number","Visiting score","Home score","Length of game in outs","Day/night indicator"
                ,"Completion information","Forfeit information","Protest information","Park ID","Attendance"
                ,"Time of game in minutes","Visiting line scores","Home line scores","Visiting at-bats","Visiting hits"
                ,"Visiting doubles","Visiting triples","Visiting homeruns","Visiting RBI","Visiting sacrifice hits"
                ,"Visiting sacrifice flies","Visiting hit-by-pitch","Visiting walks","Visiting intentional walks"
                ,"Visiting strikeouts","Visiting stolen bases","Visiting caught stealing","Visiting grounded into double plays"
                ,"Visiting awarded first on catcher's interference","Visiting left on base","Visiting pitchers used"
                ,"Visiting individual earned runs","Visiting team earned runs","Visiting wild pitches","Visiting balks"
                ,"Visiting putouts","Visiting assists","Visiting errors","Visiting passed balls","Visiting double plays"
                ,"Visiting triple plays","Home at-bats","Home hits","Home doubles","Home triples","Home homeruns"
                ,"Home RBI","Home sacrifice hits","Home sacrifice flies","Home hit-by-pitch","Home walks","Home intentional walks"
                ,"Home strikeouts","Home stolen bases","Home caught stealing","Home grounded into double plays"
                ,"Home awarded first on catcher's interference","Home left on base","Home pitchers used"
                ,"Home individual earned runs","Home team earned runs","Home wild pitches","Home balks","Home putouts"
                ,"Home assists","Home errors","Home passed balls","Home double plays","Home triple plays","Home plate umpire ID"
                ,"Home plate umpire name","1B umpire ID","1B umpire name","2B umpire ID","2B umpire name","3B umpire ID"
                ,"3B umpire name","LF umpire ID","LF umpire name","RF umpire ID","RF umpire name","Visiting team manager ID"
                ,"Visiting team manager name","Home team manager ID","Home team manager name","Winning pitcher ID"
                ,"Winning pitcher name","Losing pitcher ID","Losing pitcher name","Saving pitcher ID","Saving pitcher name"
                ,"Game Winning RBI batter ID","Game Winning RBI batter name","Visiting starting pitcher ID"
                ,"Visiting starting pitcher name","Home starting pitcher ID","Home starting pitcher name"
                ,"Visiting starting player 1 ID","Visiting starting player 1 name","Visiting starting player 1 defensive position"
                ,"Visiting starting player 2 ID","Visiting starting player 2 name","Visiting starting player 2 defensive position"
                ,"Visiting starting player 3 ID","Visiting starting player 3 name","Visiting starting player 3 defensive position"
                ,"Visiting starting player 4 ID","Visiting starting player 4 name","Visiting starting player 4 defensive position"
                ,"Visiting starting player 5 ID","Visiting starting player 5 name","Visiting starting player 5 defensive position"
                ,"Visiting starting player 6 ID","Visiting starting player 6 name","Visiting starting player 6 defensive position"
                ,"Visiting starting player 7 ID","Visiting starting player 7 name","Visiting starting player 7 defensive position"
                ,"Visiting starting player 8 ID","Visiting starting player 8 name","Visiting starting player 8 defensive position"
                ,"Visiting starting player 9 ID","Visiting starting player 9 name","Visiting starting player 9 defensive position"
                ,"Home starting player 1 ID","Home starting player 1 name","Home starting player 1 defensive position"
                ,"Home starting player 2 ID","Home starting player 2 name","Home starting player 2 defensive position"
                ,"Home starting player 3 ID","Home starting player 3 name","Home starting player 3 defensive position"
                ,"Home starting player 4 ID","Home starting player 4 name","Home starting player 4 defensive position"
                ,"Home starting player 5 ID","Home starting player 5 name","Home starting player 5 defensive position"
                ,"Home starting player 6 ID","Home starting player 6 name","Home starting player 6 defensive position"
                ,"Home starting player 7 ID","Home starting player 7 name","Home starting player 7 defensive position"
                ,"Home starting player 8 ID","Home starting player 8 name","Home starting player 8 defensive position"
                ,"Home starting player 9 ID","Home starting player 9 name","Home starting player 9 defensive position"
                ,"Additional information","Acquisition information"]
    all_files = glob.glob(os.path.join(path, "*.TXT"))
    df_from_each_file = (pd.read_csv(f,index_col=False, header=None, names=col_names) for f in all_files)
    return pd.concat(df_from_each_file, ignore_index=True)

def mlb_gameLogsLoaderCSV(path=r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Daten\Baseball Data up to 2020\GameLogs\GameLogs since 1871 to 2019.csv'):
    return(pd.read_csv(path, index_col=False))

def mlb_dataFilter(gameLogs):
    gameLogs['Date'] = pd.to_datetime(gameLogs['Date'], format="%Y%m%d")
    gameLogs = gameLogs[gameLogs['Completion information'].isnull()]
    gameLogs = gameLogs[gameLogs['Forfeit information'].isnull()]
    gameLogs = gameLogs[gameLogs['Protest information'].isnull()]
    gameLogs = gameLogs[gameLogs['Visiting league']!="FL"]
    gameLogs = gameLogs[gameLogs['Home league']!="FL"]
    gameLogs = gameLogs[gameLogs['Additional information'].isnull()]
    gameLogs = gameLogs[gameLogs['Acquisition information']!="P"]
    gameLogs = gameLogs[['Date','Visiting team','Visiting league','Home team','Home league'
                        ,'Visiting score','Home score','Home plate umpire ID','Visiting team manager ID'
                        ,'Home team manager ID','Visiting starting pitcher ID','Home starting pitcher ID'
                        ,'Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID'
                        ,'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID'
                        ,'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID'
                        ,'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID'
                        ,'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID'
                        ,'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID']]
    gameLogs = gameLogs.dropna()
    return gameLogs

mlb_dataFilter(mlb_gameLogsLoaderCSV()).to_csv(r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Daten\Baseball Data up to 2020\GameLogs\~mlb_filtered_game_logs.csv', index = False)