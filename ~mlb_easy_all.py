import pandas as pd
import numpy as np

path = r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Projekt'
scorings    = pd.read_csv(path+r'\Remerged\_mlb_recreated_scoring.csv', index_col=False)
people      = pd.read_csv(path+r'\Remerged\_mlb_remerged_People.csv', index_col=False).drop(columns=['row'])
teams       = pd.read_csv(path+r'\Remerged\_mlb_remerged_Teams.csv', index_col=False).drop(columns=['row'])
managers    = pd.read_csv(path+r'\Remerged\_mlb_remerged_Managers.csv', index_col=False).drop(columns=['row'])
pitchings   = pd.read_csv(path+r'\Remerged\_mlb_remerged_Pitchers.csv', index_col=False).drop(columns=['row'])
battings    = pd.read_csv(path+r'\Remerged\_mlb_remerged_Batting.csv', index_col=False).drop(columns=['row'])
fieldings   = pd.read_csv(path+r'\Remerged\_mlb_remerged_Fielding.csv', index_col=False).drop(columns=['row'])
mlbData     = [scorings, teams, managers, people, pitchings, battings, fieldings]
mlbAll      = pd.concat(mlbData, axis=1)
mlbAll      = mlbAll.rename(columns={'Date home team':'Date'})
columns =   ['row','Date','Home team','Home score','Home league AL','Visiting team','Visiting score','Visiting league AL','Home team win','Home team odd','Home win ratio','Home score ratio'
            ,'Home odd ratio','Visiting versus win ratio','Visiting versus score ratio','Visiting versus odd ratio'
            ,'Visiting win ratio','Visiting score ratio','Visiting odd ratio','Home versus win ratio','Home versus score ratio'
            ,'Home versus odd ratio','Visiting starting pitcher  age','Visiting starting pitcher weight'
            ,'Visiting starting pitcher height','Visiting starting pitcher bats right','Visiting starting pitcher bats left'
            ,'Visiting starting pitcher throws right','Home starting pitcher  age','Home starting pitcher weight'
            ,'Home starting pitcher height','Home starting pitcher bats right','Home starting pitcher bats left'
            ,'Home starting pitcher throws right','Visiting starting player 1  age','Visiting starting player 1 weight'
            ,'Visiting starting player 1 height','Visiting starting player 1 bats right','Visiting starting player 1 bats left'
            ,'Visiting starting player 1 throws right','Visiting starting player 2  age','Visiting starting player 2 weight'
            ,'Visiting starting player 2 height','Visiting starting player 2 bats right','Visiting starting player 2 bats left'
            ,'Visiting starting player 2 throws right','Visiting starting player 3  age','Visiting starting player 3 weight'
            ,'Visiting starting player 3 height','Visiting starting player 3 bats right','Visiting starting player 3 bats left'
            ,'Visiting starting player 3 throws right','Visiting starting player 4  age','Visiting starting player 4 weight'
            ,'Visiting starting player 4 height','Visiting starting player 4 bats right','Visiting starting player 4 bats left'
            ,'Visiting starting player 4 throws right','Visiting starting player 5  age','Visiting starting player 5 weight'
            ,'Visiting starting player 5 height','Visiting starting player 5 bats right','Visiting starting player 5 bats left'
            ,'Visiting starting player 5 throws right','Visiting starting player 6  age','Visiting starting player 6 weight'
            ,'Visiting starting player 6 height','Visiting starting player 6 bats right','Visiting starting player 6 bats left'
            ,'Visiting starting player 6 throws right','Visiting starting player 7  age','Visiting starting player 7 weight'
            ,'Visiting starting player 7 height','Visiting starting player 7 bats right','Visiting starting player 7 bats left'
            ,'Visiting starting player 7 throws right','Visiting starting player 8  age','Visiting starting player 8 weight'
            ,'Visiting starting player 8 height','Visiting starting player 8 bats right','Visiting starting player 8 bats left'
            ,'Visiting starting player 8 throws right','Visiting starting player 9  age','Visiting starting player 9 weight'
            ,'Visiting starting player 9 height','Visiting starting player 9 bats right','Visiting starting player 9 bats left'
            ,'Visiting starting player 9 throws right','Home starting player 1  age','Home starting player 1 weight'
            ,'Home starting player 1 height','Home starting player 1 bats right','Home starting player 1 bats left'
            ,'Home starting player 1 throws right','Home starting player 2  age','Home starting player 2 weight'
            ,'Home starting player 2 height','Home starting player 2 bats right','Home starting player 2 bats left'
            ,'Home starting player 2 throws right','Home starting player 3  age','Home starting player 3 weight'
            ,'Home starting player 3 height','Home starting player 3 bats right','Home starting player 3 bats left'
            ,'Home starting player 3 throws right','Home starting player 4  age','Home starting player 4 weight'
            ,'Home starting player 4 height','Home starting player 4 bats right','Home starting player 4 bats left'
            ,'Home starting player 4 throws right','Home starting player 5  age','Home starting player 5 weight'
            ,'Home starting player 5 height','Home starting player 5 bats right','Home starting player 5 bats left'
            ,'Home starting player 5 throws right','Home starting player 6  age','Home starting player 6 weight'
            ,'Home starting player 6 height','Home starting player 6 bats right','Home starting player 6 bats left'
            ,'Home starting player 6 throws right','Home starting player 7  age','Home starting player 7 weight'
            ,'Home starting player 7 height','Home starting player 7 bats right','Home starting player 7 bats left'
            ,'Home starting player 7 throws right','Home starting player 8  age','Home starting player 8 weight'
            ,'Home starting player 8 height','Home starting player 8 bats right','Home starting player 8 bats left'
            ,'Home starting player 8 throws right','Home starting player 9  age','Home starting player 9 weight'
            ,'Home starting player 9 height','Home starting player 9 bats right','Home starting player 9 bats left'
            ,'Home starting player 9 throws right','Rank home team'
            ,'Games home team','Wins home team','Losses home team','Division winner home team','League winner home team'
            ,'World series winner home team','Runs scored home team','At bats home team','Hits by batters home team'
            ,'Doubles home team','Triples home team','Homeruns home team','Walks home team','Strikeouts home team'
            ,'Stolen bases home team','Cought stealing home team','Batters hit by pitch home team','Sacrifice flies home team'
            ,'Opponents runs scored home team','Earned runs allowed home team','Earned runs average home team','Shutouts home team'
            ,'Saves home team','Hits allowed home team','Homeruns allowed home team','Walks allowed home team'
            ,'Strikeouts allowed home team','Errors home team','Double plays home team','Fielding percentage home team'
            ,'division C home team','division E home team','division W home team'
            ,'Rank visiting team','Games visiting team','Wins visiting team','Losses visiting team'
            ,'Division winner visiting team','League winner visiting team','World series winner visiting team'
            ,'Runs scored visiting team','At bats visiting team','Hits by batters visiting team','Doubles visiting team'
            ,'Triples visiting team','Homeruns visiting team','Walks visiting team','Strikeouts visiting team'
            ,'Stolen bases visiting team','Cought stealing visiting team','Batters hit by pitch visiting team'
            ,'Sacrifice flies visiting team','Opponents runs scored visiting team','Earned runs allowed visiting team'
            ,'Earned runs average visiting team','Shutouts visiting team','Saves visiting team','Hits allowed visiting team'
            ,'Homeruns allowed visiting team','Walks allowed visiting team','Strikeouts allowed visiting team'
            ,'Errors visiting team','Double plays visiting team','Fielding percentage visiting team','division C visiting team'
            ,'division E visiting team','division W visiting team','Games home manager','Wins home manager','Losses home manager'
            ,'Games visiting manager','Wins visiting manager','Losses visiting manager','Hits home pitcher'
            ,'Earned Runs home pitcher','Homeruns home pitcher','Walks home pitcher','Strikeouts home pitcher'
            ,'Opponent batting average home pitcher','Earned run average home pitcher','Intentional walks home pitcher'
            ,'Wild pitches home pitcher','Batters hit by pitch home pitcher','Balks home pitcher','Batters faced home pitcher'
            ,'Runs allowed home pitcher','Batters sacrifices home pitcher','Batters sacrifice flies home pitcher'
            ,'Grounded into double plays home pitcher','Hits visiting pitcher','Earned Runs visiting pitcher'
            ,'Homeruns visiting pitcher','Walks visiting pitcher','Strikeouts visiting pitcher'
            ,'Opponent batting average visiting pitcher','Earned run average visiting pitcher','Intentional walks visiting pitcher'
            ,'Wild pitches visiting pitcher','Batters hit by pitch visiting pitcher','Balks visiting pitcher'
            ,'Batters faced visiting pitcher','Runs allowed visiting pitcher','Batters sacrifices visiting pitcher'
            ,'Batters sacrifice flies visiting pitcher','Grounded into double plays visiting pitcher'
            ,'Visiting starting pitcher At bats','Visiting starting pitcher Runs','Visiting starting pitcher Hits'
            ,'Visiting starting pitcher Doubles','Visiting starting pitcher Triples','Visiting starting pitcher Homeruns'
            ,'Visiting starting pitcher Runs batted in','Visiting starting pitcher Stolen bases'
            ,'Visiting starting pitcher Caught stealing','Visiting starting pitcher Base on balls'
            ,'Visiting starting pitcher Strikeouts','Visiting starting pitcher Intentional walks'
            ,'Visiting starting pitcher Hit by pitch','Visiting starting pitcher Sacrifice hits'
            ,'Visiting starting pitcher Sacrifice flies','Visiting starting pitcher Grounded into double plays'
            ,'Home starting pitcher At bats','Home starting pitcher Runs','Home starting pitcher Hits'
            ,'Home starting pitcher Doubles','Home starting pitcher Triples','Home starting pitcher Homeruns'
            ,'Home starting pitcher Runs batted in','Home starting pitcher Stolen bases','Home starting pitcher Caught stealing'
            ,'Home starting pitcher Base on balls','Home starting pitcher Strikeouts','Home starting pitcher Intentional walks'
            ,'Home starting pitcher Hit by pitch','Home starting pitcher Sacrifice hits','Home starting pitcher Sacrifice flies'
            ,'Home starting pitcher Grounded into double plays','Visiting starting player 1 At bats'
            ,'Visiting starting player 1 Runs','Visiting starting player 1 Hits','Visiting starting player 1 Doubles'
            ,'Visiting starting player 1 Triples','Visiting starting player 1 Homeruns','Visiting starting player 1 Runs batted in'
            ,'Visiting starting player 1 Stolen bases','Visiting starting player 1 Caught stealing'
            ,'Visiting starting player 1 Base on balls','Visiting starting player 1 Strikeouts'
            ,'Visiting starting player 1 Intentional walks','Visiting starting player 1 Hit by pitch'
            ,'Visiting starting player 1 Sacrifice hits','Visiting starting player 1 Sacrifice flies'
            ,'Visiting starting player 1 Grounded into double plays','Visiting starting player 2 At bats'
            ,'Visiting starting player 2 Runs','Visiting starting player 2 Hits','Visiting starting player 2 Doubles'
            ,'Visiting starting player 2 Triples','Visiting starting player 2 Homeruns','Visiting starting player 2 Runs batted in'
            ,'Visiting starting player 2 Stolen bases','Visiting starting player 2 Caught stealing'
            ,'Visiting starting player 2 Base on balls','Visiting starting player 2 Strikeouts'
            ,'Visiting starting player 2 Intentional walks','Visiting starting player 2 Hit by pitch'
            ,'Visiting starting player 2 Sacrifice hits','Visiting starting player 2 Sacrifice flies'
            ,'Visiting starting player 2 Grounded into double plays','Visiting starting player 3 At bats'
            ,'Visiting starting player 3 Runs','Visiting starting player 3 Hits','Visiting starting player 3 Doubles'
            ,'Visiting starting player 3 Triples','Visiting starting player 3 Homeruns','Visiting starting player 3 Runs batted in'
            ,'Visiting starting player 3 Stolen bases','Visiting starting player 3 Caught stealing'
            ,'Visiting starting player 3 Base on balls','Visiting starting player 3 Strikeouts'
            ,'Visiting starting player 3 Intentional walks','Visiting starting player 3 Hit by pitch'
            ,'Visiting starting player 3 Sacrifice hits','Visiting starting player 3 Sacrifice flies'
            ,'Visiting starting player 3 Grounded into double plays','Visiting starting player 4 At bats'
            ,'Visiting starting player 4 Runs','Visiting starting player 4 Hits','Visiting starting player 4 Doubles'
            ,'Visiting starting player 4 Triples','Visiting starting player 4 Homeruns','Visiting starting player 4 Runs batted in'
            ,'Visiting starting player 4 Stolen bases','Visiting starting player 4 Caught stealing'
            ,'Visiting starting player 4 Base on balls','Visiting starting player 4 Strikeouts'
            ,'Visiting starting player 4 Intentional walks','Visiting starting player 4 Hit by pitch'
            ,'Visiting starting player 4 Sacrifice hits','Visiting starting player 4 Sacrifice flies'
            ,'Visiting starting player 4 Grounded into double plays','Visiting starting player 5 At bats'
            ,'Visiting starting player 5 Runs','Visiting starting player 5 Hits','Visiting starting player 5 Doubles'
            ,'Visiting starting player 5 Triples','Visiting starting player 5 Homeruns','Visiting starting player 5 Runs batted in'
            ,'Visiting starting player 5 Stolen bases','Visiting starting player 5 Caught stealing'
            ,'Visiting starting player 5 Base on balls','Visiting starting player 5 Strikeouts'
            ,'Visiting starting player 5 Intentional walks','Visiting starting player 5 Hit by pitch'
            ,'Visiting starting player 5 Sacrifice hits','Visiting starting player 5 Sacrifice flies'
            ,'Visiting starting player 5 Grounded into double plays','Visiting starting player 6 At bats'
            ,'Visiting starting player 6 Runs','Visiting starting player 6 Hits','Visiting starting player 6 Doubles'
            ,'Visiting starting player 6 Triples','Visiting starting player 6 Homeruns','Visiting starting player 6 Runs batted in'
            ,'Visiting starting player 6 Stolen bases','Visiting starting player 6 Caught stealing'
            ,'Visiting starting player 6 Base on balls','Visiting starting player 6 Strikeouts'
            ,'Visiting starting player 6 Intentional walks','Visiting starting player 6 Hit by pitch'
            ,'Visiting starting player 6 Sacrifice hits','Visiting starting player 6 Sacrifice flies'
            ,'Visiting starting player 6 Grounded into double plays','Visiting starting player 7 At bats'
            ,'Visiting starting player 7 Runs','Visiting starting player 7 Hits','Visiting starting player 7 Doubles'
            ,'Visiting starting player 7 Triples','Visiting starting player 7 Homeruns','Visiting starting player 7 Runs batted in'
            ,'Visiting starting player 7 Stolen bases','Visiting starting player 7 Caught stealing'
            ,'Visiting starting player 7 Base on balls','Visiting starting player 7 Strikeouts'
            ,'Visiting starting player 7 Intentional walks','Visiting starting player 7 Hit by pitch'
            ,'Visiting starting player 7 Sacrifice hits','Visiting starting player 7 Sacrifice flies'
            ,'Visiting starting player 7 Grounded into double plays','Visiting starting player 8 At bats'
            ,'Visiting starting player 8 Runs','Visiting starting player 8 Hits','Visiting starting player 8 Doubles'
            ,'Visiting starting player 8 Triples','Visiting starting player 8 Homeruns','Visiting starting player 8 Runs batted in'
            ,'Visiting starting player 8 Stolen bases','Visiting starting player 8 Caught stealing'
            ,'Visiting starting player 8 Base on balls','Visiting starting player 8 Strikeouts'
            ,'Visiting starting player 8 Intentional walks','Visiting starting player 8 Hit by pitch'
            ,'Visiting starting player 8 Sacrifice hits','Visiting starting player 8 Sacrifice flies'
            ,'Visiting starting player 8 Grounded into double plays','Visiting starting player 9 At bats'
            ,'Visiting starting player 9 Runs','Visiting starting player 9 Hits','Visiting starting player 9 Doubles'
            ,'Visiting starting player 9 Triples','Visiting starting player 9 Homeruns','Visiting starting player 9 Runs batted in'
            ,'Visiting starting player 9 Stolen bases','Visiting starting player 9 Caught stealing'
            ,'Visiting starting player 9 Base on balls','Visiting starting player 9 Strikeouts'
            ,'Visiting starting player 9 Intentional walks','Visiting starting player 9 Hit by pitch'
            ,'Visiting starting player 9 Sacrifice hits','Visiting starting player 9 Sacrifice flies'
            ,'Visiting starting player 9 Grounded into double plays','Home starting player 1 At bats'
            ,'Home starting player 1 Runs','Home starting player 1 Hits','Home starting player 1 Doubles'
            ,'Home starting player 1 Triples','Home starting player 1 Homeruns','Home starting player 1 Runs batted in'
            ,'Home starting player 1 Stolen bases','Home starting player 1 Caught stealing','Home starting player 1 Base on balls'
            ,'Home starting player 1 Strikeouts','Home starting player 1 Intentional walks','Home starting player 1 Hit by pitch'
            ,'Home starting player 1 Sacrifice hits','Home starting player 1 Sacrifice flies'
            ,'Home starting player 1 Grounded into double plays','Home starting player 2 At bats','Home starting player 2 Runs'
            ,'Home starting player 2 Hits','Home starting player 2 Doubles','Home starting player 2 Triples'
            ,'Home starting player 2 Homeruns','Home starting player 2 Runs batted in','Home starting player 2 Stolen bases'
            ,'Home starting player 2 Caught stealing','Home starting player 2 Base on balls','Home starting player 2 Strikeouts'
            ,'Home starting player 2 Intentional walks','Home starting player 2 Hit by pitch'
            ,'Home starting player 2 Sacrifice hits','Home starting player 2 Sacrifice flies'
            ,'Home starting player 2 Grounded into double plays','Home starting player 3 At bats','Home starting player 3 Runs'
            ,'Home starting player 3 Hits','Home starting player 3 Doubles','Home starting player 3 Triples'
            ,'Home starting player 3 Homeruns','Home starting player 3 Runs batted in','Home starting player 3 Stolen bases'
            ,'Home starting player 3 Caught stealing','Home starting player 3 Base on balls','Home starting player 3 Strikeouts'
            ,'Home starting player 3 Intentional walks','Home starting player 3 Hit by pitch'
            ,'Home starting player 3 Sacrifice hits','Home starting player 3 Sacrifice flies'
            ,'Home starting player 3 Grounded into double plays','Home starting player 4 At bats','Home starting player 4 Runs'
            ,'Home starting player 4 Hits','Home starting player 4 Doubles','Home starting player 4 Triples'
            ,'Home starting player 4 Homeruns','Home starting player 4 Runs batted in','Home starting player 4 Stolen bases'
            ,'Home starting player 4 Caught stealing','Home starting player 4 Base on balls','Home starting player 4 Strikeouts'
            ,'Home starting player 4 Intentional walks','Home starting player 4 Hit by pitch'
            ,'Home starting player 4 Sacrifice hits','Home starting player 4 Sacrifice flies'
            ,'Home starting player 4 Grounded into double plays','Home starting player 5 At bats','Home starting player 5 Runs'
            ,'Home starting player 5 Hits','Home starting player 5 Doubles','Home starting player 5 Triples'
            ,'Home starting player 5 Homeruns','Home starting player 5 Runs batted in','Home starting player 5 Stolen bases'
            ,'Home starting player 5 Caught stealing','Home starting player 5 Base on balls','Home starting player 5 Strikeouts'
            ,'Home starting player 5 Intentional walks','Home starting player 5 Hit by pitch'
            ,'Home starting player 5 Sacrifice hits','Home starting player 5 Sacrifice flies'
            ,'Home starting player 5 Grounded into double plays','Home starting player 6 At bats','Home starting player 6 Runs'
            ,'Home starting player 6 Hits','Home starting player 6 Doubles','Home starting player 6 Triples'
            ,'Home starting player 6 Homeruns','Home starting player 6 Runs batted in','Home starting player 6 Stolen bases'
            ,'Home starting player 6 Caught stealing','Home starting player 6 Base on balls','Home starting player 6 Strikeouts'
            ,'Home starting player 6 Intentional walks','Home starting player 6 Hit by pitch'
            ,'Home starting player 6 Sacrifice hits','Home starting player 6 Sacrifice flies'
            ,'Home starting player 6 Grounded into double plays','Home starting player 7 At bats','Home starting player 7 Runs'
            ,'Home starting player 7 Hits','Home starting player 7 Doubles','Home starting player 7 Triples'
            ,'Home starting player 7 Homeruns','Home starting player 7 Runs batted in','Home starting player 7 Stolen bases'
            ,'Home starting player 7 Caught stealing','Home starting player 7 Base on balls','Home starting player 7 Strikeouts'
            ,'Home starting player 7 Intentional walks','Home starting player 7 Hit by pitch'
            ,'Home starting player 7 Sacrifice hits','Home starting player 7 Sacrifice flies'
            ,'Home starting player 7 Grounded into double plays','Home starting player 8 At bats','Home starting player 8 Runs'
            ,'Home starting player 8 Hits','Home starting player 8 Doubles','Home starting player 8 Triples'
            ,'Home starting player 8 Homeruns','Home starting player 8 Runs batted in','Home starting player 8 Stolen bases'
            ,'Home starting player 8 Caught stealing','Home starting player 8 Base on balls','Home starting player 8 Strikeouts'
            ,'Home starting player 8 Intentional walks','Home starting player 8 Hit by pitch'
            ,'Home starting player 8 Sacrifice hits','Home starting player 8 Sacrifice flies'
            ,'Home starting player 8 Grounded into double plays','Home starting player 9 At bats','Home starting player 9 Runs'
            ,'Home starting player 9 Hits','Home starting player 9 Doubles','Home starting player 9 Triples'
            ,'Home starting player 9 Homeruns','Home starting player 9 Runs batted in','Home starting player 9 Stolen bases'
            ,'Home starting player 9 Caught stealing','Home starting player 9 Base on balls','Home starting player 9 Strikeouts'
            ,'Home starting player 9 Intentional walks','Home starting player 9 Hit by pitch'
            ,'Home starting player 9 Sacrifice hits','Home starting player 9 Sacrifice flies'
            ,'Home starting player 9 Grounded into double plays','Visiting starting pitcher Putouts'
            ,'Visiting starting pitcher Assists','Visiting starting pitcher Error','Visiting starting pitcher Double plays'
            ,'Home starting pitcher Putouts','Home starting pitcher Assists','Home starting pitcher Error'
            ,'Home starting pitcher Double plays','Visiting starting player 1 Putouts','Visiting starting player 1 Assists'
            ,'Visiting starting player 1 Error','Visiting starting player 1 Double plays','Visiting starting player 2 Putouts'
            ,'Visiting starting player 2 Assists','Visiting starting player 2 Error','Visiting starting player 2 Double plays'
            ,'Visiting starting player 3 Putouts','Visiting starting player 3 Assists','Visiting starting player 3 Error'
            ,'Visiting starting player 3 Double plays','Visiting starting player 4 Putouts','Visiting starting player 4 Assists'
            ,'Visiting starting player 4 Error','Visiting starting player 4 Double plays','Visiting starting player 5 Putouts'
            ,'Visiting starting player 5 Assists','Visiting starting player 5 Error','Visiting starting player 5 Double plays'
            ,'Visiting starting player 6 Putouts','Visiting starting player 6 Assists','Visiting starting player 6 Error'
            ,'Visiting starting player 6 Double plays','Visiting starting player 7 Putouts','Visiting starting player 7 Assists'
            ,'Visiting starting player 7 Error','Visiting starting player 7 Double plays','Visiting starting player 8 Putouts'
            ,'Visiting starting player 8 Assists','Visiting starting player 8 Error','Visiting starting player 8 Double plays'
            ,'Visiting starting player 9 Putouts','Visiting starting player 9 Assists','Visiting starting player 9 Error'
            ,'Visiting starting player 9 Double plays','Home starting player 1 Putouts','Home starting player 1 Assists'
            ,'Home starting player 1 Error','Home starting player 1 Double plays','Home starting player 2 Putouts'
            ,'Home starting player 2 Assists','Home starting player 2 Error','Home starting player 2 Double plays'
            ,'Home starting player 3 Putouts','Home starting player 3 Assists','Home starting player 3 Error'
            ,'Home starting player 3 Double plays','Home starting player 4 Putouts','Home starting player 4 Assists'
            ,'Home starting player 4 Error','Home starting player 4 Double plays','Home starting player 5 Putouts'
            ,'Home starting player 5 Assists','Home starting player 5 Error','Home starting player 5 Double plays'
            ,'Home starting player 6 Putouts','Home starting player 6 Assists','Home starting player 6 Error'
            ,'Home starting player 6 Double plays','Home starting player 7 Putouts','Home starting player 7 Assists'
            ,'Home starting player 7 Error','Home starting player 7 Double plays','Home starting player 8 Putouts'
            ,'Home starting player 8 Assists','Home starting player 8 Error','Home starting player 8 Double plays'
            ,'Home starting player 9 Putouts','Home starting player 9 Assists','Home starting player 9 Error'
            ,'Home starting player 9 Double plays']

mlbAll[columns].to_csv(path+r'\_mlb_remerged_all.csv', index = False)