import pandas as pd
from pathlib import Path
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

def filter_remerged_battings():
    data_folder = Path("./Remerged")
    team_data_file = data_folder / "_mlb_remerged_Batting.csv"
    df_battings = pd.read_csv(team_data_file)
    df_battings['Home Team avg At bats'] = np.nanmean([df_battings['Visiting starting player 1 At bats'], df_battings['Visiting starting player 2 At bats']])
    print(df_battings['Home Team avg At bats'])
    #df_battings['Home Team avg Runs']
    #df_battings['Home Team avg Hits']
    #df_battings['Home Team avg Doubles']
    #df_battings['Home Team avg Triples']
    #df_battings['Home Team avg Homeruns']
    #df_battings['Home Team avg Runs batted in']
    #df_battings['Home Team avg Stolen Bases']
    #df_battings['Home Team avg Caught Stealing']
    #df_battings['Home Team avg Base on Balls']
    #df_battings['Home Team avg Strikeouts']
    #df_battings['Home Team avg Hit by pitch']
    #df_battings['Home Team avg Sacrifice Hits']
    #df_battings['Visiting Team avg At bats']
    #df_battings['Visiting Team avg Runs']
    #df_battings['Visiting Team avg Hits']
    #df_battings['Visiting Team avg Doubles']
    #df_battings['Visiting Team avg Triples']
    #df_battings['Visiting Team avg Homeruns']
    #df_battings['Visiting Team avg Runs batted in']
    #df_battings['Visiting Team avg Stolen Bases']
    #df_battings['Visiting Team avg Caught Stealing']
    #df_battings['Visiting Team avg Base on Balls']
    #df_battings['Visiting Team avg Strikeouts']
    #df_battings['Visiting Team avg Hit by pitch']
    #df_battings['Visiting Team avg Sacrifice Hits']

filter_remerged_battings()