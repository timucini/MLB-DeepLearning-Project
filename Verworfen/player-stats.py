import pandas as pd
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'


def loadAppearances():
    df_player = pd.read_csv('/Users/sewerynkozlowski/Desktop/HTW_2_Semester/Analytische Anwendungen/MLB/dataset/baseballdatabank-master/core/Appearances.csv')
    return df_player


def player_stats_last_season(year, player_id):
    df_player = loadAppearances()
    df_result = df_player.loc[(df_player['yearID'] == year - 1) & (df_player['playerID'] == player_id)]
    print(df_result)
    return df_result

def player_stats_of_team_last_season(year, team_id):
    df_player = loadAppearances()
    df_result = df_player.loc[(df_player['yearID'] == year - 1) & (df_player['teamID'] == team_id)]
    print(df_result)
    return df_result


#player_stats_last_season(1872, "abercda01")
player_stats_of_team_last_season(1872, "TRO")