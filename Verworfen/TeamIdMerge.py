import pandas as pd
from pathlib import Path
from enum import Enum

pd.options.mode.chained_assignment = None  # default='warn'


class Type(Enum):
    home = 1
    visiting = 2


def get_last_occurrence(team_id):
    data_folder = Path("../Filtered")
    game_logs_file = data_folder / "_mlb_filtered_GameLogs.csv"
    df_game_logs = pd.read_csv(game_logs_file, index_col=0)
    last_visiting_game = df_game_logs.where(df_game_logs['Visiting team'] == team_id).last_valid_index()
    last_home_game = df_game_logs.where(df_game_logs['Home team'] == team_id).last_valid_index()
    if last_visiting_game > last_home_game:
        last_occurrence_row = pd.array([last_visiting_game, Type.visiting])
    else:
        last_occurrence_row = pd.array([last_home_game, Type.home])
    return last_occurrence_row


def get_all_teams():
    data_folder = Path("../Filtered")
    team_file = data_folder / "_mlb_filtered_Teams.csv"
    df_teams = pd.read_csv(team_file)
    df_unique_id = df_teams['teamID'].unique()
    print(df_unique_id)
    print(df_unique_id.size)


def get_team_data(team_id_first, team_id_second):
    data_folder = Path("../Replaced")
    all_game_file = data_folder / "_mlb_remerged_all.csv"
    df_all_games = pd.read_csv(all_game_file)
    df_columns = df_all_games.columns.values.tolist()
    home_team_columns = [i for i in df_columns if "Home" in i]
    visiting_team_columns = [i for i in df_columns if "Visiting" in i]
    df_last_occurrence_first = get_last_occurrence(team_id_first)
    if df_last_occurrence_first[1] == isinstance(Type.home, Type):
        first_team_data = df_all_games.loc[df_last_occurrence_first[0], home_team_columns]
        print(first_team_data)
    else:
        first_team_data = df_all_games.loc[df_last_occurrence_first[0], visiting_team_columns]
        print(first_team_data)


get_team_data('TBA', 'BOS')
#get_all_teams()
