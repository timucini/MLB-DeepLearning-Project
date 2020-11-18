import pandas as pd
from pathlib import Path
from enum import Enum

pd.options.mode.chained_assignment = None  # default='warn'


class Location(Enum):
    home = 1
    visiting = 2

#not used atm
def get_last_occurrence(team_id, location):
    data_folder = Path("../")
    all_game_file = data_folder / "_mlb_remerged_all.csv"
    df_game_logs = pd.read_csv(all_game_file, index_col=0)
    if location == Location.home:
        last_occurrence = df_game_logs.where(df_game_logs['Home team'] == team_id).last_valid_index()
        print('home')
    else:
        last_occurrence = df_game_logs.where(df_game_logs['Visiting team'] == team_id).last_valid_index()
        print('away')
    return last_occurrence

#not used atm
def get_all_teams():
    data_folder = Path("../")
    all_game_file = data_folder / "_mlb_remerged_all.csv"
    df_teams = pd.read_csv(all_game_file)
    df_unique_id = df_teams['teamID'].unique()
    print(df_unique_id)
    print(df_unique_id.size)


def get_team_data(home_id, visit_id):
    data_folder = Path("../")
    all_game_file = data_folder / "_mlb_remerged_all.csv"
    df_all_games = pd.read_csv(all_game_file)
    print("Number of Columns: " + str(len(df_all_games.columns)-2))
    df_columns = df_all_games.columns.values.tolist()
    home_team_columns = [i for i in df_columns if "Home" in i]
    print("Number of Home Columns: " + str(len(home_team_columns)))
    visiting_team_columns = [i for i in df_columns if "Visiting" in i]
    print("Number of Visiting Columns: " + str(len(visiting_team_columns)))
    last_occurrence_home = df_all_games.where(df_all_games['Home team'] == home_id).last_valid_index()
    #home_team_data = df_all_games.iloc[[get_last_occurrence(home_id, Location.home)]]
    home_team_data = df_all_games.iloc[[last_occurrence_home]]
    home_team_to_home_column = home_team_data[home_team_columns]
    last_occurrence_away = df_all_games.where(df_all_games['Visiting team'] == visit_id).last_valid_index()
    #visiting_team_data = df_all_games.iloc[[get_last_occurrence(visit_id, Location.visiting)]]
    visiting_team_data = df_all_games.iloc[[last_occurrence_away]]
    visiting_team_to_visiting_column = visiting_team_data[visiting_team_columns]
    df_merged_data = pd.concat([home_team_to_home_column,
                                visiting_team_to_visiting_column.set_index(home_team_to_home_column.index)], axis=1)
    print(df_merged_data)

get_team_data('SEA', 'DET')
#get_all_teams()
