import pandas as pd
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'


def merge_team_statistics():
    data_folder = Path("./Data")
    team_data_file = data_folder / "f_Teams.csv"
    df_teams = pd.read_csv(team_data_file)
    # increment relevant year for teamstatistics
    df_teams['yearID'] = df_teams['yearID'] + 1
    game_data_file = data_folder / "mlb_filtered_game_logs.csv"
    df_games = pd.read_csv(game_data_file)
    df_games['Date'] = pd.to_datetime(df_games['Date']).dt.year
    df_merged_data = pd.merge(df_games, df_teams, left_on=['Date', 'Visiting team'],
                              right_on=['yearID', 'teamID'], how='left', suffixes=['', '_away'])
    df_both_teams_merged = pd.merge(df_merged_data, df_teams, left_on=['Date', 'Home team'],
                                    right_on=['yearID', 'teamID'], how='left', suffixes=['', '_home'])
    print(df_both_teams_merged.info())


def filter_team_data():
    data_folder = Path("./Data")
    team_data_file = data_folder / "f_Teams.csv"
    df_teams = pd.read_csv(team_data_file)
    df_teams['DivWin'] = df_teams['DivWin'].cat
    print(df_teams['DivWin'])

#filter_team_data()
merge_team_statistics()


