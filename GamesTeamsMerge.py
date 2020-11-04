import pandas as pd
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'


def merge_team_statistics():
    data_folder = Path("./Filtered")
    team_data_file = data_folder / "_mlb_filtered_Teams.csv"
    df_teams = pd.read_csv(team_data_file)
    df_teams = filter_team_data(df_teams)
    game_data_file = data_folder / "_mlb_filtered_GameLogs.csv"
    df_games = pd.read_csv(game_data_file)
    df_games['Date'] = pd.to_datetime(df_games['Date']).dt.year - 1
    df_merged_data = pd.merge(df_games, df_teams, left_on=['Date', 'Visiting team'],
                              right_on=['yearID', 'teamID'], how='left')
    df_both_teams_merged = pd.merge(df_merged_data, df_teams, left_on=['Date', 'Home team'],
                                    right_on=['yearID', 'teamID'], how='left', suffixes=['', '_home'])
    merge_result(df_both_teams_merged)


def filter_team_data(df_teams):
    df_teams['Division winner'] = pd.get_dummies(df_teams['Division winner'])
    df_teams['League winner'] = pd.get_dummies(df_teams['League winner'])
    df_teams['World series winner'] = pd.get_dummies(df_teams['World series winner'])
    df_teams['Division'] = pd.get_dummies(df_teams['Division'])
    df_teams['Pythagorean_expectation'] = (df_teams['Runs scored'] ** 1.83) / (
            df_teams['Runs scored'] ** 1.83 + df_teams['Opponents runs scored'] ** 1.83)
    return df_teams


def merge_result(df_both_teams_merged):
    columns = ['row', 'Division', 'Rank', 'Games', 'Wins', 'Losses', 'Division winner',
               'League winner', 'World series winner', 'Runs scored', 'At bats', 'Hits by batters',
               'Doubles', 'Triples', 'Homeruns', 'Walks', 'Strikeouts', 'Stolen bases', 'Cought stealing',
               'Batters hit by pitch', 'Sacrifice flies', 'Opponents runs scored', 'Earned runs allowed',
               'Earned runs average', 'Shutouts', 'Saves', 'Hits allowed', 'Homeruns allowed', 'Walks allowed',
               'Strikeouts allowed', 'Errors', 'Double plays', 'Fielding percentage', 'Pythagorean_expectation',
               'Division_home', 'Rank_home', 'Games_home', 'Wins_home', 'Losses_home',
               'Division winner_home', 'League winner_home', 'World series winner_home', 'Runs scored_home',
               'At bats_home', 'Hits by batters_home', 'Doubles_home', 'Triples_home', 'Homeruns_home',
               'Walks_home', 'Strikeouts_home', 'Stolen bases_home', 'Cought stealing_home',
               'Batters hit by pitch_home', 'Sacrifice flies_home', 'Opponents runs scored_home',
               'Earned runs allowed_home', 'Earned runs average_home', 'Shutouts_home', 'Saves_home',
               'Hits allowed_home', 'Homeruns allowed_home', 'Walks allowed_home', 'Strikeouts allowed_home',
               'Errors_home', 'Double plays_home', 'Fielding percentage_home', 'Pythagorean_expectation_home'
               ]
    df_export = df_both_teams_merged[columns]
    path = Path("./Merged")
    df_export.to_csv(path/'_mlb_merged_Teams.csv', index=False)


merge_team_statistics()


