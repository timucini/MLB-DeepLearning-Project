import pandas as pd
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'


def load_team_statistics():
    data_folder = Path("./Data")
    base_data_file = data_folder / "f_Teams.csv"
    df_teams = pd.read_csv(base_data_file)
    return df_teams


def team_stats_last_season(year, team_id):
    df_teams = load_team_statistics()
    df_test = df_teams.loc[(df_teams['yearID'] == year - 1) & (df_teams['teamID'] == team_id)]
    df_test['p_exp'] = pythagorean_expectation(df_test)
    print(df_test.head())
    return df_test


def pythagorean_expectation(last_season_stats):
    py_expectation = (last_season_stats['R'] ** 1.83) / (
            last_season_stats['R'] ** 1.83 + last_season_stats['RA'] ** 1.83)
    print(py_expectation)
    return py_expectation


team_stats_last_season(2018, "ARI")
