import pandas as pd
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'


def filter_remerged_teams():
    data_folder = Path("./Remerged")
    team_data_file = data_folder / "_mlb_remerged_Teams.csv"
    df_teams = pd.read_csv(team_data_file)
    df_teams['Win_ratio home team'] = df_teams['Wins home team'] / df_teams['Losses home team']
    df_teams['Win_ratio visiting team'] = df_teams['Wins visiting team'] / df_teams['Losses visiting team']
    df_teams['BABIP home team'] = ((df_teams['Hits by batters home team'] - df_teams['Homeruns home team']) /
                                   (df_teams['At bats home team'] - df_teams['Strikeouts home team'] -
                                    df_teams['Homeruns home team'] + df_teams['Sacrifice flies home team']))
    df_teams['BABIP visiting team'] = ((df_teams['Hits by batters visiting team'] - df_teams['Homeruns visiting team'])/
                                    (df_teams['At bats visiting team'] - df_teams['Strikeouts visiting team'] -
                                    df_teams['Homeruns visiting team'] + df_teams['Sacrifice flies visiting team']))
    print(df_teams['BABIP visiting team'])
    columns_to_drop = ['Games home team', 'Cought stealing home team',
                       'Sacrifice flies home team', 'Sacrifice flies home team',
                       'Games visiting team', 'Cought stealing visiting team',
                       'Sacrifice flies visiting team', 'Sacrifice flies visiting team', 'Division winner home team',
                       'League winner home team', 'World series winner home team', 'Division winner visiting team',
                       'League winner visiting team', 'World series winner visiting team',
                       'Batters hit by pitch home team', 'Batters hit by pitch visiting team',
                       'Earned runs allowed home team', 'Earned runs allowed visiting team', 'Errors home team',
                       'Errors visiting team', 'Wins home team', 'Losses home team', 'Wins visiting team',
                       'Losses visiting team', 'Shutouts home team', 'Saves home team', 'Shutouts visiting team',
                       'Saves visiting team', 'Strikeouts allowed home team', 'Double plays home team',
                       'Strikeouts allowed visiting team', 'Double plays visiting team', 'Home league AL',
                       'division C home team', 'division E home team', 'division W home team', 'Visiting league AL',
                       'division C visiting team', 'division E visiting team', 'division W visiting team',
                       'Rank home team', 'Rank visiting team', 'Hits by batters home team',
                       'Hits by batters visiting team', 'Homeruns home team', 'Homeruns visiting team',
                       'At bats visiting team', 'At bats home team']
    df_dropped_teams = df_teams.drop(columns=columns_to_drop)
    print(df_dropped_teams)
    print(df_dropped_teams.columns.values.tolist())
    export_folder = Path("./Refiltered")
    df_dropped_teams.to_csv(export_folder / '_mlb_refiltered_Teams.csv', index=False)


filter_remerged_teams()
