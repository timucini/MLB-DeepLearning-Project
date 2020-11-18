import pandas as pd
import numpy as np
from pathlib import Path


from numpy import nanmean

pd.options.mode.chained_assignment = None  # default='warn'


def filter_remerged_battings():
    data_folder = Path("./Remerged")
    team_data_file = data_folder / "_mlb_remerged_Batting.csv"
    df_battings = pd.read_csv(team_data_file)
    avg_bat = []
    avg_runs = []
    avg_hits = []
    avg_doubles = []
    avg_triples = []
    avg_homeruns = []
    avg_batted = []
    avg_stolen = []
    avg_caught = []
    avg_base = []
    avg_strikeout = []
    avg_pitch = []
    avg_sacrifice = []
    avg_bat_home = []
    avg_runs_home = []
    avg_hits_home = []
    avg_doubles_home = []
    avg_triples_home = []
    avg_homeruns_home = []
    avg_batted_home = []
    avg_stolen_home = []
    avg_caught_home = []
    avg_base_home = []
    avg_strikeout_home = []
    avg_pitch_home = []
    avg_sacrifice_home = []
    for index, row in df_battings.iterrows():
        avg_bat_row = []
        avg_runs_row = []
        avg_hits_row = []
        avg_doubles_row = []
        avg_triples_row = []
        avg_homeruns_row = []
        avg_batted_row = []
        avg_stolen_row = []
        avg_caught_row = []
        avg_base_row = []
        avg_strikeout_row = []
        avg_pitch_row = []
        avg_sacrifice_row = []
        avg_bat_row_home = []
        avg_runs_row_home = []
        avg_hits_row_home = []
        avg_doubles_row_home = []
        avg_triples_row_home = []
        avg_homeruns_row_home = []
        avg_batted_row_home = []
        avg_stolen_row_home = []
        avg_caught_row_home = []
        avg_base_row_home = []
        avg_strikeout_row_home = []
        avg_pitch_row_home = []
        avg_sacrifice_row_home = []
        print(index)
        for x in range(9):
            x += 1
            if row['Visiting starting player ' + str(x) + ' At bats'] != 0:
                #at bats
                avg_bat_player = row['Visiting starting player ' + str(x) + ' At bats']
                avg_bat_row.append(avg_bat_player)
                #runs
                avg_runs_player = row['Visiting starting player ' + str(x) + ' Runs']
                avg_runs_row.append(avg_runs_player)
                #hits
                avg_hit_player = row['Visiting starting player ' + str(x) + ' Hits']
                avg_hits_row.append(avg_hit_player)
                #doubles
                avg_doubles_player = row['Visiting starting player ' + str(x) + ' Doubles']
                avg_doubles_row.append(avg_doubles_player)
                #triples
                avg_triples_player = row['Visiting starting player ' + str(x) + ' Triples']
                avg_triples_row.append(avg_triples_player)
                #homeruns
                avg_homeruns_player = row['Visiting starting player ' + str(x) + ' Homeruns']
                avg_homeruns_row.append(avg_homeruns_player)
                #batted in
                avg_batted_player = row['Visiting starting player ' + str(x) + ' Runs batted in']
                avg_batted_row.append(avg_batted_player)
                #stolen bases
                avg_stolen_player = row['Visiting starting player ' + str(x) + ' Stolen bases']
                avg_stolen_row.append(avg_stolen_player)
                #caught stealing
                avg_caught_player = row['Visiting starting player ' + str(x) + ' Caught stealing']
                avg_caught_row.append(avg_caught_player)
                #base on balls
                avg_base_player = row['Visiting starting player ' + str(x) + ' Base on balls']
                avg_base_row.append(avg_base_player)
                #strikeouts
                avg_strikeout_player = row['Visiting starting player ' + str(x) + ' Strikeouts']
                avg_strikeout_row.append(avg_strikeout_player)
                #hit by pitch
                avg_pitch_player = row['Visiting starting player ' + str(x) + ' Hit by pitch']
                avg_pitch_row.append(avg_pitch_player)
                #scrifice hits
                avg_sacrifice_player = row['Visiting starting player ' + str(x) + ' Sacrifice hits']
                avg_sacrifice_row.append(avg_sacrifice_player)
            if row['Home starting player ' + str(x) + ' At bats'] != 0:
                #at bats
                avg_bat_player_home = row['Home starting player ' + str(x) + ' At bats']
                avg_bat_row_home.append(avg_bat_player_home)
                #runs
                avg_runs_player_home = row['Home starting player ' + str(x) + ' Runs']
                avg_runs_row_home.append(avg_runs_player_home)
                #hits
                avg_hit_player_home = row['Home starting player ' + str(x) + ' Hits']
                avg_hits_row_home.append(avg_hit_player_home)
                #doubles
                avg_doubles_player_home = row['Home starting player ' + str(x) + ' Doubles']
                avg_doubles_row_home.append(avg_doubles_player_home)
                #triples
                avg_triples_player_home = row['Home starting player ' + str(x) + ' Triples']
                avg_triples_row_home.append(avg_triples_player_home)
                #homeruns
                avg_homeruns_player_home = row['Home starting player ' + str(x) + ' Homeruns']
                avg_homeruns_row_home.append(avg_homeruns_player_home)
                #batted in
                avg_batted_player_home = row['Home starting player ' + str(x) + ' Runs batted in']
                avg_batted_row_home.append(avg_batted_player_home)
                #stolen bases
                avg_stolen_player_home = row['Home starting player ' + str(x) + ' Stolen bases']
                avg_stolen_row_home.append(avg_stolen_player_home)
                #caught stealing
                avg_caught_player_home = row['Home starting player ' + str(x) + ' Caught stealing']
                avg_caught_row_home.append(avg_caught_player_home)
                #base on balls
                avg_base_player_home = row['Home starting player ' + str(x) + ' Base on balls']
                avg_base_row_home.append(avg_base_player_home)
                #strikeouts
                avg_strikeout_player_home = row['Home starting player ' + str(x) + ' Strikeouts']
                avg_strikeout_row_home.append(avg_strikeout_player_home)
                #hit by pitch
                avg_pitch_player_home = row['Home starting player ' + str(x) + ' Hit by pitch']
                avg_pitch_row_home.append(avg_pitch_player_home)
                #scrifice hits
                avg_sacrifice_player_home = row['Home starting player ' + str(x) + ' Sacrifice hits']
                avg_sacrifice_row_home.append(avg_sacrifice_player_home)
        avg_bat.append(np.nanmean(avg_bat_row))
        avg_runs.append(np.nanmean(avg_runs_row))
        avg_hits.append(np.nanmean(avg_hits_row))
        avg_doubles.append(np.nanmean(avg_doubles_row))
        avg_triples.append(np.nanmean(avg_triples_row))
        avg_homeruns.append(np.nanmean(avg_homeruns_row))
        avg_batted.append(np.nanmean(avg_batted_row))
        avg_stolen.append(np.nanmean(avg_stolen_row))
        avg_caught.append(np.nanmean(avg_caught_row))
        avg_base.append(np.nanmean(avg_base_row))
        avg_strikeout.append(np.nanmean(avg_strikeout_row))
        avg_pitch.append(np.nanmean(avg_pitch_row))
        avg_sacrifice.append(np.nanmean(avg_sacrifice_row))
        avg_bat_home.append(np.nanmean(avg_bat_row_home))
        avg_runs_home.append(np.nanmean(avg_runs_row_home))
        avg_hits_home.append(np.nanmean(avg_hits_row_home))
        avg_doubles_home.append(np.nanmean(avg_doubles_row_home))
        avg_triples_home.append(np.nanmean(avg_triples_row_home))
        avg_homeruns_home.append(np.nanmean(avg_homeruns_row_home))
        avg_batted_home.append(np.nanmean(avg_batted_row_home))
        avg_stolen_home.append(np.nanmean(avg_stolen_row_home))
        avg_caught_home.append(np.nanmean(avg_caught_row_home))
        avg_base_home.append(np.nanmean(avg_base_row_home))
        avg_strikeout_home.append(np.nanmean(avg_strikeout_row_home))
        avg_pitch_home.append(np.nanmean(avg_pitch_row_home))
        avg_sacrifice_home.append(np.nanmean(avg_sacrifice_row_home))
    filtered_batting = pd.DataFrame(
        {
         "row": df_battings['row'],
         "Home Team avg At bats": avg_bat_home,
         "Home Team avg Runs": avg_runs_home,
         "Home Team avg Hits": avg_hits_home,
         "Home Team avg Doubles": avg_doubles_home,
         "Home Team avg Triples": avg_triples_home,
         "Home Team avg Homeruns": avg_homeruns_home,
         "Home Team avg Runs batted in": avg_batted_home,
         "Home Team avg Stolen Bases": avg_stolen_home,
         "Home Team avg Caught Stealing": avg_caught_home,
         "Home Team avg Base on Balls": avg_base_home,
         "Home Team avg Strikeouts": avg_strikeout_home,
         "Home Team avg Hit by pitch": avg_pitch_home,
         "Home Team avg Sacrifice Hits": avg_sacrifice_home,
         "Visiting Team avg At bats": avg_bat,
         "Visiting Team avg Runs": avg_runs,
         "Visiting Team avg Hits": avg_hits,
         "Visiting Team avg Doubles": avg_doubles,
         "Visiting Team avg Triples": avg_triples,
         "Visiting Team avg Homeruns": avg_homeruns,
         "Visiting Team avg Runs batted in": avg_batted,
         "Visiting Team avg Stolen Bases": avg_stolen,
         "Visiting Team avg Caught Stealing": avg_caught,
         "Visiting Team avg Base on Balls": avg_base,
         "Visiting Team avg Strikeouts": avg_strikeout,
         "Visiting Team avg Hit by pitch": avg_pitch,
         "Visiting Team avg Sacrifice Hits": avg_sacrifice,
         }
    )
    filtered_batting.set_index('row')
    path = Path("./Refiltered")
    filtered_batting.to_csv(path / '_mlb_refiltered_Batting.csv', index=False)


filter_remerged_battings()
