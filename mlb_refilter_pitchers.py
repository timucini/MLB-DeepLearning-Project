import pandas as pd

def refilterPitchers():
    path = r'/Users/sewerynkozlowski/Desktop/HTW_2_Semester/Analytische Anwendungen/MLB-DeepLearning-Project/Remerged/'
    pitchers = pd.read_csv(path + r'_mlb_remerged_Pitchers.csv', index_col=False)

    df_pitchers = pd.DataFrame(pitchers)
    homePitcherPerformance = []
    visitingPitcherPerformance = []
    # print(df_pitchers)
    for index, row in df_pitchers.iterrows():
        if row['Walks home pitcher'] != 0:
            homeStrikeOutsPerWalks = row['Strikeouts home pitcher']/row['Walks home pitcher']
        if row['Walks visiting pitcher'] != 0:
            visitingStrikeOutsPerWalks = row['Strikeouts visiting pitcher']/row['Walks visiting pitcher']
        homePitcherPerformance.append(homeStrikeOutsPerWalks)
        visitingPitcherPerformance.append(visitingStrikeOutsPerWalks)
    # df_pitchers['pitcherPerformance'] = pitcherPerformance
    filtered_batting = pd.DataFrame(
        {
            "row": df_pitchers['row'],
            "Home Pitcher Performance": homePitcherPerformance,
            "Visiting Pitcher Performance": visitingPitcherPerformance
        }
    )
    filtered_batting.set_index('row')
    filtered_batting.to_csv(r'/Users/sewerynkozlowski/Desktop/HTW_2_Semester/Analytische Anwendungen/MLB-DeepLearning-Project/Remerged/refiltered_pitching.csv', index=False)
    return df_pitchers


refilterPitchers()
