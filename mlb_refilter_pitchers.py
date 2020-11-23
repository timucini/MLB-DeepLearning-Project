import pandas as pd

def refilterPitchers():
    path = r'/Users/sewerynkozlowski/Desktop/HTW_2_Semester/Analytische Anwendungen/MLB-DeepLearning-Project/Remerged/'
    pitchers = pd.read_csv(r'/Users/sewerynkozlowski/Desktop/HTW_2_Semester/Analytische Anwendungen/MLB-DeepLearning-Project/Remerged/mlb_remerged_Pitchers22.csv', index_col=False)

    df_pitchers = pd.DataFrame(pitchers)
    homePitcherPerformance = []
    visitingPitcherPerformance = []
    # print(df_pitchers)
    for index, row in df_pitchers.iterrows():
        if row['Walks home pitcher'] != 0:
            homeStrikeOutsPerWalks = row['Strikeouts home pitcher']/row['Walks home pitcher']
            gamesWinsRatioHome = row['Games home pitcher']/row['Wins home pitcher']
            homerunsPerGameHome = row['Homeruns home pitcher']/row['Games home pitcher']
            shutoutsPerGameHome = row['Shutouts home pitcher']/row['Games home pitcher']
            savesPerGameHome = row['Saves home pitcher']/row['Games home pitcher']
            eraHome = row['Earned runs average home pitcher']
        if row['Walks visiting pitcher'] != 0:
            visitingStrikeOutsPerWalks = row['Strikeouts visiting pitcher']/row['Walks visiting pitcher']
            gamesWinsRatioVisiting = row['Games visiting pitcher']/row['Wins visiting pitcher']
            homerunsPerGameVisiting = row['Homeruns visiting pitcher']/row['Games visiting pitcher']
            shutoutsPerGameVisiting = row['Shutouts visiting pitcher']/row['Games visiting pitcher']
            savesPerGameVisiting = row['Saves visiting pitcher']/row['Games visiting pitcher']
            eraVisitng = row['Earned runs average visiting pitcher']
        homePitcherPerformance.append((1*homeStrikeOutsPerWalks+1*gamesWinsRatioHome+1*homerunsPerGameHome+1*shutoutsPerGameHome+1.25*savesPerGameHome+2*eraHome)/6)
        visitingPitcherPerformance.append((1*visitingStrikeOutsPerWalks+1*gamesWinsRatioVisiting+1*homerunsPerGameVisiting+1*shutoutsPerGameVisiting+1.25*savesPerGameVisiting+2*eraVisitng)/6)
    # df_pitchers['pitcherPerformance'] = pitcherPerformance
    filtered_pitchers = pd.DataFrame(
        {
            "row": df_pitchers['row'],
            "Home Pitcher Performance": homePitcherPerformance,
            "Visiting Pitcher Performance": visitingPitcherPerformance
        }
    )
    filtered_pitchers.set_index('row')
    filtered_pitchers.to_csv(r'/Users/sewerynkozlowski/Desktop/HTW_2_Semester/Analytische Anwendungen/MLB-DeepLearning-Project/Remerged/refiltered_pitching.csv', index=False)
    return df_pitchers


refilterPitchers()
