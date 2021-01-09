import pandas as pd
import numpy as np
from pathlib import Path

def load(path, dt=False, stats=False):
    print("loading data from",path)
    dataFrames = {}
    dataFrames['gameLogs']  = pd.read_csv(path/'GameLogs.csv', index_col=False)
    if dt:
        dataFrames['gameLogs']['Date'] = pd.to_datetime(dataFrames['gameLogs']['Date'])
    dataFrames['people']    = pd.read_csv(path/'People.csv', index_col=False)
    dataFrames['teams']     = pd.read_csv(path/'Teams.csv', index_col=False)
    dataFrames['managers']  = pd.read_csv(path/'Managers.csv', index_col=False)
    dataFrames['fieldings'] = pd.read_csv(path/'Fielding.csv', index_col=False)
    dataFrames['pitchings'] = pd.read_csv(path/'Pitching.csv', index_col=False)
    dataFrames['battings']  = pd.read_csv(path/'Batting.csv', index_col=False)
    if stats:
        dataFrames['stats']  = pd.read_csv(path/'Stats.csv', index_col=False)
    print("data loaded")
    return dataFrames

def save(path, dataFrames, stats=False):
    print("Saving data to",path)
    dataFrames['gameLogs'].to_csv(path/'GameLogs.csv', index = False)
    dataFrames['people'].to_csv(path/'People.csv', index = False)
    dataFrames['teams'].to_csv(path/'Teams.csv', index = False)
    dataFrames['managers'].to_csv(path/'Managers.csv', index = False)
    dataFrames['fieldings'].to_csv(path/'Fielding.csv', index = False)
    dataFrames['pitchings'].to_csv(path/'Pitching.csv', index = False)
    dataFrames['battings'].to_csv(path/'Batting.csv', index = False)
    if stats:
        dataFrames['stats'].to_csv(path/'Stats.csv', index = False)
    print("Data saved")

def filter(path, saveState=True):
    def filterFrame(frame, columns, renames=None):
        frame = frame[columns] 
        if(renames!=None):
            frame = frame.rename(columns=renames)
        return frame.reset_index(drop=True)

    def filterGameLogs(gameLogs, people):
        gameLogs['Date'] = pd.to_datetime(gameLogs['Date'], format="%Y%m%d")
        gameLogs['Visiting league AL'] = gameLogs['Visiting league']=="AL"
        gameLogs['Home league AL']     = gameLogs['Home league']=="AL"
        gameLogs = gameLogs[gameLogs['Forfeit information'].isna()]
        gameLogs = gameLogs[gameLogs['Protest information'].isna()]
        generalColumns = [
            'Date','Visiting: Team','Visiting league AL','Home: Team','Home league AL','Visiting: Score','Home: Score']
        visitingStatsColumns = [
            'Visiting at-bats','Visiting hits','Visiting doubles','Visiting triples','Visiting homeruns','Visiting RBI','Visiting sacrifice hits','Visiting sacrifice flies',
            'Visiting hit-by-pitch','Visiting walks','Visiting intentional walks','Visiting strikeouts','Visiting stolen bases','Visiting caught stealing','Visiting grounded into double plays',
            'Visiting left on base','Visiting pitchers used','Visiting individual earned runs','Visiting team earned runs','Visiting wild pitches',
            'Visiting balks','Visiting putouts','Visiting assists','Visiting errors','Visiting passed balls','Visiting double plays','Visiting triple plays']
        homeStatsColumns = [
            'Home at-bats','Home hits','Home doubles','Home triples','Home homeruns','Home RBI','Home sacrifice hits','Home sacrifice flies',
            'Home hit-by-pitch','Home walks','Home intentional walks','Home strikeouts','Home stolen bases','Home caught stealing','Home grounded into double plays',
            'Home left on base','Home pitchers used','Home individual earned runs','Home team earned runs','Home wild pitches',
            'Home balks','Home putouts','Home assists','Home errors','Home passed balls','Home double plays','Home triple plays']
        visitingIDColumns = [
            'Visiting team manager ID','Visiting starting pitcher ID',
            'Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID',
            'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID',
            'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID']
        homeIDColumns = [
            'Home team manager ID','Home starting pitcher ID',
            'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID',
            'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID',
            'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID']
        identifier = people[['playerID','retroID']].drop_duplicates(subset=['retroID']).dropna()
        for column in visitingIDColumns+homeIDColumns:
            merged           = pd.merge(gameLogs[column], identifier, left_on=column, right_on='retroID', how="left")
            gameLogs[column] = merged['playerID']
        gameLogs = filterFrame(gameLogs, generalColumns+visitingStatsColumns+homeStatsColumns+visitingIDColumns+homeIDColumns)
        gameLogs = gameLogs.dropna(subset=generalColumns)
        for column in visitingStatsColumns+homeStatsColumns:
            gameLogs = gameLogs[(gameLogs[column]>=0) | (gameLogs[column].isna())]
        return gameLogs.reset_index(drop=True)

    def filterPeople(people):
        people['yearID'] = people['birthYear']
        people['weight'] = 0.453592*people['weight']
        people['height'] = 0.0254*people['height']
        people['bats right'] = (people['bats']=="R") | (people['bats']=="B")
        people['bats left'] = (people['bats']=="L") | (people['bats']=="B")
        people['throws right'] = people['throws']=="R"
        people = filterFrame(people, ['yearID','playerID','weight','height','bats right', 'bats left', 'throws right'])
        return people.reset_index(drop=True)
        
    def filterTeams(teams):
        teams = filterFrame(teams,
            ['yearID','teamIDretro','divID','Rank','G','W','L','DivWin','LgWin','WSWin','R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','SHO','SV','HA','HRA','BBA','SOA','E','DP','FP'], 
            {"teamIDretro":"teamID","divID":"Division","G":"Games","W":"Wins","L":"Losses","DivWin":"Division winner","LgWin":"League winner","WSWin":"World series winner","R":"Runs scored","AB":"At bats"
            ,"H":"Hits by batters","2B":"Doubles","3B":"Triples","HR":"Homeruns","BB":"Walks","SO":"Strikeouts","SB":"Stolen bases","CS":"Cought stealing","HBP":"Batters hit by pitch"
            ,"SF":"Sacrifice flies","RA":"Opponents runs scored","ER":"Earned runs allowed","ERA":"Earned runs average","SHO":"Shutouts","SV":"Saves","HA":"Hits allowed"
            ,"HRA":"Homeruns allowed","BBA":"Walks allowed","SOA":"Strikeouts allowed","E":"Errors","DP":"Double plays","FP":"Fielding percentage"})
        teams['division C'] = (teams['Division']=="C")
        teams['division E'] = (teams['Division']=="E")
        teams['division W'] = (teams['Division']=="W")
        teams = teams.drop(columns=['Division'])
        teams['Division winner']    = (teams['Division winner']=='Y')
        teams['League winner']      = (teams['League winner']=='Y')
        teams['World series winner']= (teams['World series winner']=='Y')
        return teams.reset_index(drop=True)

    print("start filtering")
    dataFrames = load(path/'Input')
    print("filter gameLogs")
    dataFrames['gameLogs']  = filterGameLogs(dataFrames['gameLogs'], dataFrames['people'])
    print("filter people")
    dataFrames['people']    = filterPeople(dataFrames['people'])
    print("filter teams")
    dataFrames['teams']     = filterTeams(dataFrames['teams'])    
    print("filter managers")
    dataFrames['managers']  = filterFrame(dataFrames['managers'], 
        ['yearID','playerID','G','W','L'], 
        {"G":"Games","W":"Wins","L":"Losses"})
    print("filter fieldings")
    dataFrames['fieldings'] = filterFrame(dataFrames['fieldings'], 
        ['yearID','playerID','PO','A','E','DP','PB','WP','SB','CS'], 
        {"PO":"Putouts","A":"Assists","E":"Error","DP":"Double plays","PB":"Passed Balls","WP":"Wild Pitches","SB":"Opponent Stolen Bases","CS":"Opponents Caught Stealing"})
    print("filter pitchings")
    dataFrames['pitchings'] = filterFrame(dataFrames['pitchings'], 
        ['yearID','playerID','W','L','G','H','ER','HR','BB','SO','BAOpp','ERA','IBB','WP','HBP','BK','BFP','R','SH','SF','GIDP','SV','SHO'], 
        {"G":"Games","W":"Wins","L":"Losses","H":"Hits","ER":"Earned Runs","HR":"Homeruns","BB":"Walks","SO":"Strikeouts","BAOpp":"Opponent batting average","ERA":"ERA"
        ,"IBB":"Intentional walks","WP":"Wild pitches","HBP":"Batters hit by pitch","BK":"Balks","BFP":"Batters faced","R":"Runs allowed","SH":"Batters sacrifices"
        ,"SF":"Batters sacrifice flies","GIDP":"Grounded into double plays","SV":"Saves","SHO":"Shutouts"})
    print("filter battings")
    dataFrames['battings'] = filterFrame(dataFrames['battings'], 
        ['yearID','playerID','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP'], 
        {"AB":"At bats","R":"Runs","H":"Hits","2B":"Doubles","3B":"Triples","HR":"Homeruns","RBI":"Runs batted in","SB":"Stolen bases","CS":"Caught stealing"
        ,"BB":"Base on balls","SO":"Strikeouts","IBB":"Intentional walks","HBP":"Hit by pitch","SH":"Sacrifice hits","SF":"Sacrifice flies","GIDP":"Grounded into double plays"})
    print("data filtered")
    if saveState:
        save(path/'Filtered', dataFrames)
    return dataFrames

def replace(path, dataFrames, default="mean", lastKnownState=True, saveState=True, inpurity=0.5):
    def replaceFrame(frame, targets, gameLogs, default, lastKnownState, inpurity):
        #define ID column
        mID = 'playerID'
        for column in frame.columns:
            if column=='teamID':
                mID = 'teamID'
                break
            if column=='playerID':
                break
        #drop inpure columns
        nanFrame = frame.isna().sum().reset_index()
        nanFrame['inpurity'] = nanFrame[0]/frame.index.size
        frame = frame[nanFrame[nanFrame['inpurity']<=inpurity]['index'].tolist()]
        #creating frame containing only usefull data
        onlyFrame = None
        for column in targets:
            temp = gameLogs[['Date',column]]
            temp['yearID'] = temp['Date'].dt.year-1
            temp = temp.rename(columns={column:mID})
            onlyFrame = pd.concat([onlyFrame, temp]).drop(columns=['Date']).drop_duplicates().dropna().reset_index(drop=True)
        #combining duplicates
        aggregators = {}
        for column in frame.drop(columns=['yearID',mID]).columns:
            if (column.find("average")>-1) or (column.find("percentage")>-1):
                aggregators[column] = 'mean'
            elif (column.find("winner")>-1) or (column.find("division")>-1) or (column.find("Rank")>-1):
                aggregators[column] = 'max'
            else:
                aggregators[column] = 'sum'
        temp = frame[frame.duplicated(keep=False, subset=['yearID',mID])]
        temp2 = pd.merge(temp[['yearID',mID]],temp.drop(columns=['yearID',mID]).notna(), left_index=True, right_index=True).groupby(['yearID',mID], as_index=False).sum()
        temp = temp.groupby(['yearID',mID], as_index=False).agg(aggregators)
        for column in temp.columns:
            vec = temp2[column]==0
            col = temp[column]
            col[vec] = None
            temp[column] = col
        frame = frame.drop_duplicates(keep=False, subset=['yearID',mID])
        frame = pd.concat([frame, temp])
        mIDs   = np.array(list(dict.fromkeys(frame[mID].unique().tolist()+onlyFrame[mID].unique().tolist())))
        years = np.array(list(dict.fromkeys(frame['yearID'].unique().tolist()+onlyFrame['yearID'].unique().tolist())))
        fullFrame = pd.DataFrame(np.array(np.meshgrid(years, mIDs)).T.reshape(-1,2), columns=['yearID',mID])
        fullFrame['yearID'] = pd.to_numeric(fullFrame['yearID'])
        fullFrame = pd.merge(fullFrame, frame, on=['yearID',mID], how="left")
        if lastKnownState:
            fullFrame = pd.merge(fullFrame[['yearID',mID]], fullFrame.groupby([mID]).ffill().drop(columns=['yearID']), left_index=True, right_index=True)
        frame = pd.merge(onlyFrame, fullFrame, on=['yearID',mID], how="left")
        nanFrame = frame.isna().sum().reset_index()
        nanFrame['inpurity'] = nanFrame[0]/frame.index.size
        while (not (nanFrame[nanFrame['inpurity']>inpurity/3])['index'].tolist())==False:
            frame = frame[frame[nanFrame.at[nanFrame['inpurity'].idxmax(), 'index']].notna()]
            nanFrame = frame.isna().sum().reset_index()
            nanFrame['inpurity'] = nanFrame[0]/frame.index.size
        if default!=None:
            for column in frame.columns:
                if frame[column].dtype=="bool":
                    frame[column].fillna(False)
                    continue
                if default=="mean":
                    if (frame[column].dtype=="float64") | (frame[column].dtype=="int64"):
                        frame[column] = frame[column].fillna(frame[column].mean())
                elif default=="zero":
                    if (frame[column].dtype=="float64") | (frame[column].dtype=="int64"):
                        frame[column] = frame[column].fillna(0)
        #nanFrame = frame.isna().sum().reset_index()
        #nanFrame['inpurity'] = nanFrame[0]/frame.index.size
        #print(nanFrame[nanFrame['inpurity']>0])
        return frame.dropna().reset_index(drop=True)

    def replaceGameLogs(gameLogs):
        return gameLogs.dropna().reset_index(drop=True)
    
    def replacePeople(people, gameLogs, default):
        columns = ['Visiting team manager ID','Visiting starting pitcher ID'
            ,'Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID'
            ,'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID'
            ,'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID'
            ,'Home team manager ID','Home starting pitcher ID'
            ,'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID'
            ,'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID'
            ,'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID']
        onlyPeople = None
        for column in columns:
            temp = gameLogs[[column]]
            temp = temp.rename(columns={column:'playerID'})
            onlyPeople = pd.concat([onlyPeople, temp]).drop_duplicates().dropna().reset_index(drop=True)
        people = pd.merge(onlyPeople, people, on='playerID', how="left")
        nanFrame = people.isna().sum().reset_index()
        nanFrame['inpurity'] = nanFrame[0]/people.index.size
        while (not (nanFrame[nanFrame['inpurity']>inpurity/4])['index'].tolist())==False:
            people = people[people[nanFrame.at[nanFrame['inpurity'].idxmax(), 'index']].notna()]
            nanFrame = people.isna().sum().reset_index()
            nanFrame['inpurity'] = nanFrame[0]/people.index.size
        if default!=None:
            for column in people.columns:
                if people[column].dtype=="bool":
                    people[column].fillna(False)
                    continue
                if default=="mean":
                    if (people[column].dtype=="float64") | (people[column].dtype=="int64"):
                        people[column] = people[column].fillna(people[column].mean())
                elif default=="zero":
                    if (people[column].dtype=="float64") | (people[column].dtype=="int64"):
                        people[column] = people[column].fillna(0)
        return people.dropna().reset_index(drop=True)

    print("start handeling NAs")
    print("handeling NA in gameLogs")
    dataFrames['gameLogs']  = replaceGameLogs(dataFrames['gameLogs'])
    print("handeling NA in people")
    dataFrames['people']    = replacePeople(dataFrames['people'], dataFrames['gameLogs'], default)
    print("handeling NA in teams")
    dataFrames['teams']     = replaceFrame(dataFrames['teams'],
        ['Home: Team', 'Visiting: Team']
        , dataFrames['gameLogs'], default, lastKnownState, inpurity)
    print("handeling NA in managers")
    dataFrames['managers']  = replaceFrame(dataFrames['managers'],
        ['Home team manager ID', 'Visiting team manager ID']
        , dataFrames['gameLogs'], default, lastKnownState, inpurity)
    print("handeling NA in fieldings")
    dataFrames['fieldings'] = replaceFrame(dataFrames['fieldings'],
        ['Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID'
        ,'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID'
        ,'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID'
        ,'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID'
        ,'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID'
        ,'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID']
        , dataFrames['gameLogs'], default, lastKnownState, inpurity)
    print("handeling NA in pitchings")
    dataFrames['pitchings'] = replaceFrame(dataFrames['pitchings'],
        ['Home starting pitcher ID', 'Visiting starting pitcher ID']
        , dataFrames['gameLogs'], default, lastKnownState, inpurity)
    print("handeling NA in battings")
    dataFrames['battings']  = replaceFrame(dataFrames['battings'],
        ['Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID'
        ,'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID'
        ,'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID'
        ,'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID'
        ,'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID'
        ,'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID']
        , dataFrames['gameLogs'], default, lastKnownState, inpurity)
    print("NAs handeled")
    if saveState:
        save(path/'Replaced', dataFrames)
    return dataFrames

def asPerformance(path, dataFrames, saveState=True):
    def asPerformanceGameLogs(gameLogs):
        gameLogs['Row'] = range(0,gameLogs.index.size)
        gameLogs['Visiting: Win'] = gameLogs['Visiting: Score']>gameLogs['Home: Score']
        gameLogs['Home: Win'] = gameLogs['Visiting: Score']<gameLogs['Home: Score']
        gameLogs['Visiting: Fielding performance'] = (0
            +1.5*gameLogs['Visiting putouts']
            +1.25*gameLogs['Visiting assists']
            +2.25*gameLogs['Visiting double plays']
            -2*gameLogs['Visiting errors']
            #-0.75*gameLogs['Visiting passed balls']
            #-0.75*gameLogs['Visiting wild pitches']
            #-1.5*gameLogs['Home stolen bases']
            #+3.5*gameLogs['Visiting caught stealing']
            )
        gameLogs['Home: Fielding performance'] = (0
            +1.5*gameLogs['Home putouts']
            +1.25*gameLogs['Home assists']
            +2.25*gameLogs['Home double plays']
            -2*gameLogs['Home errors']
            #-0.75*gameLogs['Home passed balls']
            #-0.75*gameLogs['Home wild pitches']
            #-1.5*gameLogs['Visiting stolen bases']
            #+3.5*gameLogs['Home caught stealing']
            )
        gameLogs['Visiting: Pitching performance'] = (0
            -1*gameLogs['Home hits']
            -2*gameLogs['Home team earned runs']
            -3*gameLogs['Home homeruns']
            -1*gameLogs['Home walks']
            +5*gameLogs['Visiting strikeouts']
            +2*gameLogs['Visiting intentional walks']
            -0.5*gameLogs['Visiting wild pitches']
            -0.25*gameLogs['Home hit-by-pitch']
            -0.25*gameLogs['Visiting balks']
            -2*gameLogs['Home: Score']
            -0.75*gameLogs['Home sacrifice hits']
            -0.75*gameLogs['Home sacrifice flies']
            +3*gameLogs['Visiting grounded into double plays'])
        gameLogs['Home: Pitching performance'] = (0
            -1*gameLogs['Visiting hits']
            -2*gameLogs['Visiting team earned runs']
            -3*gameLogs['Visiting homeruns']
            -1*gameLogs['Visiting walks']
            +5*gameLogs['Home strikeouts']
            +2*gameLogs['Home intentional walks']
            -0.5*gameLogs['Home wild pitches']
            -0.25*gameLogs['Visiting hit-by-pitch']
            -0.25*gameLogs['Home balks']
            -2*gameLogs['Visiting: Score']
            -0.75*gameLogs['Visiting sacrifice hits']
            -0.75*gameLogs['Visiting sacrifice flies']
            +3*gameLogs['Home grounded into double plays'])
        gameLogs['Visiting: Batting performance'] = (0
            +0.5*gameLogs['Visiting at-bats']
            +2*gameLogs['Visiting: Score']
            +1*gameLogs['Visiting hits']
            +2*gameLogs['Visiting doubles']
            +3*gameLogs['Visiting triples']
            +3*gameLogs['Visiting homeruns']
            +0.5*gameLogs['Visiting RBI']
            +1.25*gameLogs['Visiting stolen bases']
            -1*gameLogs['Home caught stealing']
            +0.25*gameLogs['Visiting walks']
            -2*gameLogs['Home strikeouts']
            +0.75*gameLogs['Visiting intentional walks']
            +0.25*gameLogs['Visiting hit-by-pitch']
            +0.75*gameLogs['Visiting sacrifice hits']
            +0.75*gameLogs['Visiting sacrifice flies']
            -3*gameLogs['Home grounded into double plays'])
        gameLogs['Home: Batting performance'] = (0
            +0.5*gameLogs['Home at-bats']
            +2*gameLogs['Home: Score']
            +1*gameLogs['Home hits']
            +2*gameLogs['Home doubles']
            +3*gameLogs['Home triples']
            +3*gameLogs['Home homeruns']
            +0.5*gameLogs['Home RBI']
            +1.25*gameLogs['Home stolen bases']
            -1*gameLogs['Visiting caught stealing']
            +0.25*gameLogs['Home walks']
            -2*gameLogs['Visiting strikeouts']
            +0.75*gameLogs['Home intentional walks']
            +0.25*gameLogs['Home hit-by-pitch']
            +0.75*gameLogs['Home sacrifice hits']
            +0.75*gameLogs['Home sacrifice flies']
            -3*gameLogs['Visiting grounded into double plays'])
        gameLogs['Visiting: Pythagorean expectation'] = (
            gameLogs['Visiting: Score']**1.83)/(gameLogs['Visiting: Score']**1.83+gameLogs['Home: Score']**1.83)
        gameLogs['Home: Pythagorean expectation'] = (
            gameLogs['Home: Score']**1.83)/(gameLogs['Home: Score']**1.83+gameLogs['Visiting: Score']**1.83)
        #gameLogs['Visiting: BABIP'] = (
        #    (gameLogs['Visiting hits']-gameLogs['Visiting homeruns'])/(gameLogs['Visiting at-bats']-gameLogs['Visiting strikeouts']-gameLogs['Visiting homeruns']+gameLogs['Visiting sacrifice flies']))
        #gameLogs['Home: BABIP'] = (
        #    (gameLogs['Home hits']-gameLogs['Home homeruns'])/(gameLogs['Home at-bats']-gameLogs['Home strikeouts']-gameLogs['Home homeruns']+gameLogs['Home sacrifice flies']))
        gameLogs['League Diffrence'] = gameLogs['Home league AL'].astype('int32')-gameLogs['Visiting league AL'].astype('int32')
        return gameLogs[['Row','Date','Visiting: Team','Home: Team','Visiting: Score','Home: Score','Visiting: Win','Home: Win','League Diffrence',
            'Visiting: Fielding performance','Home: Fielding performance','Visiting: Pitching performance','Home: Pitching performance',
            'Visiting: Batting performance','Home: Batting performance','Visiting: Pythagorean expectation','Home: Pythagorean expectation',
            #'Visiting: BABIP','Home: BABIP',
            'Visiting team manager ID','Visiting starting pitcher ID',
            'Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID',
            'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID',
            'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID',
            'Home team manager ID','Home starting pitcher ID',
            'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID',
            'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID',
            'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID']]

    def asPerformancePeople(people):
        people['BMI'] = people['weight']/(people['height']**2)
        return people[['yearID','playerID','BMI','bats right','bats left','throws right']]

    def asPerformanceTeams(teams):
        teams['Win rate'] = teams['Wins']/teams['Games']
        teams['Season Performance'] = teams[['Division winner','League winner','World series winner']].mean(axis=1)+1/teams['Rank']
        teams['Pythagorean expectation'] = (teams['Runs scored']**1.83)/(teams['Runs scored']**1.83+teams['Opponents runs scored']**1.83)
        #teams['BABIP'] = ((teams['Hits by batters']-teams['Homeruns'])/
        #    (teams['At bats']-teams['Strikeouts']-teams['Homeruns']+teams['Sacrifice flies']))
        #return teams[['yearID','teamID','Win rate','Season Performance','Pythagorean expectation','BABIP']]
        return teams[['yearID','teamID','Win rate','Season Performance','Pythagorean expectation']]

    def asPerformanceManagers(managers):
        managers['Win rate'] = managers['Wins']/managers['Games']
        return managers[['yearID','playerID','Win rate']]

    def asPerformanceFieldings(fieldings):
        fieldings['Fielding performance'] = (0
            +1.5*fieldings['Putouts']
            +1.25*fieldings['Assists']
            +2.25*fieldings['Double plays']
            -2*fieldings['Error']
            #-0.75*fieldings['Passed Balls']
            #-0.75*fieldings['Wild Pitches']
            #-1.5*fieldings['Opponent Stolen Bases']
            #+3.5*fieldings['Opponents Caught Stealing']
            )
        return fieldings[['yearID','playerID','Fielding performance']]
    
    def asPerformancePitchings(pitchings):
        pitchings['Pitching performance'] = (0
            -1*pitchings['Hits']
            -2*pitchings['Earned Runs']
            -3*pitchings['Homeruns']
            -1*pitchings['Walks']
            +5*pitchings['Strikeouts']
            +2*pitchings['Intentional walks']
            -0.5*pitchings['Wild pitches']
            -0.25*pitchings['Batters hit by pitch']
            -0.25*pitchings['Balks']
            -2*pitchings['Runs allowed']
            -0.75*pitchings['Batters sacrifices']
            -0.75*pitchings['Batters sacrifice flies']
            +3*pitchings['Grounded into double plays'])
        pitchings['Strikeouts per walk'] = pitchings['Strikeouts']/pitchings['Walks'].replace(0,1)
        pitchings['Win rate'] = pitchings['Wins']/pitchings['Games']
        pitchings['Homeruns per game'] = pitchings['Homeruns']/pitchings['Games']
        pitchings['Shutouts per game'] = pitchings['Shutouts']/pitchings['Games']
        pitchings['Saves per game'] = pitchings['Saves']/pitchings['Games']
        return pitchings[['yearID','playerID','Win rate','Pitching performance','Strikeouts per walk','Homeruns per game','Shutouts per game','Saves per game','ERA']]

    def asPerformanceBattings(battings):
        battings['Batting performance'] = (0
            +0.5*battings['At bats']
            +2*battings['Runs']
            +1*battings['Hits']
            +2*battings['Doubles']
            +3*battings['Triples']
            +3*battings['Homeruns']
            +0.5*battings['Runs batted in']
            +1.25*battings['Stolen bases']
            -1*battings['Caught stealing']
            +0.25*battings['Base on balls']
            -2*battings['Strikeouts']
            +0.75*battings['Intentional walks']
            +0.25*battings['Hit by pitch']
            +0.75*battings['Sacrifice hits']
            +0.75*battings['Sacrifice flies']
            -3*battings['Grounded into double plays'])
        return battings[['yearID','playerID','Batting performance']]
    
    print("creating performances")
    print("evaluating performance in gameLogs")
    dataFrames['gameLogs'] = asPerformanceGameLogs(dataFrames['gameLogs'])
    print("evaluating performance in people")
    dataFrames['people'] = asPerformancePeople(dataFrames['people'])
    print("evaluating performance in teams")
    dataFrames['teams'] = asPerformanceTeams(dataFrames['teams'])
    print("evaluating performance in managers")
    dataFrames['managers'] = asPerformanceManagers(dataFrames['managers'])
    print("evaluating performance in fieldings")
    dataFrames['fieldings'] = asPerformanceFieldings(dataFrames['fieldings'])
    print("evaluating performance in pitchings")
    dataFrames['pitchings'] = asPerformancePitchings(dataFrames['pitchings'])
    print("evaluating performance in battings")
    dataFrames['battings'] = asPerformanceBattings(dataFrames['battings'])
    print("performances created")

    if saveState:
        save(path/'Performance', dataFrames)
    return dataFrames
    
def merge(path, dataFrames, saveState=True):
    def mergeFrame(frame, gameLogs, visitingColumns, homeColumns):
        def mergeColumns(columns, gameLogs, frameColumn):
            temp = gameLogs[columns]
            temp['yearID'] = gameLogs['Date'].dt.year-1
            for column in columns:
                temp = pd.merge(temp, frame[['playerID', 'yearID', frameColumn]], left_on=[column,'yearID'], right_on=['playerID','yearID'], how="left").drop(columns=['playerID',column])
            return temp.drop(columns=['yearID']).mean(axis=1)

        merged = gameLogs[['Row']]
        for frameColumn in frame.drop(columns=['playerID','yearID']).columns:
            merged['Visiting: Average '+frameColumn] = mergeColumns(visitingColumns, gameLogs, frameColumn)
            merged['Home: Average '+frameColumn] = mergeColumns(homeColumns, gameLogs, frameColumn)
        return merged

    def mergePeople(people, gameLogs):
        def getAges(yearIDs, teamLogs):
            teamLogs = gameLogs[teamLogs]
            teamLogs['year'] = gameLogs['Date'].dt.year-1
            for column in teamLogs.drop(columns=['year']):
                teamLogs = pd.merge(teamLogs, yearIDs, left_on=column ,right_on='playerID').drop(columns=['playerID'])
            return teamLogs['year']-teamLogs.drop(columns=['year']).mean(axis=1)
        def getSide(sideIDs, teamLogs):
            teamLogs = gameLogs[teamLogs]
            merged = pd.DataFrame()
            for sideColumn in sideIDs.drop(columns=['playerID']):
                temp = teamLogs
                for column in teamLogs.columns:
                    temp = pd.merge(temp, sideIDs[['playerID', sideColumn]], left_on=column ,right_on='playerID').drop(columns=['playerID'])
                merged[sideColumn] = temp.sum(axis=1)
            merged['Batting side'] = merged['bats right']-merged['bats left']
            merged['Throwing side'] = merged['throws right']-(10-merged['throws right'])
            return merged[['Batting side','Throwing side']]
        def getBMI(BMIids, teamLogs):
            teamLogs = gameLogs[teamLogs]
            for column in teamLogs.columns:
                teamLogs = pd.merge(teamLogs, BMIids, left_on=column, right_on='playerID').drop(columns=['playerID'])
            return teamLogs.mean(axis=1)
        visiting = [
            'Visiting starting pitcher ID',
            'Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID',
            'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID',
            'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID']
        home = [
            'Home starting pitcher ID',
            'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID',
            'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID',
            'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID']
        merged = gameLogs[['Row']]
        merged['Visiting: Average age'] = getAges(people[['playerID', 'yearID']], visiting)
        sides = getSide(people[['bats right','bats left','throws right','playerID']], visiting)
        merged['Visiting: Batting side'] = sides['Batting side']
        merged['Visiting: Throwing side'] = sides['Throwing side']
        merged['Visiting: Average BMI'] = getBMI(people[['BMI','playerID']], visiting)
        merged['Home: Average age'] = getAges(people[['playerID', 'yearID']], home)
        sides = getSide(people[['bats right','bats left','throws right','playerID']], home)
        merged['Home: Batting side'] = sides['Batting side']
        merged['Home: Throwing side'] = sides['Throwing side']
        merged['Home: Average BMI'] = getBMI(people[['BMI','playerID']], home)
        return merged

    def mergeTeams(teams, gameLogs):
        teamLogs = gameLogs[['Visiting: Team','Home: Team']]
        teamLogs['yearID'] = gameLogs['Date'].dt.year-1
        merged = gameLogs[['Row']]
        for teamColumn in teams.drop(columns=['teamID','yearID']):
            merged['Visiting: Team - '+teamColumn] = pd.merge(teamLogs, teams[['teamID','yearID',teamColumn]], left_on=['yearID','Visiting: Team'], right_on=['yearID', 'teamID'], how="left")[teamColumn]
            merged['Home: Team - '+teamColumn] = pd.merge(teamLogs, teams[['teamID','yearID',teamColumn]], left_on=['yearID','Home: Team'], right_on=['yearID', 'teamID'], how="left")[teamColumn]
        return merged

    def mergePitchings(pitchings, gameLogs):
        teamLogs = gameLogs[['Visiting starting pitcher ID','Home starting pitcher ID']]
        teamLogs['yearID'] = gameLogs['Date'].dt.year-1
        merged = gameLogs[['Row']]
        for pitchColumn in pitchings.drop(columns=['playerID','yearID']):
            merged['Visiting: Pitcher - '+pitchColumn] = pd.merge(teamLogs, pitchings[['playerID','yearID',pitchColumn]], left_on=['yearID','Visiting starting pitcher ID'], right_on=['yearID', 'playerID'], how="left")[pitchColumn]
            merged['Home: Pitcher - '+pitchColumn] = pd.merge(teamLogs, pitchings[['playerID','yearID',pitchColumn]], left_on=['yearID','Home starting pitcher ID'], right_on=['yearID', 'playerID'], how="left")[pitchColumn]
        return merged

    def mergeManagers(managers, gameLogs):
        teamLogs = gameLogs[['Visiting team manager ID','Home team manager ID']]
        teamLogs['yearID'] = gameLogs['Date'].dt.year-1
        merged = gameLogs[['Row']]
        for managerColumn in managers.drop(columns=['playerID','yearID']):
            merged['Visiting: Manager - '+managerColumn] = pd.merge(teamLogs, managers[['playerID','yearID',managerColumn]], left_on=['yearID','Visiting team manager ID'], right_on=['yearID', 'playerID'], how="left")[managerColumn]
            merged['Home: Manager - '+managerColumn] = pd.merge(teamLogs, managers[['playerID','yearID',managerColumn]], left_on=['yearID','Home team manager ID'], right_on=['yearID', 'playerID'], how="left")[managerColumn]
        return merged
    
    def createRollingStats(gameLogs, toRollVisiting=[], toRollHome=[], vsWindow=5, generalWindow=10, replaceNA="mean"):
        def getTeamPart(teamColumn ,fromFrame):
            teams = []
            for team in fromFrame[teamColumn].unique():
                teams.append(fromFrame[fromFrame[teamColumn]==team])
            return teams

        def getRollingMean(target ,team, window):
            return team.loc[:,target].rolling(window).mean().shift(1)

        def getValues(targetTeamType, teams, columns, vsWindow, generalWindow):
            teamColumns = []
            versusColumns = []
            versus = []
            for team in teams:
                for column in columns:
                    newColumn = column + ' ratio'
                    teamColumns.append(newColumn)
                    team[newColumn] = getRollingMean(column, team, generalWindow)
                tempvs = getTeamPart(targetTeamType, team)
                for vsteam in tempvs:
                    for column in columns:
                        newColumn = column + ' versus ratio'
                        versusColumns.append(newColumn)
                        vsteam[newColumn] = getRollingMean(column, vsteam, vsWindow)
                versus.append(pd.concat(tempvs))
            return pd.merge(pd.concat(teams)[['Row']+list(dict.fromkeys(teamColumns))], pd.concat(versus)[['Row']+list(dict.fromkeys(versusColumns))], on='Row')

        gameLogs = gameLogs[['Row','Visiting: Team','Home: Team','Visiting: Score','Home: Score','Visiting: Win','Home: Win','League Diffrence']+toRollVisiting+toRollHome]
        gameLogs['Home: Odd'] = gameLogs['Home: Score']/(gameLogs['Home: Score']+gameLogs['Visiting: Score']).replace(0,1)
        gameLogs['Visiting: Odd'] = gameLogs['Visiting: Score']/(gameLogs['Visiting: Score']+gameLogs['Home: Score']).replace(0,1)

        toRollVisiting = ['Visiting: Score','Visiting: Win','Visiting: Odd']+toRollVisiting
        toRollHome = ['Home: Score','Home: Win','Home: Odd']+toRollHome

        visiting = gameLogs[['Row','Visiting: Team','Home: Team']+toRollVisiting]
        home = gameLogs[['Row','Visiting: Team','Home: Team']+toRollHome]

        visitings = getValues('Home: Team', getTeamPart('Visiting: Team', visiting), toRollVisiting, vsWindow, generalWindow)
        homes = getValues('Visiting: Team', getTeamPart('Home: Team', home), toRollHome, vsWindow, generalWindow)        
        teams = pd.merge(visitings, homes, on='Row').sort_values('Row').reset_index(drop=True)
        teams = pd.merge(gameLogs[['Row','League Diffrence']], teams, on='Row')
        if replaceNA=="mean":
            for column in teams.columns:
                teams[column] = teams[column].fillna(teams[column].mean())
        elif replaceNA=="drop":
            teams = teams.dropna().reset_index()
        return teams

    print("start merging")
    print("merge people")
    dataFrames['people'] = mergePeople(dataFrames['people'], dataFrames['gameLogs'])
    print("merge teams")
    dataFrames['teams'] = mergeTeams(dataFrames['teams'], dataFrames['gameLogs'])
    print("merge managers")
    dataFrames['managers'] = mergeManagers(dataFrames['managers'], dataFrames['gameLogs'])
    print("merge fieldings")
    dataFrames['fieldings'] = mergeFrame(dataFrames['fieldings'], dataFrames['gameLogs'],[
        'Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID',
        'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID',
        'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID'],[
        'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID',
        'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID',
        'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID'])
    print("merge pitchings")
    dataFrames['pitchings'] = mergePitchings(dataFrames['pitchings'],dataFrames['gameLogs'])
    print("merge battings")
    dataFrames['battings'] = mergeFrame(dataFrames['battings'], dataFrames['gameLogs'],[
        'Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID',
        'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID',
        'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID'],[
        'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID',
        'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID',
        'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID'])
    print("merging complete")
    print("creating rolling stats")
    dataFrames['stats'] = createRollingStats(dataFrames['gameLogs'],
        ['Visiting: Fielding performance','Visiting: Pitching performance','Visiting: Batting performance','Visiting: Pythagorean expectation'#,'Visiting: BABIP'
        ],
        ['Home: Fielding performance','Home: Pitching performance','Home: Batting performance','Home: Pythagorean expectation'#,'Home: BABIP'
        ])
    print("dropping ID columns")
    dataFrames['gameLogs'] = dataFrames['gameLogs'].drop(columns=[
        'Visiting team manager ID','Visiting starting pitcher ID','League Diffrence',
        'Visiting starting player 1 ID','Visiting starting player 2 ID','Visiting starting player 3 ID',
        'Visiting starting player 4 ID','Visiting starting player 5 ID','Visiting starting player 6 ID',
        'Visiting starting player 7 ID','Visiting starting player 8 ID','Visiting starting player 9 ID',
        'Home team manager ID','Home starting pitcher ID',
        'Home starting player 1 ID','Home starting player 2 ID','Home starting player 3 ID',
        'Home starting player 4 ID','Home starting player 5 ID','Home starting player 6 ID',
        'Home starting player 7 ID','Home starting player 8 ID','Home starting player 9 ID'])
    
    if saveState:
        save(path/'Merged', dataFrames, stats=True)
    return dataFrames

def createLearningData(data, path, excludes=[], operator="-", dropRowIndex=True):
    def createDiffrence(dataFrame, operator, outcast=pd.DataFrame()):
        homes = []
        visitings = {}
        temp = outcast
        for column in dataFrame.drop(columns=temp.columns).columns:
            if column.find("Visiting")>-1:
                visitings[column.replace("Visiting: ","")] = column.replace("Visiting: ","")
            elif column.find("Home")>-1:
                homes.append(column.replace("Home: ",""))
            else:
                temp[column] = dataFrame[column]
        for homeCol in homes:
            visitingCol = visitings.pop(homeCol)
            if operator=="-":
                temp[homeCol+" diffrence"] = dataFrame['Home: '+homeCol]-dataFrame['Visiting: '+visitingCol]
            elif operator=="/":
                temp[homeCol+" diffrence"] = dataFrame['Home: '+homeCol]/dataFrame['Visiting: '+visitingCol].replace(0,1)
            elif operator=="/sum":
                temp[homeCol+" diffrence"] = dataFrame['Home: '+homeCol]/(dataFrame['Home: '+homeCol]+dataFrame['Visiting: '+visitingCol]).replace(0,1)
            else:
                temp[homeCol+" diffrence"] = dataFrame['Home: '+homeCol]>dataFrame['Visiting: '+visitingCol]
        return temp
    for exc in excludes:
        data.pop(exc)
    path = path/'Learning'
    gameLogs = data.pop('gameLogs')
    predictors = gameLogs[['Row']]
    for frame in data:
        predictors = pd.merge(predictors, data[frame], on='Row', how="left")
    predictors = predictors.dropna()
    targets = pd.merge(predictors[['Row']], gameLogs, on='Row', how="left")
    if operator!=None:
        predictors = createDiffrence(predictors, operator)
        targets = createDiffrence(targets, operator, targets[['Date','Visiting: Team','Home: Team','Visiting: Score','Home: Score','Visiting: Win','Home: Win']])
    else:
        operator="None"
    print("Saving data to",path)
    if dropRowIndex:
        predictors.drop(columns=['Row']).to_csv(path/(operator+"_"+'Predictors.csv'), index = False)
        targets.drop(columns=['Row']).to_csv(path/(operator+"_"+'Targets.csv'), index = False)
    else:
        predictors.to_csv(path/'Predictors.csv', index = False)
        targets.to_csv(path/'Targets.csv', index = False)
    print("Data saved")

def createView(columns, data):
    view = data['gameLogs'][['Row']]
    for frameName in data:
        frame = data[frameName]
        for column in frame.drop(columns=['Row']).columns:
            if column in columns:
                view = pd.merge(frame[[column]+['Row']], view, on='Row')
    return view

path = Path(__file__).parent.absolute()
print(path)
#data = filter(path)
#data = load(path/'Filtered', dt=True)
#data = replace(path, data, inpurity=0.7)
#data = asPerformance(path, data)
#data = merge(path, data)
#print(data)
data = load(path/'Merged', dt=True, stats=True)
createLearningData(data, path, operator="-")