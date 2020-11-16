import numpy as np
import pandas as pd

def compressManagers(managers):
    compressed = pd.DataFrame(managers['row'])
    compressed['Home: Manger - Win rate'] = managers['Wins home manager'].fillna(0)/managers['Games home manager'].fillna(managers['Games home manager'].mean())
    compressed['Visiting: Manger - Win rate'] = managers['Wins visiting manager'].fillna(0)/managers['Games visiting manager'].fillna(managers['Games visiting manager'].mean())
    return compressed
    
def compressFielding(fieldings):
    compressed = pd.DataFrame(fieldings['row'])
    colName = ' starting player '
    for t in ['Visiting','Home']:
        temp = pd.DataFrame()
        for i in range(1,10):
            putouts = fieldings[t+colName+str(i)+' Putouts'].fillna(fieldings[t+colName+str(i)+' Putouts'].mean())
            assists = fieldings[t+colName+str(i)+' Assists'].fillna(fieldings[t+colName+str(i)+' Assists'].mean())
            doubles = fieldings[t+colName+str(i)+' Error'].fillna(fieldings[t+colName+str(i)+' Error'].mean())
            errors  = fieldings[t+colName+str(i)+' Double plays'].fillna(fieldings[t+colName+str(i)+' Double plays'].mean())
            passeds = fieldings[t+colName+str(i)+' Passed Balls'].fillna(fieldings[t+colName+str(i)+' Passed Balls'].mean())
            wilds   = fieldings[t+colName+str(i)+' Wild Pitches'].fillna(fieldings[t+colName+str(i)+' Wild Pitches'].mean())
            stolen  = fieldings[t+colName+str(i)+' Opponent Stolen Bases'].fillna(fieldings[t+colName+str(i)+' Opponent Stolen Bases'].mean())
            caught  = fieldings[t+colName+str(i)+' Opponents Caught Stealing'].fillna(fieldings[t+colName+str(i)+' Opponents Caught Stealing'].mean())
            temp[str(i)] = (1.25*putouts+0.75*assists+1.5*doubles-6*errors-0.75*passeds-0.75*wilds-1.5*stolen+1.5*caught)/8
        compressed[t+': Fielding - Average player performance'] = temp.mean(axis=1)
    return compressed

def compressPeople(people):
    compressed = pd.DataFrame(people['row'])
    colName = ' starting player '
    for t in ['Visiting','Home']:
        age = pd.DataFrame()
        bmi = pd.DataFrame()
        batsRight = pd.DataFrame()
        batsLeft = pd.DataFrame()
        throwsRight = pd.DataFrame()
        for i in range(1,11):
            if i==10:
                age[str(i)]         = people[t+' starting pitcher  age'].fillna(people[t+' starting pitcher  age'].mean())
                bmi[str(i)]         = people[t+' starting pitcher weight'].fillna(people[t+' starting pitcher weight'].mean())/(people[t+' starting pitcher height']**2)
                batsRight[str(i)]   = people[t+' starting pitcher bats right'].fillna(people[t+' starting pitcher bats right'].mean())
                batsLeft[str(i)]    = people[t+' starting pitcher bats left'].fillna(people[t+' starting pitcher bats left'].mean())
                throwsRight[str(i)] = people[t+' starting pitcher throws right'].fillna(people[t+' starting pitcher throws right'].mean())
                continue
            age[str(i)]         = people[t+colName+str(i)+'  age'].fillna(people[t+colName+str(i)+'  age'].mean())
            bmi[str(i)]         = people[t+colName+str(i)+' weight'].fillna(people[t+colName+str(i)+' weight'].mean())/(people[t+colName+str(i)+' height']**2)
            batsRight[str(i)]   = people[t+colName+str(i)+' bats right'].fillna(people[t+colName+str(i)+' bats right'].mean())
            batsLeft[str(i)]    = people[t+colName+str(i)+' bats left'].fillna(people[t+colName+str(i)+' bats left'].mean())
            throwsRight[str(i)] = people[t+colName+str(i)+' throws right'].fillna(people[t+colName+str(i)+' throws right'].mean())
        compressed[t+': People - Average player age'] = age.mean(axis=1)
        compressed[t+': People - Average player bmi'] = bmi.mean(axis=1)
        compressed[t+': People - Average player bats right'] = batsRight.sum(axis=1)/(batsRight.sum(axis=1)+batsLeft.sum(axis=1))
        compressed[t+': People - Average player throws right'] = throwsRight.mean(axis=1)
    return compressed

path = r'F:\Dokumente\HTW\2. Semester\Analytische Anwendungen\Projekt'
#stats       = pd.read_csv(path+r'\Remerged\_mlb_game_stats.csv', index_col=False)
people      = pd.read_csv(path+r'\Remerged\_mlb_remerged_People.csv', index_col=False)
#teams       = pd.read_csv(path+r'\Remerged\_mlb_remerged_Teams.csv', index_col=False)
managers    = pd.read_csv(path+r'\Remerged\_mlb_remerged_Managers.csv', index_col=False)
#pitchings   = pd.read_csv(path+r'\Remerged\_mlb_remerged_Pitchers.csv', index_col=False)
#battings    = pd.read_csv(path+r'\Remerged\_mlb_remerged_Batting.csv', index_col=False)
fieldings   = pd.read_csv(path+r'\Remerged\_mlb_remerged_Fielding.csv', index_col=False)

def toFloat16(df):
    for col in df.drop(columns=['row']):
        df[col] = df[col].astype('float16')
    return df

toFloat16(compressPeople(people)).to_csv(path+r'\Compressed\_mlb_compressed_People.csv', index=False)
toFloat16(compressManagers(managers)).to_csv(path+r'\Compressed\_mlb_compressed_Managers.csv', index=False)
toFloat16(compressFielding(fieldings)).to_csv(path+r'\Compressed\_mlb_compressed_Fielding.csv', index=False)