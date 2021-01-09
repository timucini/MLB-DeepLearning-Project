import pandas as pd
import numpy as np
import random as rd
from pathlib import Path
from datetime import datetime as dt

def loadLog(name):
    df = pd.read_csv(path/'Logs'/name, index_col=False)
    if df.empty:
        return df
    df['identifier'] = df['identifier'].astype(str)
    for column in df:
        if df[column].dtype==object:
            if (df[column][0].find("[")>-1 and df[column][0].find("]")>-1):
                df[column] = df[column].str.replace("'","").str.replace(", ",",").str.replace("[","").str.replace("]","").str.split(",")
                if column=='layers' or column=='dropouts':
                    newCol = []
                    for element in df[column].tolist():
                        newElement = []
                        for value in element:
                            newElement.append(int(value))
                        newCol.append(newElement)
                    df[column] = pd.Series(newCol)
    return df
def changeIdentifier(log):
    def getIdentifier(sample):
        def string2int(string):
            value = 0
            for p, char in enumerate(list(string)):
                value = value+(p+1)*31*ord(char)*113*len(string)*271
            return value
        identifier = 0
        for string in sample:
            identifier = identifier + string2int(string)
        b = sample[0][0].upper()+str(len(sample)).zfill(2)+'D'
        return (b+str(identifier).zfill(16-len(b)))[:16]
    print('Old:')
    print(log[['predictors','identifier']])
    log = log.to_dict('records')
    for entry in log:
        entry['identifier'] = getIdentifier(entry['predictors'])
    log = pd.DataFrame(log)
    print('New:')
    print(log[['predictors','identifier']])
    return log
def dropDuplicates(log):
    print('Duplicates:')
    dups = log[log[['identifier']].duplicated(keep=False)].sort_values(by=['identifier','binary_accuracy', 'loss', 'epochs', 'nodes', 'time'], ascending=[True,False, True, True, True, True])
    print(dups)
    print('Worse tries:')
    dups = dups.drop_duplicates(subset=['identifier'], keep='last')
    print(dups)
    return log.drop(dups.index)

path = Path(__file__).parent.absolute()/'Deep Training'
pre = loadLog('None_predictors_log.csv')
par = loadLog('None_parameter_log.csv')
pre = changeIdentifier(pre)
pre = dropDuplicates(pre)
par = changeIdentifier(par)
pre.to_csv(path/'Logs'/'None_predictors_log.csv', index=False)
par.to_csv(path/'Logs'/'None_parameter_log.csv', index=False)