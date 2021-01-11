import pandas as pd
import numpy as np
import random as rd
from pathlib import Path
from datetime import datetime as dt
from tensorflow.keras.models import load_model

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
        b = 0
        for string in sample:
            b = b+ord(string[0])
            identifier = identifier + string2int(string)
        b = chr(b//len(sample))+str(len(sample)).zfill(2)+'D'
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
def appendModels(log):
    models = []
    for model in (path/'Models').glob('*.h5'):
        models.append(model.name)
    models = pd.DataFrame({'model':models})
    models['identifier'] = models['model'].str.replace('.h5','')
    return pd.merge(log, models, how='left', on='identifier')
def renameModels(log):
    log = log[['identifier','model']].dropna()
    log = log[log['identifier']!=log['model'].str.replace('.h5','')]
    rename = log['identifier'].tolist()
    old = log['model'].tolist()
    for index, model in enumerate(old):
        load_model(path/'Models'/model).save(path/'New Models'/(rename[index]+'.h5'))


path = Path(__file__).parent.absolute()/'Deep Training'
pre = loadLog('None_predictors_log.csv')
#pre = appendModels(pre)
par = loadLog('None_parameter_log.csv')
pre = changeIdentifier(pre)
pre = dropDuplicates(pre)
#renameModels(pre)
par = changeIdentifier(par)
pre.to_csv(path/'Logs'/'None_predictors_log.csv', index=False)
par.to_csv(path/'Logs'/'None_parameter_log.csv', index=False)