import pandas as pd
import numpy as np
import random as rd
import math as mt
from pathlib import Path
from datetime import datetime as dt

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def initGPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
def getData(path, targets, predictors, save=False, load=False, targetsCols=None, predictorsCols=None, centerBy=False, centerSize=False, centerSeed=1337):
    def loadData():
        return (
            (pd.read_csv(path/(predictors+'_training.csv'), index_col=0), pd.read_csv(path/(predictors+'_validation.csv'), index_col=0)),
            (pd.read_csv(path/(targets+'_training.csv'), index_col=0), pd.read_csv(path/(targets+'_validation.csv'), index_col=0)))
    def saveData(dataTuple):
        dataTuple[0][0].to_csv(path/(predictors+'_training.csv'))
        dataTuple[0][1].to_csv(path/(predictors+'_validation.csv'))
        dataTuple[1][0].to_csv(path/(targets+'_training.csv'))
        dataTuple[1][1].to_csv(path/(targets+'_validation.csv'))
    if load:
        return loadData()
    ts = pd.read_csv(path/(targets+'.csv'), index_col=False, usecols=targetsCols)
    ps = pd.read_csv(path/(predictors+'.csv'), index_col=False, usecols=predictorsCols)
    if centerBy:
        if not centerSize:
            centerSize = len(targets)
        rd.seed(centerSeed)
        def flatter(series, size):
            pos = series[series].index.tolist()
            neg = series[~series].index.tolist()
            pos = pos + rd.sample(pos*(mt.ceil(size/len(pos))-1),max((size//2)-len(pos),0))
            neg = neg + rd.sample(pos*(mt.ceil(size/len(neg))-1),max((size//2)-len(neg),0))
            pos = rd.sample(pos, size//2)
            neg = rd.sample(neg, size//2)
            mix = []
            for i in range(len(pos)):
                mix.append(pos[i])
                mix.append(neg[i])
            return mix
        def divideDataByIndex(data, tra, val):
            return data.loc[tra], data.loc[val]
        val = flatter(ts[centerBy], centerSize//20)
        tra = flatter(ts[centerBy].drop(val), centerSize)
        ts = divideDataByIndex(ts, tra, val)
        ps = divideDataByIndex(ps, tra, val)
    if save:
        saveData((ps, ts))
    return ps, ts
def trainingRoutine(trainName, path, predictors, targets, metric, minimise, minDuration, maxDuration, start, stop, startNodes=10, worker=-1, epsilon=8, validationMetrics=[]):
    def getBatchSize(size):
        sizes = []
        for i in range((size//6)+1, 2, -1):
            if ((size % i)) == 0 and ((size//i)>1000):
                sizes.append(size//i)
        return sizes[len(sizes)//2]
    batchSize = getBatchSize(len(targets[0]))
    def row2log(name, row):
        with open(path/'Logs'/(trainName+'_'+name),'a') as logCSV:
            logCSV.write('\n')
            logCSV.write(row)
    def loadLog(name):
        df = pd.read_csv(path/'Logs'/(trainName+'_'+name), index_col=False)
        if df.empty:
            return df
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
    def getBest(frame, identifier, output=[]):
        if not output:
            output = list(frame.columns)
        frame = frame[frame['identifier']==identifier].sort_values(by=[metric, 'loss', 'epochs', 'nodes', 'time'], ascending=[minimise, True, True, True, True])
        return frame[output].to_dict('records')[0]
    def predictorTraining():
        def sample(columns, bias, maxNodes):
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
            tries = []
            def appendTry(tries, row):
                if row[metric]<bias:
                    return tries
                tries.append(row)
                frame = pd.DataFrame(tries)
                frame = frame.sort_values(by=[metric, 'loss', 'epochs', 'nodes', 'time'], ascending=[minimise, True, True, True, True])
                return frame.to_dict('records')[0:epsilon]
            def check(identifier):
                frame = loadLog('predictors_log.csv')
                return frame[frame['identifier']==identifier]
            pool = predictors[0].drop(columns=columns).columns.tolist()
            for column in pool:
                samples = columns+[column]
                identifier = getIdentifier(samples)
                backlog = check(identifier)
                if backlog.empty:
                    samples = (predictors[0][samples], predictors[1][samples])
                    row = parameterTraining(samples, maxNodes, identifier, minDuration, maxDuration, start, stop)
                    row2log('predictors_log.csv', row)
                    row = getBest(loadLog('predictors_log.csv'), identifier)
                    tries = appendTry(tries, row)
                else:
                    backlog = getBest(backlog, identifier)
                    tries = appendTry(tries, backlog)
            return tries
        def trace(line, bias, nodes):
            print('Worker:    ', worker)
            print('Predictors:', line)
            print('Bias:      ', bias)
            print('Nodes:     ', nodes, '\n')
            trial = sample(line, bias, nodes)
            if not trial:
                print('Dead end @:', line, '\n')
                return
            for entry in trial:
                preds = entry['predictors']
                maxNodes = max([(entry['nodes']/len(preds))*(len(preds)+1), len(preds)*5])
                trace(preds, entry[metric], round(maxNodes))
        if worker<0:
            trace([], 0.5, startNodes)
        else:
            work = sample([], 0.5, startNodes)[worker]
            trace(work['predictors'], work[metric], startNodes)
        print('Worker', worker, 'finished!')
    def parameterTraining(predictors, maxNodes, identifier, minDuration, maxDuration, start, stop):
        header = 'timestamp,predictors,identifier,optimizer,layers,activations,dropouts,dimensions,length,nodes,loss,'+metric+',time,epochs'
        if worker<0:
            bufferName = 'buffer_log.csv'
        else:
            bufferName = str(worker)+'_buffer_log.csv'
        with open(path/'Logs'/(trainName+'_'+bufferName),'w') as logCSV:
            logCSV.write(header)
        keys = ['predictors','identifier','optimizer','layers','activations','dropouts']
        def check(blueprint):
            frame = loadLog(bufferName).astype(str)
            model = pd.Series(blueprint).astype(str).loc[['optimizer','layers','activations','dropouts']].tolist()
            return frame[frame[['optimizer','layers','activations','dropouts']].isin(model).all(axis=1)]
        def training(blueprint, patience=False, epochs=maxDuration, start=start, stop=stop):
            def getModel(blueprint, predictors, targets, metric):
                def getOutput():
                    if bool==targets[0].dtypes[0]:
                        activation = 'sigmoid'
                        loss = 'binary_crossentropy'
                    else:
                        activation = None
                        loss = 'MSE'
                    model.add(Dense(targets[0].columns.size, activation=activation, kernel_initializer='ones', name=("T_"+str(hash(name))[-4:]+"_"+str(len(model.layers)+2))))
                    model.compile(optimizer=blueprint['optimizer'], loss=loss, metrics=[metric])
                    return model
                name = blueprint['identifier']
                model = Sequential(name=name)
                model.add(Input(shape=(predictors[0].columns.size,), name=("I_"+str(hash(name))[-8:]+"_"+str(0))))
                for index, nodes in enumerate(blueprint['layers']):
                    activation = blueprint['activations'][index]
                    if activation=='None':
                        activation = None
                    model.add(Dense(nodes, activation, kernel_initializer='ones', name=("D_"+str(hash(name))[-4:]+"_"+str(index+1))))
                    if blueprint['dropouts'][index]>0:
                        model.add(Dropout(blueprint['dropouts'][index]/nodes, name=("O_"+str(hash(name))[-4:]+"_"+str(index+1))))
                model.add(BatchNormalization(name=("B_"+str(hash(name))[-4:]+"_"+str(len(model.layers)+1))))
                return getOutput()
            def evaluating(model, patience, epochs):
                monitor = EarlyStopping(monitor=('val_'+metric),restore_best_weights=True, patience=patience)
                start = dt.now()
                history = model.fit(predictors[0], targets[0], batchSize, epochs, 0, [monitor], validation_data=(predictors[1], targets[1]))
                time = (dt.now()-start).total_seconds()
                metrics = model.evaluate(predictors[1], targets[1], return_dict=True, verbose=0)
                metrics['time'] = time
                metrics['epochs'] = len(history.history[metric])
                return metrics
            def metrics2row(blueprint, metrics):
                def row2string(row):
                    s = ""
                    for value in row.values():
                        if isinstance(value, list):
                            v = '"'+str(value)+'",'
                        else:
                            v = str(value)+","
                        s = s + v
                    return s[0:-1]
                row = {}
                row['timestamp'] = dt.now()
                row.update(blueprint.copy())
                row['dimensions'] = len(blueprint['predictors'])
                row['length'] = len(blueprint['layers'])
                row['nodes'] = sum(blueprint['layers'])
                row.update(metrics)
                return row2string(row)
            epochRange = range(epochs, 0, -round(epochs/(start/stop)**0.7))
            decrease = (stop/start)**(1/max((len(epochRange)-1,1)))
            model = getModel(blueprint, predictors, targets, metric)
            trained = []
            times = []
            for epoch in epochRange:
                model.optimizer.lr = start
                backup = model.get_weights()
                image = model.evaluate(predictors[1], targets[1], return_dict=True, verbose=0)
                if patience:
                    metrics = evaluating(model, patience, epoch)
                else:
                    metrics = evaluating(model, epoch, epoch)
                if image[metric] <= metrics[metric]:
                    trained.append(metrics['epochs'])
                    times.append(metrics['time'])
                    model.save(path/'Models'/(blueprint['identifier']+'.h5'))
                else:
                    model.set_weights(backup)
                start = start*decrease
            metrics = model.evaluate(predictors[1], targets[1], return_dict=True, verbose=0)
            metrics['time'] = sum(times)
            metrics['epochs'] = sum(trained)
            return metrics2row(blueprint, metrics)
        def getSize():
            def getDuration(blueprint):
                nodes = sum(blueprint['layers'])
                return minDuration+round((maxDuration-minDuration)*((nodes-1)/(maxNodes-1)))
            blueprint = dict(zip(keys, [list(predictors[0].columns),identifier,'adam',[0],['None'],[0]]))
            for i in range(maxNodes):
                i = i
                for width in range(1, maxNodes-sum(blueprint['layers'])+1):
                    blueprint['layers'][-1] = width
                    backlog = check(blueprint)
                    if backlog.empty:
                        row2log(bufferName, training(blueprint, patience=minDuration, epochs=getDuration(blueprint), start=0.1, stop=0.1))
                blueprint = getBest(loadLog(bufferName), identifier, output=list(blueprint.keys()))
                if blueprint['layers'][-1]==1:
                    break
                if sum(blueprint['layers'])>=maxNodes:
                    break
                blueprint['layers'] = blueprint['layers']+[0]
                blueprint['activations'] = blueprint['activations']+[blueprint['activations'][0]]
                blueprint['dropouts'] = blueprint['dropouts']+[blueprint['dropouts'][0]]
        def getActivations():
            maxD = maxDuration
            possibilities = ['None','relu','selu','elu','tanh','softsign','softplus']
            blueprint = getBest(loadLog(bufferName), identifier, output=keys)
            for i in range(len(blueprint['layers'])):
                for activation in possibilities:
                    blueprint['activations'][i] = activation
                    backlog = check(blueprint)
                    if backlog.empty:
                        row2log(bufferName, training(blueprint, patience=minDuration, epochs=maxD, start=0.1, stop=0.1))
                    else:
                        best = getBest(backlog, identifier=identifier)
                        maxD = int(best['epochs'])
                blueprint = getBest(loadLog(bufferName), identifier, output=keys)
        def getDropouts():
            maxD = maxDuration
            blueprint = getBest(loadLog(bufferName), identifier, output=keys)
            for i, v in enumerate(blueprint['layers']):
                for drop in range(v):
                    blueprint['dropouts'][i] = drop
                    backlog = check(blueprint)
                    if backlog.empty:
                        row2log(bufferName, training(blueprint, patience=minDuration, epochs=maxD, start=0.1, stop=0.1))
                    else:
                        best = getBest(backlog, identifier=identifier)
                        maxD = int(best['epochs'])
                blueprint = getBest(loadLog(bufferName), identifier, output=keys)
        def getOptimizer():
            maxD = maxDuration
            blueprint = getBest(loadLog(bufferName), identifier, output=keys)
            possibilities = ['adam','sgd','rmsprop','adadelta','adagrad','adamax','nadam']
            for optimizer in possibilities:
                blueprint['optimizer'] = optimizer
                backlog = check(blueprint)
                if backlog.empty:
                    row2log(bufferName, training(blueprint, patience=minDuration, epochs=maxD, start=0.1, stop=0.1))
                else:
                    best = getBest(backlog, identifier=identifier)
                    maxD = int(best['epochs'])
        def flushLog():
            with open(path/'Logs'/(trainName+'_'+bufferName),'r') as logCSV:
                next(logCSV)
                for line in logCSV:
                    row2log('parameter_log.csv',line.replace('\n',''))
            return getBest(loadLog(bufferName), identifier, output=keys)
        getSize()
        getActivations()
        getDropouts()
        getOptimizer()
        return training(flushLog())
    def deeperTraining():
        best = loadLog('predictors_log.csv').sort_values(by=[metric, 'loss', 'epochs', 'nodes', 'time'], ascending=[minimise, True, True, True, True]).to_dict('rocords')[:epsilon]
        for good in best:
            row2log('deeper_log.csv', parameterTraining((predictors[0][good['predictors']], predictors[1][good['predictors']]), 100, good['identifier'], round(minDuration*2.5), round(maxDuration*2.5), start, stop/10))
        best = loadLog('deeper_log.csv').sort_values(by=[metric, 'loss', 'epochs', 'nodes', 'time'], ascending=[minimise, True, True, True, True]).to_dict('rocords')[0]
        print(parameterTraining((predictors[0][best['predictors']], predictors[1][best['predictors']]), 150, best['identifier'], 250, 1250, 1, 0.001))
    #predictorTraining()
    deeperTraining()
path = Path(__file__).parent.absolute()/'Deep Training'
initGPU()
predictors, targets = getData(path/'Data', 'None_Targets', 'None_Predictors', load=True, targetsCols=['Home: Win','Visiting: Win'], centerBy='Home: Win', centerSize=100000)
trainingRoutine('None', path, predictors, targets, 'binary_accuracy', False, 20, 100, 0.1, 0.01, worker=int(input()))