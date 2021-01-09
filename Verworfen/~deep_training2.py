import pandas as pd
import numpy as np
import random as rd
from pathlib import Path
from datetime import datetime as dt

import tensorflow as tf
import tensorflow.keras.optimizers as opt
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model, clone_model, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

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
def getBiasFreeIndex(boolSeries, size, seed=1337):
    rd.seed(seed)
    def getCenteredIndex(onSize=True):
        def flatter(a, b):
            c = []
            for i in range(len(a)):
                c.append(a[i])
                c.append(b[i])
            return c
        positive = boolSeries[boolSeries==True]
        negative = boolSeries[boolSeries==False]
        if onSize:
            positive = rd.sample(list(positive.index), size//2)
            negative = rd.sample(list(negative.index), size//2)
        else:
            if len(positive) > len(negative):
                positive = rd.sample(list(positive.index), len(negative))
                negative = negative.index.tolist()
            else:
                positive = positive.index.tolist()
                negative = rd.sample(list(negative.index), len(positive))
        return flatter(positive, negative)
    training = getCenteredIndex()
    boolSeries = boolSeries.loc[list(set(boolSeries.index)-set(training))]
    validation = getCenteredIndex(False)
    return training, validation
def divideDataByIndex(data, index):
    return data.loc[index[0]], data.loc[index[1]]
def saveData(path, dataTuple):
    dataTuple[0][0].to_csv(path/'predictors_training.csv')
    dataTuple[0][1].to_csv(path/'predictors_validation.csv')
    dataTuple[1][0].to_csv(path/'targets_training.csv')
    dataTuple[1][1].to_csv(path/'targets_validation.csv')
def loadData(path):
    predictors = [None, None]
    targets = [None, None]
    predictors[0] = pd.read_csv(path/'predictors_training.csv', index_col=0)
    predictors[1] = pd.read_csv(path/'predictors_validation.csv', index_col=0)
    targets[0] = pd.read_csv(path/'targets_training.csv', index_col=0)
    targets[1] = pd.read_csv(path/'targets_validation.csv', index_col=0)
    return tuple(predictors), tuple(targets)
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
def getBatchSize(size, minimum=1000):
    sizes = []
    for i in range((size//2)+1, 2, -1):
        if ((size % i)) == 0 and (size//i>1000) and (size//i<size//6):
            sizes.append(size//i)
    return sizes[len(sizes)//2]
def row2string(row):
    s = ""
    for value in row.values():
        if isinstance(value, list):
            v = '"'+str(value)+'",'
        else:
            v = str(value)+","
        s = s + v
    return s[0:-1]
def metrics2row(blueprint, metrics):
    def sumNodes(layers):
        return len(layers), sum(layers)
    row = {}
    row['timestamp'] = dt.now()
    row.update(blueprint.copy())
    row['dimensions'] = len(blueprint['predictors'])
    row['length'], row['nodes'] = sumNodes(blueprint['layers'])
    row.update(metrics)
    return row2string(row)
def training(path, blueprint, predictors, targets, metric, epochs=100, start=0.1, stop=0.01, output='row'):
    stepping = round(epochs/(start/stop)**0.7)
    epochRange = range(epochs, 0, -stepping)
    decrease = (stop/start)**(1/(len(epochRange)-1))
    model = getModel(blueprint, predictors, targets, metric)
    model.optimizer.lr = start
    lr = start
    modelPath = path/(blueprint['identifier']+'.h5')
    model.save(modelPath)
    trained = 0
    start = dt.now()
    for epoch in epochRange:
        monitor = EarlyStopping(monitor=('val_'+metric),restore_best_weights=True, patience=epoch)
        history = model.fit(predictors[0], targets[0], getBatchSize(len(targets[0])), epoch, 0, [monitor], validation_data=(predictors[1], targets[1]))
        image = load_model(modelPath)
        imageMetric = image.evaluate(predictors[1], targets[1], return_dict=True, verbose=0)[metric]
        modelMetric = model.evaluate(predictors[1], targets[1], return_dict=True, verbose=0)[metric]
        if imageMetric>modelMetric:
            model = image
        else:
            trained = trained+len(history.history[metric])
            model.save(modelPath)
        lr = lr*decrease
        model.optimizer.lr = lr
    time = round((dt.now()-start).microseconds/1000000, 2)
    metrics = model.evaluate(predictors[1], targets[1], return_dict=True, verbose=0)
    metrics['time'] = time
    metrics['epochs'] = trained
    if output=='metric':
        return metrics
    elif output=='row':
        return metrics2row(blueprint, metrics)
    elif output=='ensemble':
        return metrics2row(blueprint, metrics),(
            pd.DataFrame(model.predict(predictors[0]), columns=targets[0].columns, index=targets[0].index),
            pd.DataFrame(model.predict(predictors[1]), columns=targets[1].columns, index=targets[1].index))
def trainingRoutine(path, predictors, targets, metric, minimise, minDuration, maxDuration, start, stop):
    def row2log(name, row):
        with open(path/name,'a') as logCSV:
            logCSV.write('\n')
            logCSV.write(row)
    def loadLog(name):
        df = pd.read_csv(path/name, index_col=False)
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
    def getBest(frame, identifier, output=[]):
        if not output:
            output = list(frame.columns)
        frame = frame[frame['identifier']==identifier]
        if minimise:
            frame = frame[frame[metric]==frame[metric].min()]
        else:
            frame = frame[frame[metric]==frame[metric].max()]
        if len(frame)>1:
            frame = frame[frame['loss']==frame['loss'].min()]
            if len(frame)>1:
                frame = frame[frame['epochs']==frame['epochs'].min()]
                if len(frame)>1:
                    frame = frame[frame['nodes']==frame['nodes'].min()]
                    if len(frame)>1:
                        frame = frame[frame['time']==frame['time'].min()]
                        return frame[output].to_dict('records')[0]
                    else:
                      return frame[output].to_dict('records')[0]  
                else:
                    return frame[output].to_dict('records')[0]
            else:
                return frame[output].to_dict('records')[0]
        else:
            return frame[output].to_dict('records')[0]
    def getIdentifier(sample):
        def string2int(string):
            value = 0
            for char in string:
                value = value+ord(char)*1337*len(string)*31
            return value
        identifier = 0
        for string in sample:
            identifier = identifier + string2int(string)
        return str(identifier)[-16:]
    def predictorTraining(epsilon=8):
        def sample(columns, bias, maxNodes):
            tries = []
            def appendTry(tries, row):
                if row[metric]<bias:
                    return
                if len(tries)<epsilon:
                    tries.append(row)
                    tries = sorted(tries, key=lambda tup: tup[metric])
                elif tries[0][metric]<row[metric]:
                    tries[0] = row
                    tries = sorted(tries, key=lambda tup: tup[metric])
            def check(identifier):
                frame = loadLog('predictors_log.csv')
                if frame.empty:
                    return frame
                else:
                    return frame[frame['identifier']==identifier]
            pool = list(set(predictors[0].columns)-set(columns))
            for column in pool:
                samples = columns+[column]
                identifier = getIdentifier(samples)
                backlog = check(identifier)
                if backlog.empty:
                    samples = (predictors[0][samples], predictors[1][samples])
                    print("Trying:    ", column)
                    print("Identifier:", identifier)
                    tryStart = dt.now()
                    row = parameterTraining(samples, maxNodes, identifier, maxNodes*10)
                    print("Endurance:", dt.now()-tryStart)
                    row2log('predictors_log.csv', row)
                    row = getBest(loadLog('predictors_log.csv'), identifier)
                    print("Metric:   ", row[metric])
                    appendTry(tries, row)
                else:
                    backlog = getBest(backlog, identifier)
                    print("Skipping:  ", column)
                    print("Identifier:", identifier)
                    print("Metric:    ", backlog[metric])
                    appendTry(tries, backlog)
            return tries
        def trace(line, bias, nodes):
            trial = sample(line, bias, nodes)
            if not trial:
                return
            for entry in trial:
                preds = entry['predictors']
                maxNodes = max([(entry['nodes']/len(preds))*(len(preds)+1), len(preds)*5])
                print(maxNodes)
                trace(preds, entry[metric], round(maxNodes))
        trace([], 0.5, 10)
    def parameterTraining(predictors, maxNodes, identifier, epochs):
        keys = ['predictors','identifier','optimizer','layers','activations','dropouts']
        def getDuration(nodes):
            return minDuration+round((maxDuration-minDuration)*((nodes-1)/(maxNodes-1)))
        def check(blueprint):
            frame = loadLog('parameter_log.csv').astype(str)
            return frame[frame[list(blueprint.keys())].isin(pd.Series(blueprint).astype(str).tolist()).all(axis=1)]
        def evaluating(model, epochs):
            monitor = EarlyStopping(monitor=('val_'+metric),restore_best_weights=True, patience=minDuration)
            start = dt.now()
            history = model.fit(predictors[0], targets[0], getBatchSize(len(targets[0])), epochs, 0, [monitor], validation_data=(predictors[1], targets[1]))
            time = (dt.now()-start).total_seconds()
            metrics = model.evaluate(predictors[1], targets[1], return_dict=True, verbose=0)
            metrics['time'] = time
            metrics['epochs'] = len(history.history[metric])
            return metrics
        def getSize():
            blueprint = dict(zip(keys, [list(predictors[0].columns),identifier,'adam',[0],['None'],[0]]))
            for i in range(maxNodes):
                i = i
                for width in range(1, maxNodes-sum(blueprint['layers'])+1):
                    blueprint['layers'][-1] = width
                    backlog = check(blueprint)
                    if backlog.empty:
                        model = getModel(blueprint, predictors, targets, metric)
                        metrics = evaluating(model, getDuration(sum(blueprint['layers'])))
                        row2log('parameter_log.csv', metrics2row(blueprint, metrics))
                blueprint = getBest(loadLog('parameter_log.csv'), identifier, output=list(blueprint.keys()))
                if blueprint['layers'][-1]==1:
                    break
                blueprint['layers'] = blueprint['layers']+[0]
                blueprint['activations'] = blueprint['activations']+[blueprint['activations'][0]]
                blueprint['dropouts'] = blueprint['dropouts']+[blueprint['dropouts'][0]]
        def getActivations():
            maxD = maxDuration
            possibilities = ['None','relu','selu','elu','tanh','softsign','softplus']
            blueprint = getBest(loadLog('parameter_log.csv'), identifier, output=keys)
            for i in range(len(blueprint['layers'])):
                for activation in possibilities:
                    blueprint['activations'][i] = activation
                    backlog = check(blueprint)
                    if backlog.empty:
                        model = getModel(blueprint, predictors, targets, metric)
                        metrics = evaluating(model, maxD)
                        row2log('parameter_log.csv', metrics2row(blueprint, metrics))
                    else:
                        best = getBest(backlog, identifier=identifier)
                        maxD = int(best['epochs'])
                blueprint = getBest(loadLog('parameter_log.csv'), identifier, output=keys)
        def getDropouts():
            maxD = maxDuration
            blueprint = getBest(loadLog('parameter_log.csv'), identifier, output=keys)
            for i, v in enumerate(blueprint['layers']):
                for drop in range(v):
                    blueprint['dropouts'][i] = drop
                    backlog = check(blueprint)
                    if backlog.empty:
                        model = getModel(blueprint, predictors, targets, metric)
                        metrics = evaluating(model, maxD)
                        row2log('parameter_log.csv', metrics2row(blueprint, metrics))
                    else:
                        best = getBest(backlog, identifier=identifier)
                        maxD = int(best['epochs'])
                blueprint = getBest(loadLog('parameter_log.csv'), identifier, output=keys)
        def getOptimizer():
            maxD = maxDuration
            blueprint = getBest(loadLog('parameter_log.csv'), identifier, output=keys)
            possibilities = ['adam','sgd','rmsprop','adadelta','adagrad','adamax','nadam']
            for optimizer in possibilities:
                blueprint['optimizer'] = optimizer
                backlog = check(blueprint)
                if backlog.empty:
                    model = getModel(blueprint, predictors, targets, metric)
                    metrics = evaluating(model, maxD)
                    row2log('parameter_log.csv', metrics2row(blueprint, metrics))
                else:
                    best = getBest(backlog, identifier=identifier)
                    maxD = int(best['epochs'])
        getSize()
        getActivations()
        getDropouts()
        getOptimizer()
        return training(path/'Models', getBest(loadLog('parameter_log.csv'), identifier, output=keys), predictors, targets, metric, epochs, start, stop)
    predictorTraining()
path = Path(__file__).parent.absolute()/'Learning'/'Deep Training'
#targets = pd.read_csv(Path(__file__).parent.absolute()/'Learning'/'None_Targets.csv', index_col=False, usecols=['Home: Win', 'Visiting: Win'])
#index   = getBiasFreeIndex(targets['Home: Win'], 72500)
#targets = divideDataByIndex(targets, index)
#predictors = pd.read_csv(Path(__file__).parent.absolute()/'Learning'/'None_Predictors.csv', index_col=False)
#predictors = divideDataByIndex(predictors, index)
#saveData(path/'Data', (predictors, targets))
predictors, targets = loadData(path/'Data')
trainingRoutine(path, predictors, targets, 'binary_accuracy', False, 20, 100, 0.1, 0.01)