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
def getPredifinedPatterns():
    patterns = {
        ('pythagorean','all'):([
            'Pythagorean expectation versus ratio diffrence',
            'Pythagorean expectation ratio diffrence',
            'Team - Pythagorean expectation diffrence'],
            ['Versus','Ratio','Average']),
        ('performance','versus'):([
            'Fielding performance versus ratio diffrence',
            'Pitching performance versus ratio diffrence',
            'Batting performance versus ratio diffrence',
            'Pythagorean expectation versus ratio diffrence'],
            ['Fielding','Pitching','Batting','Pythagorean']),
        ('performance','ratio'):([
            'Fielding performance ratio diffrence',
            'Pitching performance ratio diffrence',
            'Batting performance ratio diffrence',
            'Pythagorean expectation ratio diffrence'],
            ['Fielding','Pitching','Batting','Pythagorean']),
        ('performance','average'):([
            'Average Fielding performance diffrence',
            'Pitcher - Pitching performance diffrence',
            'Average Batting performance diffrence',
            'Team - Pythagorean expectation diffrence'],
            ['Fielding','Pitching','Batting','Pythagorean']),
        ('score','ratio'):([
            'Score ratio diffrence',
            'Odd ratio diffrence'],
            ['Score','Odd']),
        ('score','versus'):([
            'Score versus ratio diffrence',
            'Odd versus ratio diffrence'],
            ['Score','Odd']),
        ('people','average'):([
            'Average age diffrence',
            'Batting side diffrence',
            'Throwing side diffrence',
            'Average BMI diffrence'],
            ['Age','Batting','Throwing','BMI']),
        ('pitcher','average'):([
            'Pitcher - Strikeouts per walk diffrence',
            'Pitcher - Homeruns per game diffrence',
            'Pitcher - Shutouts per game diffrence',
            'Pitcher - Saves per game diffrence',
            'Pitcher - ERA diffrence'],
            ['Strikeouts','Homeruns','Shutouts','Saves','ERA'])
        }
    return patterns
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
    if isinstance(data, pd.DataFrame):
        return data.loc[index[0]], data.loc[index[1]]
    if isinstance(data, dict):
        for frame in data:
            data[frame] = (data[frame].loc[index[0]],data[frame].loc[index[1]])
        return data
def getDataAsDictonary(data, patterns, index=None, savePath=None):
    def load():
        dataDictionary = {}
        for frame, pattern in patterns.items():
            entry = data[pattern[0]]
            if pattern[1]!=None:
                entry = entry.rename(columns=dict(zip(pattern[0],pattern[1])))
            dataDictionary[frame] = entry
        return dataDictionary
    data = load()
    if index!=None:
        data = divideDataByIndex(data, index)
    if savePath!=None:
        for frame in data:
            name = frame[0]+'-'+frame[1]
            if index!=None:
                data[frame][0].to_csv(savePath/(name+'_training.csv'))
                data[frame][1].to_csv(savePath/(name+'_validation.csv'))
            else:
                data[frame].to_csv(savePath/(name+'.csv'))
    return data
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
    name = blueprint['predictor']+"_"+blueprint['description']
    model = Sequential(name=name)
    model.add(Input(shape=(predictors[0].columns.size,), name=("I_"+str(hash(name))[-8:]+"_"+str(0))))
    for index, nodes in enumerate(blueprint['layers']):
        model.add(Dense(nodes, blueprint['activations'][index], kernel_initializer='ones', name=("D_"+str(hash(name))[-8:]+"_"+str(index+1))))
        if blueprint['dropouts'][index]>0:
            model.add(Dropout(blueprint['dropouts'][index]/nodes, name=("O_"+str(hash(name))[-8:]+"_"+str(index+1))))
    model.add(BatchNormalization(name=("B_"+str(hash(name))[-8:]+"_"+str(len(model.layers)+1))))
    return getOutput()
def getBatchSize(size, minimum=1000):
    sizes = []
    for i in range((size//2)+1, 2, -1):
        if ((size % i)) == 0 and (size//i>1000) and (size//i<size//6):
            sizes.append(size//i)
    return sizes[len(sizes)//2]
def metrics2row(blueprint, metrics):
    def getCopy(blueprint):
        copy = {}
        for key, item in blueprint.items():
            if isinstance(item, str):
                copy[key] = item
            else:
                copy[key] = item.copy()
        return copy
    row = {}
    row['timestamp'] = dt.now()
    row.update(getCopy(blueprint.copy()))
    row['length'] = len(blueprint['layers'])
    row['nodes'] = sum(blueprint['layers'])
    row.update(metrics)
    return row
def training(path, blueprint, predictors, targets, metric, epochs=100, start=0.1, stop=0.01, output='metric'):
        stepping = round(epochs/(start/stop)**0.7)
        epochRange = range(epochs, 0, -stepping)
        decrease = (stop/start)**(1/(len(epochRange)-1))
        model = getModel(blueprint, predictors, targets, metric)
        model.optimizer.lr = start
        lr = start
        modelPath = path/(blueprint['predictor']+"-"+blueprint['description']+'.h5')
        model.save(modelPath)
        trained = 0
        start = dt.now()
        for epoch in epochRange:
            print("epoche:", epoch,"learning rate:", round(model.optimizer.lr.numpy(), 16))
            monitor = EarlyStopping(monitor=('val_'+metric),restore_best_weights=True, patience=epoch)
            history = model.fit(predictors[0], targets[0], getBatchSize(len(targets[0])), epoch, 0, [monitor], validation_data=(predictors[1], targets[1]))
            image = load_model(modelPath)
            imageMetric = image.evaluate(predictors[1], targets[1], return_dict=True, verbose=0)[metric]
            modelMetric = model.evaluate(predictors[1], targets[1], return_dict=True, verbose=0)[metric]
            print("Image:", imageMetric, "Model:", modelMetric)
            if imageMetric>modelMetric:
                model = image
            else:
                trained = trained+len(history.history[metric])
                model.save(modelPath)
            lr = lr*decrease
            model.optimizer.lr = lr
        time = round((dt.now()-start).microseconds/1000000, 2)
        metrics = model.evaluate(predictors[1], targets[1], return_dict=True)
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
def trainingRoutine(path, predictors, targets, metric, description, log=[], minimise=False, minDuration=50, maxDuration=250, epochs=1000, start=0.1, stop=0.001):
    def parameterTraining(output='ensemble'):
        def logModel(blueprint, metrics, update=True):
            row = metrics2row(blueprint, metrics)
            if update:
                log.append(row)
                frame = pd.DataFrame(log)
                frame.to_csv(path/(str(dt.now().date())+"_log.csv"), index=False)
            return row
        def getBest(log):
            frame = pd.DataFrame(log)
            frame = frame[frame['predictor']==blueprint['predictor']]
            frame = frame[frame['description']==blueprint['description']]
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
                            return frame[list(blueprint.keys())].to_dict('records')[0]
                        else:
                          return frame[list(blueprint.keys())].to_dict('records')[0]  
                    else:
                        return frame[list(blueprint.keys())].to_dict('records')[0]
                else:
                    return frame[list(blueprint.keys())].to_dict('records')[0]
            else:
                return frame[list(blueprint.keys())].to_dict('records')[0]
        def evaluating(model, patience=minDuration, epochs=maxDuration):
            monitor = EarlyStopping(monitor=('val_'+metric),restore_best_weights=True, patience=patience)
            start = dt.now()
            history = model.fit(predictors[0], targets[0], getBatchSize(len(targets[0])), epochs, 0, [monitor], validation_data=(predictors[1], targets[1]))
            time = (dt.now()-start).total_seconds()
            metrics = model.evaluate(predictors[1], targets[1], return_dict=True)
            metrics['time'] = time
            metrics['epochs'] = len(history.history[metric])
            return metrics
        def getLength(blueprint):
            minLength = 1
            maxLenght = predictors[0].columns.size*2
            def getDuration():
                nodes = sum(blueprint['layers'])
                minNodes = blueprint['layers'][0]*minLength
                maxNodes = blueprint['layers'][0]*maxLenght
                return minDuration+round((maxDuration-minDuration)*(nodes-minNodes)/maxNodes)
            tempLog = []
            for length in range(minLength, maxLenght+1):
                blueprint['layers'] = [blueprint['layers'][0]]*length
                blueprint['activations'] = [blueprint['activations'][0]]*length
                blueprint['dropouts'] = [blueprint['dropouts'][0]]*length
                model = getModel(blueprint, predictors, targets, metric)
                metrics = evaluating(model, patience=getDuration()//4, epochs=getDuration())
                tempLog.append(logModel(blueprint, metrics))
            return getBest(tempLog)
        def getSize(blueprint):
            minWidth = 1
            maxWidth = predictors[0].columns.size*2
            def getDuration():
                nodes = sum(blueprint['layers'])
                minNodes = len(blueprint['layers'])*minWidth
                maxNodes = len(blueprint['layers'])*maxWidth
                return minDuration+round((maxDuration-minDuration)*(nodes-minNodes)/maxNodes)
            tempLog = []
            for index in range(len(blueprint['layers'])):
                for width in range(minWidth, maxWidth+1):
                    blueprint['layers'][index] = width
                    model = getModel(blueprint, predictors, targets, metric)
                    metrics = evaluating(model, patience=getDuration()//4, epochs=getDuration())
                    tempLog.append(logModel(blueprint, metrics))
                blueprint = getBest(tempLog)
            return blueprint
        def getActivations(blueprint):
            tempLog = []
            possibilities = [None,'relu','selu','elu','tanh','softsign','softplus']
            for index in range(len(blueprint['layers'])):
                for activation in possibilities:
                    blueprint['activations'][index] = activation
                    model = getModel(blueprint, predictors, targets, metric)
                    metrics = evaluating(model)
                    tempLog.append(logModel(blueprint, metrics))
                blueprint = getBest(tempLog)
            return blueprint
        def getDropouts(blueprint):
            tempLog = []
            for index, nodes in enumerate(blueprint['layers']):
                for drop in range(nodes):
                    blueprint['dropouts'][index] = drop
                    model = getModel(blueprint, predictors, targets, metric)
                    metrics = evaluating(model)
                    tempLog.append(logModel(blueprint, metrics))
                blueprint = getBest(tempLog)
            return blueprint
        def getOptimizer(blueprint):
            tempLog = []
            possibilities = ['sgd','rmsprop','adam','adadelta','adagrad','adamax','nadam']
            for optimizer in possibilities:
                blueprint['optimizer'] = optimizer
                model = getModel(blueprint, predictors, targets, metric)
                metrics = evaluating(model, patience=2*minDuration, epochs=2*maxDuration)
                tempLog.append(logModel(blueprint, metrics))
            return getBest(tempLog)
        blueprint = {
            'predictor':description[0],'description':description[1],'optimizer':'adam',
            'layers':[predictors[0].columns.size],'activations':['relu'],'dropouts':[0]}
        blueprint = getActivations(blueprint)
        blueprint = getSize(blueprint)
        blueprint = getLength(blueprint)
        blueprint = getSize(blueprint)
        blueprint = getActivations(blueprint)
        blueprint = getDropouts(blueprint)
        blueprint = getOptimizer(blueprint)
        return training(path, blueprint, predictors, targets, metric, epochs, start, stop, output)
    if isinstance(predictors, pd.DataFrame):
        return parameterTraining(output='ensemble')
    elif isinstance(predictors, dict):
        metrics = []
        predictions = [targets[0], targets[1]]
        for description, predictors in predictors.items():
            row, prediction = parameterTraining()
            predictions[0] = pd.merge(predictions[0], prediction[0], how="left", left_index=True, right_index=True, suffixes=(None, "_"+row['predictor']+"-"+row['description']))
            predictions[1] = pd.merge(predictions[1], prediction[1], how="left", left_index=True, right_index=True, suffixes=(None, "_"+row['predictor']+"-"+row['description']))
            metrics.append(row)
            pd.DataFrame(metrics).to_csv(path/'Deep Training'/(str(dt.now().date())+"_metrics.csv"), index=False)
            predictions[0].drop(columns=targets[0].columns).to_csv(path/'Data'/'ensemble_training.csv')
            predictions[1].drop(columns=targets[1].columns).to_csv(path/'Data'/'ensemble_validation.csv')
def predictorsTraining():
    print("tbd")
path = Path(__file__).parent.absolute()/'Learning'
targets = pd.read_csv(path/'-_Targets.csv', index_col=False, usecols=['Home: Win', 'Visiting: Win'])
index   = getBiasFreeIndex(targets['Home: Win'], 72500)
print(len(index[0]), len(index[1]))
targets = divideDataByIndex(targets, index)
print(targets[0])
print(targets[1])
predictors = pd.read_csv(path/'-_Predictors.csv', index_col=False)
predictors = divideDataByIndex(predictors, index)
print(predictors[0])
print(predictors[1])