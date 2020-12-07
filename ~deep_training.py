import pandas as pd
import numpy as np
import random as rd
from pathlib import Path

import tensorflow as tf
import tensorflow.keras.optimizers as opt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

def createData(path):
    def renameTrainingProgrammColumns(frame):
        newNames = {}
        for column in frame:
            if column.find("Pythagorean")>-1:
                newNames[column] = "Pythagorean"
            elif column.find("Pitching")>-1:
                newNames[column] = "Pitching"
            elif column.find("Batting")>-1:
                newNames[column] = "Batting"
            elif column.find("Fielding")>-1:
                newNames[column] = "Fielding"
        return frame.rename(columns=newNames)

    def loadPredictors(path, winOptimised=True):
        dictonary = {}
        dictonary['predictors'] = pd.read_csv(path/'Predictors.csv', dtype="float32", index_col=False)
        versusRatioTrainees = [
            'Fielding performance versus ratio diffrence','Pitching performance versus ratio diffrence',
            'Batting performance versus ratio diffrence']
        dictonary['versusRatioTrainees'] = dictonary['predictors'][versusRatioTrainees]
        dictonary['versusRatioTrainees'] = renameTrainingProgrammColumns(dictonary['versusRatioTrainees'])
        ratioTrainees = [
            'Fielding performance ratio diffrence','Pitching performance ratio diffrence',
            'Batting performance ratio diffrence']
        dictonary['ratioTrainees'] = dictonary['predictors'][ratioTrainees]
        dictonary['ratioTrainees'] = renameTrainingProgrammColumns(dictonary['ratioTrainees'])
        trainees = [
            'Average Fielding performance diffrence','Pitcher - Pitching performance diffrence',
            'Average Batting performance diffrence']
        dictonary['trainees'] = dictonary['predictors'][trainees]
        dictonary['trainees'] = renameTrainingProgrammColumns(dictonary['trainees'])
        scoreTrainees = [
            'Score ratio diffrence','Odd ratio diffrence']
        dictonary['scoreTrainees'] = dictonary['predictors'][scoreTrainees]
        dictonary['scoreTrainees'] = dictonary['scoreTrainees'].rename(columns=dict(zip(scoreTrainees,['Score','Odd'])))
        scoreVersusTrainees = [
            'Score versus ratio diffrence','Odd versus ratio diffrence']
        dictonary['scoreVersusTrainees'] = dictonary['predictors'][scoreVersusTrainees]
        dictonary['scoreVersusTrainees'] = dictonary['scoreVersusTrainees'].rename(columns=dict(zip(scoreVersusTrainees,['Score','Odd'])))
        dictonary['predictors'] = dictonary['predictors'].drop(columns=versusRatioTrainees+ratioTrainees+trainees+scoreTrainees+scoreVersusTrainees)
        if winOptimised:
            dictonary['others'] = dictonary['predictors'][[
                'Pitcher - Strikeouts per walk diffrence',
                'Pitcher - Homeruns per game diffrence',
                'Pitcher - Shutouts per game diffrence',
                'Pitcher - Saves per game diffrence',
                'Pitcher - ERA diffrence',
                'Pythagorean expectation versus ratio diffrence',
                'Pythagorean expectation ratio diffrence',
                'Team - Pythagorean expectation diffrence']]
            dictonary['predictors'] = dictonary['predictors'].drop(columns=[
                'Pitcher - Strikeouts per walk diffrence',
                'Pitcher - Homeruns per game diffrence',
                'Pitcher - Shutouts per game diffrence',
                'Pitcher - Saves per game diffrence',
                'Pitcher - ERA diffrence',
                'Pythagorean expectation versus ratio diffrence',
                'Pythagorean expectation ratio diffrence',
                'Team - Pythagorean expectation diffrence'])
        return dictonary

    def loadTargets(path):
        def loadPerformanceTrainers(path):
            performanceTrainers = pd.read_csv(path/'Targets.csv', dtype="float32", index_col=False, usecols=[
                'Fielding performance diffrence','Pitching performance diffrence','Batting performance diffrence'])
            return renameTrainingProgrammColumns(performanceTrainers)
        def loadScoreTrainer(path):
            scoreTrainers = pd.read_csv(path/'Targets.csv', dtype="float32", index_col=False, usecols=[
                'Visiting: Score','Home: Score'])
            scoreTrainers['Score'] = scoreTrainers['Home: Score'] - scoreTrainers['Visiting: Score']
            scoreTrainers['Odd'] = (scoreTrainers['Home: Score'] - scoreTrainers['Visiting: Score'])/(scoreTrainers['Home: Score'] + scoreTrainers['Visiting: Score']).replace(0,1)
            return scoreTrainers[['Score','Odd']]
        dictonary = {}
        dictonary['targets'] = pd.read_csv(path/'Targets.csv', dtype="float32", index_col=False, usecols=[
            'Visiting: Win','Home: Win'])
        dictonary['performanceTrainers'] = loadPerformanceTrainers(path)
        dictonary['scoreTrainers'] = loadScoreTrainer(path)
        return dictonary

    def getCenteredIndex(target, seed=1337):
        rd.seed(seed)
        positive = target[target==True]
        negative = target[target==False]
        if len(positive) > len(negative):
            positive = positive[rd.sample(list(positive.index), len(negative))]
        else:
            negative = negative[rd.sample(list(negative.index), len(positive))]
        return sorted(list(positive.index)+list(negative.index))

    def getDataByIndex(data, index):
        newData = {}
        for frame in data:
            newData[frame] = data[frame].iloc[index].reset_index(drop=True)
        return newData

    def centerData(data, index=pd.Series(), seed=1337):
        if index.empty:
            index = getCenteredIndex(data['targets']['Home: Win'], seed)
        return getDataByIndex(data, index)

    def getSplit(target, size=0.1337, seed=1337):
        rd.seed(seed)
        split = rd.sample(list(target.index),round(size*target.index.size))
        split = getCenteredIndex(target[split], seed)
        return (list(set(target.index)-set(split)), split)

    def saveData(overDictonary, path):
        for dataType in overDictonary:
            for data in overDictonary[dataType]:
                for frame in overDictonary[dataType][data]:
                    overDictonary[dataType][data][frame].to_csv(path/'Deep Training'/dataType/data/(frame+".csv"), index = False)

    targets = centerData(loadTargets(path))
    predictors = centerData(loadPredictors(path), targets['targets']['Home: Win'].index)
    split = getSplit(targets['targets']['Home: Win'])
    overDictonary = {}
    overDictonary['training'] = {'targets':getDataByIndex(targets, split[0]),'predictors':getDataByIndex(predictors, split[0])}
    overDictonary['validation'] = {'targets':getDataByIndex(targets, split[1]),'predictors':getDataByIndex(predictors, split[1])}
    saveData(overDictonary, path)

def loadData(path):
    def getData():
        return {
            'predictors':dict.fromkeys(['predictors','versusRatioTrainees','ratioTrainees','trainees','scoreTrainees','scoreVersusTrainees','others']),
            'targets':dict.fromkeys(['targets','performanceTrainers','scoreTrainers'])}
    overDictonary = {'validation':getData(),'training':getData()}
    for dataType in overDictonary:
        for data in overDictonary[dataType]:
            for frame in overDictonary[dataType][data]:
                temp = pd.read_csv(path/'Deep Training'/dataType/data/(frame+".csv"), index_col= False)
                for column in temp.columns:
                    if column.find(": Win")>-1:
                        temp[column] = temp[column].astype('bool')
                overDictonary[dataType][data][frame] = temp
    return overDictonary

def deepTrain(predictors, targets, path, name, targetType="binary"):
    predictors, valPredictors = predictors
    targets, valTargets = targets
    modelLog = {'model':[],'accuracy':[]}
    def logModels(path, name, modelLog):
        frame = pd.DataFrame(modelLog)
        frame.to_csv(path/(name+"_modelLog.csv"), index=False)
        return frame.iloc[frame['accuracy'].idxmax()]
    def checkModel(model, logModel, predictors, valPredictors, targets, valTargets, targetType, batchSize):
        model_accuracy = evaluatingTraining(model, predictors, valPredictors, targets, valTargets, targetType, batchSize, 100, 10)
        print("model_accuracy:", model_accuracy)
        log_model_accuracy = evaluatingTraining(logModel, predictors, valPredictors, targets, valTargets, targetType, batchSize, 100, 10)
        print("log_model_accuracy:", log_model_accuracy)
        if log_model_accuracy>model_accuracy:
            return logModel
        return model
    def getBatchSize(size, min=1000):
        for i in range((size//2)+1, 2, -1):
            if ((size % i)) == 0 and (size/i>min) and (size/i<size//3):
                return size//i
        return min
    def evaluatingTraining(model, predictors, valPredictors, targets, valTargets, targetType, batchSize, epochs, patience):
        if targetType=="binary":
            monitor = EarlyStopping(monitor='val_binary_accuracy',restore_best_weights=True, patience=patience)
            model.add(BatchNormalization())            
            model.add(Dense(targets.columns.size, activation='sigmoid', kernel_initializer='ones'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
        elif targetType=="regression":
            monitor = EarlyStopping(monitor='val_accuracy',restore_best_weights=True, patience=patience)          
            model.add(BatchNormalization())
            model.add(Dense(targets.columns.size, kernel_initializer='ones'))
            model.compile(optimizer='adam', loss='MSE', metrics=['accuracy'])
        model.fit(x=predictors, y=targets, epochs=epochs, batch_size=batchSize, callbacks=[monitor], validation_data=(valPredictors, valTargets))
        return model.evaluate(valPredictors, valTargets)[1]
    def getLength(predictors, valPredictors, targets, valTargets, targetType, batchSize):
        metrics = []
        lengths = []
        for length in range(2, predictors.columns.size*4+4):
            model = getModel(predictors, targets, targetType, [predictors.columns.size]*length, ['tanh']*length, [0]*length, returnBlueprint=True)
            lengths.append(length)
            metric = evaluatingTraining(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize, 1000, 100)
            metrics.append(metric)
            print(length, metric)
            modelLog['model'].append(model[1])
            modelLog['accuracy'].append(metric)
        return lengths[metrics.index(max(metrics))]
    def getSize(predictors, valPredictors, targets, valTargets, targetType, batchSize, length):
        size = []
        for layer in range(1, length+1):
            widths = []
            metrics = []
            for width in range(2, predictors.columns.size*2+2):
                model = getModel(predictors, targets, targetType, size+[width]+([predictors.columns.size]*(length-layer)), ['tanh']*length, [0]*length, returnBlueprint=True)
                widths.append(width)
                metric = evaluatingTraining(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize, 100, 10)
                metrics.append(metric)
                print(layer, width, metric)
                modelLog['model'].append(model[1])
                modelLog['accuracy'].append(metric)
            size.append(widths[metrics.index(max(metrics))])
        return size
    def getDrops(predictors, valPredictors, targets, valTargets, targetType, batchSize, size, activations=[]):
        drops = []
        if not activations:
            activations = ['tanh']*len(size)
        for layer, width in enumerate(size,1):
            tempDrops = []
            metrics = []
            for DOR in range(width):
                drop = DOR/width
                model = getModel(predictors, targets, targetType, size, activations, drops+[drop]+[0]*(len(size)-layer), returnBlueprint=True)
                tempDrops.append(drop)
                metric = evaluatingTraining(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize, 100, 10)
                metrics.append(metric)
                print(layer, DOR, metric)
                modelLog['model'].append(model[1])
                modelLog['accuracy'].append(metric)
            drops.append(tempDrops[metrics.index(max(metrics))])
        return drops
    def getActivations(predictors, valPredictors, targets, valTargets, targetType, batchSize, size, drops=[]):
        possibilities = ['relu','tanh','selu','elu']
        activations = []
        if not drops:
            drops = [0]*len(size)
        for layer in range(1, len(size)+1):
            tempActivations = []
            metrics = []
            for activation in possibilities:
                model = getModel(predictors, targets, targetType, size, activations+[activation]+['tanh']*(len(size)-layer), drops, returnBlueprint=True)
                tempActivations.append(activation)
                metric = evaluatingTraining(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize, 100, 10)
                metrics.append(metric)
                print(layer, activation, metric)
                modelLog['model'].append(model[1])
                modelLog['accuracy'].append(metric)
            activations.append(tempActivations[metrics.index(max(metrics))])
        return activations
    def getModel(predictors, targets, targetType, size, activations, drops, optimizer='adam', returnBlueprint=False):
        model = Sequential()
        model.add(Input(shape=(predictors.columns.size,)))
        for i in range(len(size)):
            model.add(Dense(size[i], activations[i], kernel_initializer='ones'))
            if drops[i]>0:
                model.add(Dropout(drops[i]))
        if targetType=="binary":
            model.add(BatchNormalization())            
            model.add(Dense(targets.columns.size, activation='sigmoid', kernel_initializer='ones'))
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
        elif targetType=="regression":
            model.add(BatchNormalization())
            model.add(Dense(targets.columns.size, kernel_initializer='ones'))
            model.compile(optimizer=optimizer, loss='MSE', metrics=['accuracy'])
        if returnBlueprint:
            return model, optimizer+str(size)+str(activations)+str(drops)
        return model
    def getOptimizer(predictors, targets, targetType, batchSize, size, activations, drops, epochs=250):
        possibilities = ['sgd','rmsprop','adam','adadelta','adagrad','adamax','nadam']
        metrics = []
        for optimizer in possibilities:
            model = getModel(predictors, targets, targetType, size, activations, drops, optimizer=optimizer, returnBlueprint=True)
            metric = evaluatingTraining(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize, epochs, epochs//5)
            metrics.append(metric)
            print(optimizer, metric)
            modelLog['model'].append(model[1])
            modelLog['accuracy'].append(metric)
        return possibilities[metrics.index(max(metrics))]
    def getOptimzerInstance(optimizer, lr=1):
            if optimizer=='sgd':
                return opt.SGD(learning_rate=lr)
            elif optimizer=='rmsprop':
                return opt.RMSprop(learning_rate=lr)
            elif optimizer=='adadelta':
                return opt.Adadelta(learning_rate=lr)
            elif optimizer=='adagrad':
                return opt.Adagrad(learning_rate=lr)
            elif optimizer=='adamax':
                return opt.Adamax(learning_rate=lr)
            elif optimizer=='nadam':
                return opt.Nadam(learning_rate=lr)
            elif optimizer=='ftrl':
                return opt.Ftrl(learning_rate=lr)
            else:
                return opt.Adam(learning_rate=lr)
    def training(path, name, model, predictors, valPredictors, targets, valTargets, targetType, batchSize, steps=1000, decrease=0.9):
        def getES(targetType, patience):
            if targetType=="binary":
                return EarlyStopping(monitor='val_binary_accuracy', patience=steps//10, restore_best_weights=True)
            elif targetType=="regression":
                return EarlyStopping(monitor='val_accuracy', patience=steps//10, restore_best_weights=True)
        #def getMC(targetType, modelPath):
            #if targetType=="binary":
                #return ModelCheckpoint(filepath=modelPath, monitor='val_binary_accuracy')
            #elif targetType=="regression":
                #return ModelCheckpoint(filepath=modelPath, monitor='val_accuracy')
        def getLRS(decrease):
            def scheduler(epoche, lr):
                if epoche==0:
                    return lr*decrease
                else:
                    return lr
            return LearningRateScheduler(scheduler)
        def restore(model, modelPath, valPredictors, valTargets):
            temp = load_model(modelPath)
            tempAccuracy = temp.evaluate(valPredictors, valTargets)[1]
            modelAccuracy = model.evaluate(valPredictors, valTargets)[1]
            if tempAccuracy>modelAccuracy:
                return temp
            else:
                model.save(modelPath)
                return model
        modelPath = path/(name+'.h5')
        model.fit(x=predictors, y=targets, epochs=steps, batch_size=batchSize, callbacks=[getES(targetType, steps//4)], validation_data=(valPredictors, valTargets))
        print(steps, model.evaluate(valPredictors, valTargets)[1])
        model.save(modelPath)
        steps = round(steps*decrease)
        while steps>10:
            model = restore(model, modelPath, valPredictors, valTargets)
            model.fit(x=predictors, y=targets, epochs=steps, batch_size=batchSize, callbacks=[getES(targetType, steps//4), getLRS(0.5)], validation_data=(valPredictors, valTargets))
            print(steps, model.evaluate(valPredictors, valTargets)[1])
            steps = round(steps*decrease)
        model = restore(model, modelPath, valPredictors, valTargets)
        return model.evaluate(valPredictors, valTargets)[1]
    def modelFromBlueprintString(predictors, targets, targetType, blueprint):
        blueprint = blueprint.replace("]","").replace(" ","").replace("'","").split("[")
        optimizer = blueprint[0]
        size = list(map(int ,blueprint[1].split(",")))
        activations = blueprint[2].split(",")
        drops = list(map(float, blueprint[3].split(",")))
        return getModel(predictors, targets, targetType, size, activations, drops, optimizer)
    batchSize = getBatchSize(targets.index.size)
    length = getLength(predictors, valPredictors, targets, valTargets, targetType, batchSize)
    print(logModels(path, name, modelLog))
    size = getSize(predictors, valPredictors, targets, valTargets, targetType, batchSize, length)
    print(logModels(path, name, modelLog))
    activations = getActivations(predictors, valPredictors, targets, valTargets, targetType, batchSize, size)
    print(logModels(path, name, modelLog))
    drops = getDrops(predictors, valPredictors, targets, valTargets, targetType, batchSize, size, activations)
    print(logModels(path, name, modelLog))
    optimizer = getOptimizer(predictors, targets, targetType, batchSize, size, activations, drops)
    logModelBlueprint = logModels(path, name, modelLog)['model']
    print(logModelBlueprint)
    print(optimizer, size, activations, drops)
    model = getModel(predictors, targets, targetType, size, activations, drops, getOptimzerInstance(optimizer))
    logModel = modelFromBlueprintString(predictors, targets, targetType, logModelBlueprint)
    model = checkModel(model, logModel, predictors, valPredictors, targets, valTargets, targetType, batchSize)
    return training(path, name, model, predictors, valPredictors, targets, valTargets, targetType, batchSize)

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

def getData(data, dataType, frame, cols=[]):
    training = data['training'][dataType]
    validation = data['validation'][dataType]
    if not cols:
        return (training[frame], validation[frame])
    else:
        return (training[frame][cols], validation[frame][cols])

def initTraining(path, data, predictors):
    name = predictors
    if predictors.find('Trainer')>-1:
        predictors = getData(data, 'targets', predictors)
        path = path/'Models'/'Trainers'
    else:
        predictors = getData(data, 'predictors', predictors)
        path = path/'Models'
    homeAccuracy = deepTrain(path = path,
        name        = name+"_home",
        predictors  = predictors,
        targets     = getData(data, 'targets', 'targets', ['Home: Win']))
    visitingAccuracy = deepTrain(path = path,
        name        = name+"_visiting",
        predictors  = predictors,
        targets     = getData(data, 'targets', 'targets', ['Visiting: Win']))
    return (name, homeAccuracy, visitingAccuracy)

path = Path(__file__).parent.absolute()/'Learning'
createData(path)
data = loadData(path)
path = path/'Deep Training'
initGPU()
print(initTraining(path, data, 'others'))
#accuracies = []
#accuracies.append(initTraining(path, data, 'scoreTrainers'))
#accuracies.append(initTraining(path, data, 'performanceTrainers'))
#accuracies.append(initTraining(path, data, 'predictors'))
#accuracies.append(initTraining(path, data, 'ratioTrainees'))
#accuracies.append(initTraining(path, data, 'scoreTrainees'))
#accuracies.append(initTraining(path, data, 'scoreVersusTrainees'))
#accuracies.append(initTraining(path, data, 'trainees'))
#accuracies.append(initTraining(path, data, 'versusRatioTrainees'))
#print(accuracies)
#pd.DataFrame(accuracies, columns=['name','home accuracy', 'visiting accuracy']).to_csv(path/'accuracies.csv', index=False)