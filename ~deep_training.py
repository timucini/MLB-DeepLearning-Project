import pandas as pd
import numpy as np
import random as rd
from pathlib import Path

import tensorflow as tf
import tensorflow.keras.optimizers as opt
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

def createData(path, uniform=False, uni_target=[], uni_predictors=[], uni_hot=True, maxSize=100000):
    def loadUniform(path, target, predictors, hot):
        dictonary = {}
        data = pd.read_csv(path, index_col=0)
        targetFrame = data[[target]]
        predictorFrame = pd.DataFrame()
        if hot:
            for column in uni_predictors:
                if data[column].dtype == "object":
                    predictorFrame = pd.concat([pd.get_dummies(data[column]), predictorFrame], axis=1)
                else:
                    predictorFrame[column] = data[column]
        else:
            predictorFrame = data[predictors]
        index = getSplit(targetFrame.iloc[:,0])
        dictonary['targets'] = (targetFrame.loc[index[0]], targetFrame.loc[index[1]])
        dictonary['predictors'] = (predictorFrame.loc[index[0]], predictorFrame.loc[index[1]])
        return dictonary

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

    def getCenteredIndex(target, seed=1337, maxSize=0):
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

    def getSplit(target, size=0.1, seed=1337):
        rd.seed(seed)
        split = rd.sample(list(target.index),round(size*target.index.size))
        split = getCenteredIndex(target[split], seed)
        unSplit = getCenteredIndex(target[list(set(target.index)-set(split))], seed)
        return (unSplit, split)

    def saveData(overDictonary, path):
        for dataType in overDictonary:
            for data in overDictonary[dataType]:
                for frame in overDictonary[dataType][data]:
                    overDictonary[dataType][data][frame].to_csv(path/'Deep Training'/dataType/data/(frame+".csv"), index = False)

    if uniform:
        return loadUniform(path, uni_target, uni_predictors, uni_hot)
    else:
        targets = loadTargets(path)
        predictors = loadPredictors(path)
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

def training(path, name, model, predictors, valPredictors, targets, valTargets, targetType, batchSize, steps=1000, decrease=0.9, lr=1, lrDecrease=0.32):
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
                lr = lr*decrease
                print("learning rate:", lr)
                return lr
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
    blueprint = model[1]
    model = model[0]
    K.set_value(model.optimizer.lr, lr)
    model.fit(x=predictors, y=targets, epochs=steps, batch_size=batchSize, callbacks=[getES(targetType, steps//4)], validation_data=(valPredictors, valTargets))
    model.save(modelPath)
    while steps>10:
        print(steps, blueprint, model.evaluate(valPredictors, valTargets)[1])
        steps = round(steps*decrease)
        model = restore(model, modelPath, valPredictors, valTargets)
        model.fit(x=predictors, y=targets, epochs=steps, batch_size=batchSize, callbacks=[getES(targetType, steps//2), getLRS(lrDecrease)], validation_data=(valPredictors, valTargets))
    model = restore(model, modelPath, valPredictors, valTargets)
    return name, blueprint, lr, model.evaluate(valPredictors, valTargets)[1]

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

def modelFromBlueprintString(predictors, targets, targetType, blueprint):
    blueprint = blueprint.replace("]","").replace(" ","").replace("'","").split("[")
    optimizer = blueprint[0]
    size = list(map(int ,blueprint[1].split(",")))
    activations = blueprint[2].split(",")
    drops = list(map(float, blueprint[3].split(",")))
    return getModel(predictors, targets, targetType, size, activations, drops, optimizer)

def deepTrain(predictors, targets, path, name, onPerform, targetType="binary"):
    def logModels(path, name, modelLog):
        frame = pd.DataFrame(modelLog)
        frame.to_csv(path/(name+"_modelLog.csv"), index=False)
        return frame.iloc[frame[onPerform].idxmax()]
    def checkModel(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, size, activations, drops, optimizer):
        model = getModel(predictors, targets, targetType, size, activations, drops, optimizer, returnBlueprint=True)
        logModelBlueprint = logModels(path, name, modelLog)['model']
        logModel = modelFromBlueprintString(predictors, targets, targetType, logModelBlueprint)
        model_accuracy = evaluatingTraining(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize, 100, 10)
        print("model_accuracy:", model[1], model_accuracy)
        log_model_accuracy = evaluatingTraining(logModel, predictors, valPredictors, targets, valTargets, targetType, batchSize, 100, 10)
        print("log_model_accuracy:", logModelBlueprint, log_model_accuracy)
        if log_model_accuracy>model_accuracy:
            return logModel, logModelBlueprint
        return getModel(predictors, targets, targetType, size, activations, drops, getOptimzerInstance(optimizer)), model[1]
    def getBatchSize(size, min=1000):
        for i in range((size//2)+1, 2, -1):
            if ((size % i)) == 0 and (size/i>min) and (size/i<size//3):
                return size//i
        return min
    def evaluatingTraining(model, predictors, valPredictors, targets, valTargets, targetType, batchSize, epochs, patience, lr=0.1):
        if targetType=="binary":
            monitor = EarlyStopping(monitor='val_binary_accuracy',restore_best_weights=True, patience=patience)
        elif targetType=="regression":
            monitor = EarlyStopping(monitor='val_accuracy',restore_best_weights=True, patience=patience)
        K.set_value(model.optimizer.lr, lr)
        model.fit(x=predictors, y=targets, epochs=epochs, batch_size=batchSize, callbacks=[monitor], validation_data=(valPredictors, valTargets))
        return model.evaluate(valPredictors, valTargets)[1]
    def getLength(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, standardActivation, standardWidth):
        metrics = []
        lengths = []
        minLength = 2
        maxLenght = predictors.columns.size*2
        for length in range(minLength, maxLenght+1):
            model = getModel(predictors, targets, targetType, [standardWidth]*length, [standardActivation]*length, [0]*length, returnBlueprint=True)
            lengths.append(length)
            metric = evaluatingTraining(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize, 250, round(40*(length/maxLenght))+10)
            metrics.append(metric)
            print(length, metric)
            modelLog['model'].append(model[1])
            modelLog[onPerform].append(metric)
        print(logModels(path, name, modelLog))
        return lengths[metrics.index(max(metrics))]
    def getSize(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, length, standardActivation):
        size = []
        for layer in range(1, length+1):
            widths = []
            metrics = []
            minWidth = 2
            maxWidth = predictors.columns.size*2
            for width in range(minWidth, maxWidth+1):
                model = getModel(predictors, targets, targetType, size+[width]+([minWidth]*(length-layer)), [standardActivation]*length, [0]*length, returnBlueprint=True)
                widths.append(width)
                metric = evaluatingTraining(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize, 100, 10)
                metrics.append(metric)
                print(layer, width, metric)
                modelLog['model'].append(model[1])
                modelLog[onPerform].append(metric)
            size.append(widths[metrics.index(max(metrics))])
        print(logModels(path, name, modelLog))
        return size
    def getDrops(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, size, standardActivation, activations=[]):
        drops = []
        if not activations:
            activations = [standardActivation]*len(size)
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
                modelLog[onPerform].append(metric)
            drops.append(tempDrops[metrics.index(max(metrics))])
        print(logModels(path, name, modelLog))
        return drops
    def getActivations(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, size, standardActivation='relu', drops=[]):
        possibilities = ['relu','tanh','selu','elu']
        activations = []
        if not drops:
            drops = [0]*len(size)
        for layer in range(1, len(size)+1):
            tempActivations = []
            metrics = []
            for activation in possibilities:
                model = getModel(predictors, targets, targetType, size, activations+[activation]+[standardActivation]*(len(size)-layer), drops, returnBlueprint=True)
                tempActivations.append(activation)
                metric = evaluatingTraining(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize, 100, 10)
                metrics.append(metric)
                print(layer, activation, metric)
                modelLog['model'].append(model[1])
                modelLog[onPerform].append(metric)
            activations.append(tempActivations[metrics.index(max(metrics))])
        print(logModels(path, name, modelLog))
        return activations
    def getOptimizer(predictors, targets, targetType, batchSize, path, name, modelLog, size, activations, drops, epochs=250):
        possibilities = ['sgd','rmsprop','adam','adadelta','adagrad','adamax','nadam']
        metrics = []
        for optimizer in possibilities:
            model = getModel(predictors, targets, targetType, size, activations, drops, optimizer=optimizer, returnBlueprint=True)
            metric = evaluatingTraining(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize, epochs, epochs//5)
            metrics.append(metric)
            print(optimizer, metric)
            modelLog['model'].append(model[1])
            modelLog[onPerform].append(metric)
        print(logModels(path, name, modelLog))
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
    def getLR(model, predictors, valPredictors, targets, valTargets, targetType, batchSize):
        lrs = [1.00,0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
        dcs = [0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.54,0.59,0.63,0.68,0.72,0.77,0.81,0.86,0.90]
        metrics = []
        for lr in lrs:
            metric = evaluatingTraining(model, predictors, valPredictors, targets, valTargets, targetType, batchSize, 100, 100, lr)
            metrics.append(metric)
            print(lr, metric)
        return lrs[metrics.index(max(metrics))], dcs[metrics.index(max(metrics))]
    predictors, valPredictors = predictors
    targets, valTargets = targets
    modelLog            = {'model':[],onPerform:[]}
    batchSize           = getBatchSize(targets.index.size)
    print(batchSize)    
    standardActivation  = getActivations(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, [predictors.columns.size])[0]
    standardWidth       = getSize(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, 1, standardActivation)[0]
    length              = getLength(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, standardActivation, standardWidth)
    size                = getSize(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, length, standardActivation)
    activations         = getActivations(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, size, standardActivation)
    drops               = getDrops(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, size, standardActivation, activations)
    optimizer           = getOptimizer(predictors, targets, targetType, batchSize, path, name, modelLog, size, activations, drops)
    model               = checkModel(predictors, valPredictors, targets, valTargets, targetType, batchSize, path, name, modelLog, size, activations, drops, optimizer)
    lr, lrDecrease      = getLR(model[0], predictors, valPredictors, targets, valTargets, targetType, batchSize)
    return training(path, name, model, predictors, valPredictors, targets, valTargets, targetType, batchSize, lr=lr, lrDecrease=lrDecrease)
    
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

def initTraining(path, data, predictors, onPerform):
    name = predictors
    if predictors.find('Trainer')>-1:
        predictors = getData(data, 'targets', predictors)
        path = path/'Models'/'Trainers'
    else:
        predictors = getData(data, 'predictors', predictors)
        path = path/'Models'
    home = deepTrain(path = path,
        name        = name+"_home",
        predictors  = predictors,
        targets     = getData(data, 'targets', 'targets', ['Home: Win']),
        onPerform   = onPerform)
    visiting = deepTrain(path = path,
        name        = name+"_visiting",
        predictors  = predictors,
        targets     = getData(data, 'targets', 'targets', ['Visiting: Win']),
        onPerform   = onPerform)
    return [home, visiting]

def standardTrainActivities(path, data, onPerform='accuracy'):
    onPerforms = []
    def routine(on, onPerforms):
        onPerforms = onPerforms + initTraining(path, data, on, onPerform)
        pd.DataFrame(onPerforms, columns=['name','blueprint','learning_rate',onPerform]).to_csv(path/(onPerform+'.csv'), index=False)
        print(onPerforms)
        return onPerforms
    onPerforms = routine('scoreTrainers', onPerforms)
    onPerforms = routine('performanceTrainers', onPerforms)
    onPerforms = routine('predictors', onPerforms)
    onPerforms = routine('ratioTrainees', onPerforms)
    onPerforms = routine('scoreTrainees', onPerforms)
    onPerforms = routine('scoreVersusTrainees', onPerforms)
    onPerforms = routine('trainees', onPerforms)
    onPerforms = routine('versusRatioTrainees', onPerforms)
    onPerforms = routine('others', onPerforms)

def restoreFromLog(path, name, predictors, targets, targetType, onPerform):
    log = pd.read_csv(path/(name+'_modelLog.csv'))
    log = log.iloc[log[onPerform].idxmax()]
    blueprint = log['model']
    onPerform = log[onPerform]
    return (modelFromBlueprintString(predictors, targets, targetType, blueprint), blueprint), onPerform

def checkAllModels(path, data, onPerform='accuracy', targetType="binary"):
    onPerforms = pd.read_csv(path/(onPerform+'.csv'), index_col=False)
    newPerforms = []
    for performer in range(onPerforms.index.size):
        performer = onPerforms.iloc[performer]
        if performer['name'].find('Trainer')>-1:
            predictors, valPredictors = getData(data, 'targets', performer['name'].split("_")[0])
            performerPath = path/'Models'/'Trainers'
        else:
            predictors, valPredictors = getData(data, 'predictors', performer['name'].split("_")[0])
            performerPath = path/'Models'
        if performer['name'].find('home')>-1:
            targets, valTargets = getData(data, 'targets', 'targets', ['Home: Win'])
        else:
            targets, valTargets = getData(data, 'targets', 'targets', ['Visiting: Win'])
        model, onPerformLog = restoreFromLog(performerPath, performer['name'], predictors, targets, targetType, onPerform)
        print(performer['name'], performer[onPerform], onPerformLog)
        if performer[onPerform]<onPerformLog:
            newPerforms.append(training(
                performerPath, performer['name'], model, predictors, valPredictors, targets, valTargets, targetType,
                batchSize=5000, steps=1000, decrease=0.9, lr=0.10, lrDecrease=0.50))
        else:
            newPerforms.append(list(performer))
    print(newPerforms)
    pd.DataFrame(newPerforms, columns=['name','blueprint','learning_rate',onPerform]).to_csv(path/(onPerform+'_new.csv'), index=False)

path = Path(__file__).parent.absolute()/'Learning'
#path = path.parent.parent.parent/'Data Mining'/'Projekt'/'dnn_set.csv'
#data = createData(path, uniform=True, uni_target='Buy_now', uni_predictors=[
#    "Departure_hour","route_abb","Price_In_Eur","Request_Hour","Hours_Before_departure","Request_Day","Departure_Day","Departure_Month"])
#initGPU()
#print(deepTrain(predictors=data['predictors'], targets=data['targets'], path=path.parent, name="flights"))
#createData(path)
data = loadData(path)
#print(data)
path = path/'Deep Training'
initGPU()
standardTrainActivities(path, data)
checkAllModels(path, data)