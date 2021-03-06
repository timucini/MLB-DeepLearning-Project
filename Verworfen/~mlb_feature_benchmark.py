import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def bench(trainTargets, trainPredictors, validationTargets, validationPredictors, size=(16,16), prediction="binary", dropout=True):
    bench = Sequential()
    bench.add(Dense(size[1], input_shape=(trainPredictors.columns.size,), activation='relu'))
    for i in range(size[0]-1):
        bench.add(Dense(size[1], activation='relu'))
        if(i%2==0)&(dropout):
            bench.add(Dropout(0.1))
    bench.add(BatchNormalization())
    if prediction=="binary":
        bench.add(Dense(1, activation='sigmoid'))
        bench.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    elif prediction=="regression":
        bench.add(Dense(2))
        bench.compile(optimizer='adam', loss='MSE', metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=((size[0]+size[1])//8))
    batch = ((size[0]+size[1])**2)//2
    epochs = size[0]+size[1]
    bench.fit(x=trainPredictors, y=trainTargets, epochs=epochs, batch_size=batch, callbacks=earlyStopping, validation_data=(validationPredictors, validationTargets))
    evaluation = bench.evaluate(validationPredictors, validationTargets)
    return evaluation[0], evaluation[1]

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

def getPredictors(path):
    return pd.read_csv(path/'Learning'/'Predictors.csv', index_col=False, dtype="float32")

def predictorsBench(path, validationSplit=0.1, maxDims=False):
    predictors = getPredictors(path)
    targets = getBinaryTargets(path)
    split = np.random.rand(len(targets))<=validationSplit
    dims = predictors.columns.size
    if (maxDims!=False) & (maxDims<=predictors.columns.size):
        dims = maxDims
    predictorSet = pd.DataFrame()
    for dim in range(1, dims+1):
        usedPredictors = []
        validationAccuracy = []
        validationLoss = []
        for predictor in predictors.columns:
            usedPredictors.append(predictor)
            tempSet = pd.concat([predictorSet, predictors[predictor]], axis=1)
            print(tempSet.columns)
            start = datetime.now()
            benchmark = bench(targets[~split], tempSet[~split], targets[split], tempSet[split])
            validationAccuracy.append(benchmark[1])
            validationLoss.append(benchmark[0])
            print(predictor, datetime.now()-start)
        frame = pd.DataFrame({"Predictors":usedPredictors,"Dimension":dim,"Accuracies":validationAccuracy,"Losses":validationLoss})
        frame['Score'] = frame['Accuracies']/frame['Losses']
        predictor = frame.at[frame['Score'].idxmax(),'Predictors']
        predictorSet = pd.concat([predictorSet, predictors[predictor]], axis=1)
        predictors.drop(columns=[predictor])
        frame.to_csv(path/'Learning'/'Benchmark'/(str(dim)+'_'+predictor.replace(':','')+'.csv'), index=False)

def validate(model, predictors, targets):
    frame = pd.DataFrame()
    frame['Targets'] = targets
    frame['Predictions'] = model.predict(predictors)>0.5
    return (frame['Targets']!=frame['Predictions']).sum()/frame.index.size

def getBinaryTargets(path):
    return pd.read_csv(path/'Learning'/'Targets.csv', dtype="float32", usecols=['Home: Win'])

path = Path(__file__).parent.absolute()
initGPU()
predictorsBench(path)