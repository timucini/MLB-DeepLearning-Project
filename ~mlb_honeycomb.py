import pandas as pd
import numpy as np
from pathlib import Path
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Average, LSTM
from tensorflow.keras.callbacks import EarlyStopping

path = Path
targets     = pd.read_csv(path+r'\_mlb_targets.csv', index_col=False)[-40000:]
predictors  = pd.read_csv(path+r'\_mlb_predictors.csv', index_col=False, dtype="float32")[-40000:]

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

print("Data loaded")
batch_size = 500
#for i in range((targets.index.size//2)+1, 2, -1):
#    if((targets.index.size) % i == 0):
#        batch_size = i
#        break
print("Building Model")
x = predictors.columns.size*2
honeycomb = Sequential()
honeycomb.add(Dense(x, input_shape=(predictors.columns.size,), activation='relu'))
s = 204
while (x-s)>0:
    x = x-s
    honeycomb.add(Dense(x, activation='relu'))
    if(x%2==0):
        honeycomb.add(Dropout(x/(predictors.columns.size*4)))
honeycomb.add(BatchNormalization())
honeycomb.add(Dense(2))
#size_key = predictors.columns.size*2
#honeycomb = Sequential()
#honeycomb.add(Dense(size_key, input_shape=(predictors.columns.size,), activation='relu'))
#stepping = 10
#for x in range(1,1000):
#    a = math.exp(-(x*stepping**2/size_key))
#    b = math.cos(math.pi**2/stepping*x)*size_key/4+size_key*3/4
#    y = (a*b)//1
#    if y==1:
#        honeycomb.add(BatchNormalization())
#        honeycomb.add(Dense(1, activation='sigmoid'))
#        break
#    honeycomb.add(Dropout(a/2))
#    honeycomb.add(Dense(y, activation='relu'))

honeycomb.compile(optimizer='adam', loss='MSE', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
print(honeycomb.summary())
print("Start training")
honeycomb.fit(predictors, targets[['Visiting score','Home score']], epochs=batch_size, batch_size=batch_size, validation_split=0.15, callbacks=[early_stopping])
sampleSelectors = np.random.choice([True,False], size=targets.index.size)
print(sampleSelectors)
sampleTargets = targets[sampleSelectors]
pred = honeycomb.predict(predictors[sampleSelectors])
sampleTargets['predictions'] = pred[:,0]<pred[:,1]
sampleTargets['diff'] = sampleTargets['Home team win']!=sampleTargets['predictions']
print(sampleTargets)
print(sampleTargets['diff'].sum(), sampleTargets.index.size)