from tkinter import *
import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

root = Tk()
root.title('MLB Predictor')
root.geometry("500x400")


def selectedNL():
    homeTeam = clickedHome.get()
    visitingTeam = clickedVisiting.get()
    print(homeTeam)
    print(visitingTeam)
    predictNL(homeTeam, visitingTeam)

def selectedAL():
    homeTeam = clickedHomeAL.get()
    visitingTeam = clickedVisitingAL.get()
    print(homeTeam)
    print(visitingTeam)
    predictAL(homeTeam, visitingTeam)

def predictNL(homeId, visitingId):
    blueprintColumns = ['Visiting: Pythagorean expectation ratio', 'Home: Pythagorean expectation versus ratio',
                        'League Diffrence', 'Visiting: Odd ratio', 'Home: Team - Win rate',
                        'Home: Pitcher - Homeruns per game', 'Visiting: Team - Pythagorean expectation',
                        'Home: Pitcher - Saves per game', 'Home: Pitcher - Shutouts per game',
                        'Visiting: Pitcher - Saves per game', 'Home: Pythagorean expectation ratio',
                        'Home: Win ratio', 'Visiting: Team - Win rate']
    targets_columns = ['Home: Win', 'Visiting: Win']
    pathModel = '../Learning/Deep Training/Models/13D1956836164396.h5'
    data_folder = Path('./FrontendData')
    target_data = data_folder / "None_Targets_Frontend.csv"
    predictor_data = data_folder / "None_Predictors_Frontend.csv"
    df_targets = pd.read_csv(target_data)
    df_predictors = pd.read_csv(predictor_data)
    mergeData = pd.concat([df_targets, df_predictors], axis=1)
    matches = mergeData[(mergeData['Home: Team'] == homeId) & (mergeData['Visiting: Team'] == visitingId)]
    idx = matches.last_valid_index()
    model = keras.models.load_model(pathModel)
    lastStats = matches.loc[[idx]]
    relevantPrediction = lastStats[blueprintColumns]
    predictions = pd.DataFrame(model.predict(relevantPrediction), columns=targets_columns)
    print(predictions)
    if predictions['Home: Win'].values > predictions['Visiting: Win'].values:
        stringWinner = "Winner : Home"
    else:
        stringWinner = "Winner: Visiting"
    LabelWinner = Label(root, text=stringWinner).grid(row=4, column=1, padx='0', pady='0', sticky='ew')
    labelHome = "Value Home: " + str(predictions['Home: Win'].values)
    LabelHome = Label(root, text=labelHome).grid(row=5, column=1, padx='0', pady='0', sticky='ew')
    labelVisiting = "Value Visiting: " + str(predictions['Visiting: Win'].values)
    LabelVisiting = Label(root, text=labelVisiting).grid(row=6, column=1, padx='0', pady='0', sticky='ew')

def predictAL(homeId, visitingId):
    blueprintColumns = ['Visiting: Pythagorean expectation ratio', 'Home: Pythagorean expectation versus ratio',
                        'League Diffrence', 'Visiting: Odd ratio', 'Home: Team - Win rate',
                        'Home: Pitcher - Homeruns per game', 'Visiting: Team - Pythagorean expectation',
                        'Home: Pitcher - Saves per game', 'Home: Pitcher - Shutouts per game',
                        'Visiting: Pitcher - Saves per game', 'Home: Pythagorean expectation ratio',
                        'Home: Win ratio', 'Visiting: Team - Win rate']
    targets_columns = ['Home: Win', 'Visiting: Win']
    pathModel = '../Learning/Deep Training/Models/13D1956836164396.h5'
    data_folder = Path('./FrontendData')
    target_data = data_folder / "None_Targets_Frontend.csv"
    predictor_data = data_folder / "None_Predictors_Frontend.csv"
    df_targets = pd.read_csv(target_data)
    df_predictors = pd.read_csv(predictor_data)
    mergeData = pd.concat([df_targets, df_predictors], axis=1)
    matches = mergeData[(mergeData['Home: Team'] == homeId) & (mergeData['Visiting: Team'] == visitingId)]
    idx = matches.last_valid_index()
    model = keras.models.load_model(pathModel)
    lastStats = matches.loc[[idx]]
    relevantPrediction = lastStats[blueprintColumns]
    predictions = pd.DataFrame(model.predict(relevantPrediction), columns=targets_columns)
    print(predictions)
    if predictions['Home: Win'].values > predictions['Visiting: Win'].values:
        stringWinner = "Winner : Home"
    else:
        stringWinner = "Winner: Visiting"
    LabelWinner = Label(root, text=stringWinner).grid(row=4, column=2, padx='0', pady='0', sticky='ew')
    labelHome = "Value Home: " + str(predictions['Home: Win'].values)
    LabelHome = Label(root, text=labelHome).grid(row=5, column=2, padx='0', pady='0', sticky='ew')
    labelVisiting = "Value Visiting: " + str(predictions['Visiting: Win'].values)
    LabelVisiting = Label(root, text=labelVisiting).grid(row=6, column=2, padx='0', pady='0', sticky='ew')


options = ['ARI', 'ATL', 'LAN', 'MIA', 'NYN', 'SDN', 'CIN', 'MIL',
           'PIT', 'SFN', 'PHI', 'SLN', 'WAS', 'COL', 'CHN']

optionsAL = ['BAL', 'KCA', 'OAK', 'SEA', 'TEX', 'TOR', 'DET', 'TBA',
             'ANA', 'HOU',  'NYA', 'CHA', 'MIN', 'CLE', 'BOS']

print(len(options))
print(len(optionsAL))


clickedHome = StringVar()
clickedHome.set(options[0])

clickedVisiting = StringVar()
clickedVisiting.set(options[0])

clickedHomeAL = StringVar()
clickedHomeAL.set(optionsAL[0])

clickedVisitingAL = StringVar()
clickedVisitingAL.set(optionsAL[0])

League = Label(root, text="League: ").grid(row=0, column=0, padx='10', pady='10', sticky='ew')
HomeTeam = Label(root, text="HomeTeam ID: ").grid(row=1, column=0, padx='10', pady='10', sticky='ew')
VisitingTeam = Label(root, text="Visiting ID: ").grid(row=2, column=0, padx='10', pady='10', sticky='ew')
Prediction = Label(root, text="Prediction: ").grid(row=3, column=0, padx='10', pady='10', sticky='ew')


LeagueTitle1 = Label(root, text="National League").grid(row=0, column=1, padx='10', pady='10', sticky='ew')
dropHome = OptionMenu(root, clickedHome, *options)
dropHome.grid(row=1, column=1, padx='10', pady='10', sticky='ew')

dropVisiting = OptionMenu(root, clickedVisiting, *options)
dropVisiting.grid(row=2, column=1, padx='10', pady='10', sticky='ew')

predictButton = Button(root, text="Predict Winner", command=selectedNL)
predictButton.grid(row=3, column=1, padx='10', pady='10', sticky='ew')

LeagueTitle2 = Label(root, text="American League").grid(row=0, column=2, padx='10', pady='10', sticky='ew')
dropHomeAL = OptionMenu(root, clickedHomeAL, *optionsAL)
dropHomeAL.grid(row=1, column=2, padx='10', pady='10', sticky='ew')

dropVisitingAL = OptionMenu(root, clickedVisitingAL, *optionsAL)
dropVisitingAL.grid(row=2, column=2, padx='10', pady='10', sticky='ew')

predictButtonAL = Button(root, text="Predict Winner", command=selectedNL)
predictButtonAL.grid(row=3, column=1, padx='10', pady='10', sticky='ew')

predictButtonAL = Button(root, text="Predict Winner", command=selectedAL)
predictButtonAL.grid(row=3, column=2, padx='10', pady='10', sticky='ew')

root.mainloop()