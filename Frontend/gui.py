from tkinter import *
import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

root = Tk()
root.title('MLB Predictor')
root.geometry("400x300")

targets_columns = ['Home: Win','Visiting: Win']


def selected():
    homeTeam = clickedHome.get()
    visitingTeam = clickedVisiting.get()
    print(homeTeam)
    print(visitingTeam)
    predict(homeTeam, visitingTeam)

def predict(homeId, visitingId):
    blueprintColumns = ['Visiting: Team - Season Performance', 'Home: Pitcher - Strikeouts per walk']
    targets_columns = ['Home: Win', 'Visiting: Win']
    pathModel = '../Learning/Deep Training/Models/02D3945947641755.h5'
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
    LabelWinner = Label(root, text=stringWinner).pack()
    labelHome = "Value Home: " + str(predictions['Home: Win'].values)
    LabelHome = Label(root, text=labelHome).pack()
    labelVisiting = "Value Visiting: " + str(predictions['Visiting: Win'].values)
    LabelVisiting = Label(root, text=labelVisiting).pack()


options = ['ARI', 'ATL', 'LAN', 'MIA', 'NYN', 'SDN', 'BAL', 'KCA', 'OAK', 'SEA', 'TEX', 'TOR', 'CIN', 'DET', 'TBA',
           'MIL', 'PIT', 'ANA', 'HOU', 'SFN', 'NYA', 'PHI', 'SLN', 'WAS', 'CHA', 'MIN', 'COL', 'CLE', 'BOS', 'CHN']

clickedHome = StringVar()
clickedHome.set(options[0])

clickedVisiting = StringVar()
clickedVisiting.set(options[0])

LabelHome = Label(root, text="Home Team ID").pack()
dropHome = OptionMenu(root, clickedHome, *options)
dropHome.pack()

LabelPlace = Label(root, text="").pack()

LabelVisiting = Label(root, text="Visiting Team ID").pack()
dropVisiting = OptionMenu(root, clickedVisiting, *options)
dropVisiting.pack()

LabelPlace2 = Label(root, text="").pack()

predictButton = Button(root, text="Predict Winner", command=selected)
predictButton.pack()

root.mainloop()