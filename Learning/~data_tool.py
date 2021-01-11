import pandas as pd
from pathlib import Path
#enviroment settings
path = Path(__file__).parent.absolute()/'Deep Training'/'Data'
targets_name = 'None_Targets.csv'
predictors_name = 'None_Predictors.csv'
targets_columns = ['Home: Win','Visiting: Win']#None
predictors_columns = None
targets_index = False
predictors_index = False
spitt = 1/20
#centration settings
center_by = 'Home: Win'#False
fill = False
seed = 1337
#expanded settings
mix_data = True
save_data = True#False
name_data = 'none_'
print_data = True
#functions
def flatter(positives_index, negetives_index):
    mixture_index = []
    for index in range(len(positives_index)):
        mixture_index.append(positives_index[index])
        mixture_index.append(negetives_index[index])
    return mixture_index
def center(positives, negatives):
    diff = len(positives)-len(negatives)
    if diff<0:
        if fill:
            positives = positives.append(positives.sample(abs(diff), replace=True, random_state=seed))
        else:
            negatives = negatives.sample(len(positives))
    elif diff>0:
        if fill:
            negatives = negatives.append(negatives.sample(abs(diff), replace=True, random_state=seed))
        else:
            positives = positives.sample(len(negatives))
    return positives.sort_index().index, negatives.sort_index().index
def save(validation_targets, validation_predictors, training_targets, training_predictors):
    validation_targets.to_csv(path/'Validation'/(name_data+'validation_targets.csv'), index=targets_index)
    validation_predictors.to_csv(path/'Validation'/(name_data+'validation_predictors.csv'), index=predictors_index)
    training_targets.to_csv(path/'Training'/(name_data+'training_targets.csv'), index=targets_index)
    training_predictors.to_csv(path/'Training'/(name_data+'training_predictors.csv'), index=predictors_index)
def printing(validation_targets, validation_predictors, training_targets, training_predictors):
    print('validation_targets')
    print(validation_targets)
    print('validation_predictors')
    print(validation_predictors)
    print('training_targets')
    print(training_targets)
    print('training_predictors')
    print(training_predictors)
#procedure
targets = pd.read_csv(path/targets_name, usecols=targets_columns, index_col=targets_index)
predictors = pd.read_csv(path/predictors_name, usecols=predictors_columns, index_col=predictors_index)
splitt_size = round(len(targets)*spitt)
if center_by:
    positives = targets[center_by][targets[center_by]]
    negatives = targets[center_by][~targets[center_by]]
    validation_positives = positives.sample(splitt_size//2).sort_index().index
    validation_negatives = negatives.sample(splitt_size//2).sort_index().index
    training_positives, training_negatives = center(positives.drop(validation_positives), negatives.drop(validation_negatives))
    if mix_data:
        validation_index = flatter(validation_positives, validation_negatives)
        training_index = flatter(training_positives, training_negatives)
    else:
        validation_index = sorted(validation_positives.tolist()+validation_negatives.tolist())
        training_index = sorted(positives.tolist()+negatives.tolist())
else:
    validation_index = targets.sample(splitt_size).index
    training_index = targets.drop(validation_index).index
validation_targets = targets.loc[validation_index]
validation_predictors = predictors.loc[validation_index]
training_targets = targets.loc[training_index]
training_predictors = predictors.loc[training_index]
if print_data:
    printing(validation_targets, validation_predictors, training_targets, training_predictors)
if save_data:
    save(validation_targets, validation_predictors, training_targets, training_predictors)