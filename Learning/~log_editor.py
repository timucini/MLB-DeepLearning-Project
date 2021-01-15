import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model
#enviroment settings
path = Path(__file__).parent.absolute()/'Deep Training'
name_data = 'none_'#''
metric = 'binary_accuracy'
minimise = False
#parameter settings
model_keys = ['optimizer','layers','activations','dropouts']
blueprint_keys = ['predictors','identifier']+model_keys
#log settings
log_keys = ['timestamp']+blueprint_keys+['dimensions','length','nodes','loss',metric,'time','epochs']
sort_fields = [metric, 'loss', 'epochs', 'nodes', 'time']
sort_conditions = [minimise, True, True, True, True]
predictor_log_path = path/'Logs'/(name_data+'predictor_evaluation_log.csv')
parameter_log_path = path/'Logs'/(name_data+'parameter_evaluation_log.csv')
#model settings
models_path = path/'Models'
#data settings
data_path = path/'Data'
targets_name = 'None_Targets.csv'
predictors_name = 'None_Predictors.csv'
targets_columns = ['Home: Win','Visiting: Win']
predictors_columns = None
targets_index = False
predictors_index = False
#data enviroment
targets = pd.read_csv(data_path/targets_name, usecols=targets_columns, index_col=targets_index)
predictors = pd.read_csv(data_path/predictors_name, usecols=predictors_columns, index_col=predictors_index)
data_date = datetime.fromtimestamp((data_path/'Validation'/(name_data+'validation_targets.csv')).stat().st_mtime)
validation_targets = pd.read_csv(data_path/'Validation'/(name_data+'validation_targets.csv'), index_col=targets_index)
validation_predictors = pd.read_csv(data_path/'Validation'/(name_data+'validation_predictors.csv'), index_col=predictors_index)
training_targets = pd.read_csv(data_path/'Training'/(name_data+'training_targets.csv'), index_col=targets_index)
training_predictors = pd.read_csv(data_path/'Training'/(name_data+'training_predictors.csv'), index_col=predictors_index)
#gpu settings
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
#functions
def load_log(log_path):
    #enviroment
    log = pd.read_csv(log_path, index_col=False)
    #procedure
    if log.empty:
        return log
    for column in log.loc[:,log.dtypes==object]:
        if (log[column][0].find('[')>-1 and log[column][0].find(']')>-1):
            log[column] = log[column].str.replace("'",'').str.replace(', ',',').str.replace('[','').str.replace(']','').str.split(',')
            if column=='layers' or column=='dropouts':
                newCol = []
                for element in log[column].tolist():
                    newElement = []
                    for value in element:
                        newElement.append(int(value))
                    newCol.append(newElement)
                log[column] = pd.Series(newCol)
    return log
def change_identifier(log_path, get_identifier_function):
    #enviroment
    log = load_log(log_path)
    #procedure
    print('Old:')
    print(log[['predictors','identifier']])
    log = log.to_dict('records')
    for entry in log:
        entry['identifier'] = get_identifier_function(entry['predictors'])
    log = pd.DataFrame(log)
    print('New:')
    print(log[['predictors','identifier']])
    return log
def find_duplicates():
    #enviroment
    predictor_log = load_log(predictor_log_path)
    #procedure
    duplicates = predictor_log[predictor_log.duplicated(keep=False, subset='identifier')]
    return duplicates.sort_values(by=['identifier']+sort_fields, ascending=[True]+sort_conditions)
def drop_duplicates():
    #enviroment
    duplicates = find_duplicates().drop_duplicates(subset=['identifier'], keep='last')
    log = load_log(predictor_log_path)
    #procedure
    return log.drop(duplicates.index)
def find_best(n=1):
    #enviroment
    predictor_log = load_log(parameter_log_path)
    #procedure
    best = predictor_log.sort_values(by=sort_fields, ascending=sort_conditions)
    return best[:n]
def test_best():
    #enviroment
    best = find_best().to_dict('records')[0]
    #procedeure
    model_predictors = predictors[best['predictors']]
    model_file = best['identifier']+'.h5'
    model = load_model(models_path/model_file)
    predictions = pd.DataFrame(model.predict(model_predictors), columns=targets.columns)
    evaluation_frame = pd.merge(targets, predictions, how='left', left_index=True, right_index=True, suffixes=('',' prediction'))
    return evaluation_frame
#procedure
print(load_log(predictor_log_path))
print(find_duplicates())
print(find_best())
#drop_duplicates().to_csv(predictor_log_path, index=False)
#print(test_best())
def get_identifier(predictor_sample):
    #enviroment
    identifier = 0
    #functions
    def numberficate(string):
        #enviroment
        value = 0
        #procedure
        for index, char in enumerate(string):
            value = value+(index+1)*31*ord(char)*113*len(string)*271
        return value
    #procedure
    for predictor in predictor_sample:
        identifier = identifier+numberficate(predictor)
    preface = str(len(predictor_sample)).zfill(2)+'D'
    return (preface+(str(identifier).zfill(16-len(preface))))[:16]
#print(change_identifier(predictor_log_path, get_identifier))