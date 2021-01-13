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
def find_duplicates():
    #enviroment
    predictor_log = load_log(predictor_log_path)
    #procedure
    duplicates = predictor_log[predictor_log.duplicated(keep=False, subset='identifier')]
    return duplicates
def find_best(n=1):
    #enviroment
    predictor_log = load_log(predictor_log_path)
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
    #specific
    evaluation_frame['Win'] = evaluation_frame['Home: Win']>evaluation_frame['Visiting: Win']
    evaluation_frame['Win prediction'] = evaluation_frame['Home: Win prediction']>evaluation_frame['Visiting: Win prediction']
    evaluation_frame = evaluation_frame[['Win', 'Win prediction']]
    evaluation_frame['Diffrence'] = False==(evaluation_frame['Win']==evaluation_frame['Win prediction'])
    print(evaluation_frame)
    print(sum(evaluation_frame['Diffrence']), len(evaluation_frame))
#procedure
print(find_duplicates())
print(find_best())
test_best()