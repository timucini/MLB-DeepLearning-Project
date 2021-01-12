import pandas as pd
from pathlib import Path
#enviroment settings
path = Path(__file__).parent.absolute()/'Deep Training'/'Logs'
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
#data enviroment
parameter_log = pd.read_csv(parameter_log_path, index_col=False)
predictor_log = pd.read_csv(predictor_log_path, index_col=False)