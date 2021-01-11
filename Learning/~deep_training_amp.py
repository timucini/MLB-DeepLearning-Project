import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
#enviroment settings
path = Path(__file__).parent.absolute()/'Deep Training'
targets_index = False
predictors_index = False
metric = 'binary_accuracy'
minimise = False
#expanded settings
name_data = 'none_'#''
#gpu settings
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
#parameter settings
blueprint_keys = ['predictors','identifier','optimizer','layers','activations','dropouts']
#log settings
log_keys = ['timestamp']+blueprint_keys+['dimensions','length','nodes','loss',metric,'time','epochs']
sort_fields = [metric, 'loss', 'epochs', 'nodes', 'time']
sort_conditions = [minimise, True, True, True, True]
predictor_log_path = path/'Logs'/(name_data+'predictor_evaluation_log.csv')
parameter_log_path = path/'Logs'/(name_data+'parameter_evaluation_log.csv')
#model settings
models_path = path/'Models'
#data enviroment
data_date = datetime.fromtimestamp((path/'Data'/'Validation'/(name_data+'validation_targets.csv')).stat().st_mtime)
validation_targets = pd.read_csv(path/'Data'/'Validation'/(name_data+'validation_targets.csv'), index_col=targets_index)
validation_predictors = pd.read_csv(path/'Data'/'Validation'/(name_data+'validation_predictors.csv'), index_col=predictors_index)
training_targets = pd.read_csv(path/'Data'/'Training'/(name_data+'training_targets.csv'), index_col=targets_index)
training_predictors = pd.read_csv(path/'Data'/'Training'/(name_data+'training_predictors.csv'), index_col=predictors_index)
batch_size = len(training_targets)//10
#functions
def training(blueprint, maximial_epochs, start_learning_rate=0.1, stop_learning_rate=0.1, patience=False):
    #enviroment
    epoch_range = range(maximial_epochs, 0, -round(maximial_epochs/(start_learning_rate/stop_learning_rate)**0.7))
    learning_rate_decrease = (start_learning_rate/stop_learning_rate)**(1/max((len(epoch_range)-1,1)))
    learning_rate = start_learning_rate
    trained = []
    times = []
    #functions
    def get_model():
        #enviroment
        name = blueprint['identifier']
        #procedure
        model = Sequential(name=name)
        model.add(Input(shape=(len(blueprint['predictors']),), name=('I'+name)))
        for index, nodes in enumerate(blueprint['layers']):
            activation = blueprint['activations'][index]
            if activation=='None':
                activation = None
            model.add(Dense(nodes, activation, kernel_initializer='ones', name=('D'+str(index)+'_'+name)))
            if blueprint['dropouts'][index]>0:
                model.add(Dropout(blueprint['dropouts'][index]/nodes, name=('O'+str(index)+'_'+name)))
        model.add(BatchNormalization(name=('B'+name)))
        if (validation_targets.dtypes==bool).any():
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            activation = None
            loss = 'MSE'
        model.add(Dense(len(validation_targets.columns), activation=activation, kernel_initializer='ones', name=('T'+name)))
        model.compile(optimizer=blueprint['optimizer'], loss=loss, metrics=[metric])
        return model
    def evaluating(model, patience, epochs):
        monitor = EarlyStopping(monitor=('val_'+metric),restore_best_weights=True, patience=patience)
        start = datetime.now()
        history = model.fit(training_predictors[blueprint['predictors']], training_targets, batch_size, epochs, 0, [monitor], validation_data=(validation_predictors[blueprint['predictors']], validation_targets))
        time = (datetime.now()-start).total_seconds()
        metrics = model.evaluate(validation_predictors, validation_targets, return_dict=True, verbose=0)
        metrics['time'] = time
        metrics['epochs'] = len(history.history[metric])
        return metrics
    def to_row(metrics):
        #enviroment
        row = {'timestamp':datetime.now()}
        #procedure
        row.update(blueprint.copy())
        row['dimensions'] = len(blueprint['predictors'])
        row['length'] = len(blueprint['layers'])
        row['nodes'] = sum(blueprint['layers'])
        row.update(metrics.copy())
        return row.copy()
    #procedure
    model = get_model()
    for epochs in epoch_range:
        model.optimizer.lr = learning_rate
        backup = model.get_weights()
        image = model.evaluate(validation_predictors, validation_targets, return_dict=True, verbose=0)
        if patience:
            metrics = evaluating(model, patience, epochs)
        else:
            metrics = evaluating(model, epochs, epochs)
        if image[metric]<=metrics[metric]:
            trained.append(metrics['epochs'])
            times.append(metrics['time'])
            model.save(path/'Models'/(blueprint['identifier']+'.h5'))
        else:
            model.set_weights(backup)
        learning_rate = learning_rate*learning_rate_decrease
    metrics = model.evaluate(validation_predictors, validation_targets, return_dict=True, verbose=0)
    metrics['time'] = sum(times)
    metrics['epochs'] = sum(trained)
    return to_row(metrics)
def predictor_evaluation(start_bias=0.5, start_nodes=10, minimal_node_increase=3, epsilon=8, worker=0):
    #enviroment
    start_learning_rate = 0.1
    stop_learning_rate = 0.01
    parameter_patience = 20
    node_multiplier = 10
    #functions
    def trace_predictor(used_predictors, bias, nodes):
        #functions
        def try_predictors(try_at, bias, nodes):
            #enviroment
            tries = []
            #functions
            def get_identifier(predictor_sample):
                #enviroment
                identifier = 0
                preface = 0
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
                preface = chr(preface//len(predictor_sample))+str(len(predictor_sample)).zfill(2)+'D'
                return (preface+(str(identifier).zfill(16-len(preface))))[:16]
            def append_tries(result, tries):
                #procedure
                if result[metric]>=bias:
                    tries.append(result)
                    return pd.DataFrame(tries).sort_values(by=sort_fields, ascending=sort_conditions).to_dict('records')[:epsilon]
                return tries
            def to_log(row):
                #functions
                def to_string():
                    s = ''
                    for value in row.values():
                        if isinstance(value, list):
                            v = '"'+str(value)+'",'
                        else:
                            v = str(value)+','
                        s = s + v
                    return s[0:-1]
                #procedure
                with open(predictor_log_path,'a') as log:
                    log.write('\n')
                    log.write(to_string())
            def check(identifier):
                #functions
                def load_log():
                    #enviroment
                    log = pd.read_csv(predictor_log_path, index_col=False)
                    #procedure
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
                def get_latest(backlog):
                    #enviroment
                    best = backlog.sort_values(by=sort_fields, ascending=sort_conditions)
                    blueprint = best[blueprint_keys].to_dict('records')[0]
                    #procedure
                    for model_path in models_path.glob('*.h5'):
                        if model_path.stem==identifier:
                            if datetime.fromtimestamp(model_path.stat().st_mtime)<=data_date:
                                row = training(blueprint, nodes*node_multiplier, stop_learning_rate, stop_learning_rate, parameter_patience)
                                return row
                            else:
                                return best.to_dict('records')[0]
                    row = training(blueprint, nodes*node_multiplier, stop_learning_rate, stop_learning_rate, parameter_patience)
                    return row
                #procedure
                backlog = load_log()
                backlog = backlog[backlog['identifier']=='identifier']
                if backlog.empty:
                    return {}
                else:
                    return get_latest(backlog)
            #procedure
            for predictor in validation_predictors.drop(columns=used_predictors).columns:

        #procedure
        print('Worker:    ', worker)
        print('Predictors:', used_predictors)
        print('Bias:      ', bias)
        print('Nodes:     ', nodes, '\n')
        tries = try_predictors(used_predictors, bias, nodes)
        if not tries:
            print('Dead end @:', used_predictors, '\n')
            return
        for predictor_combination in tries:
            dimension_lenght = len(predictor_combination['predictors'])
            nodes = max((dimension_lenght*minimal_node_increase)+start_nodes, (predictor_combination['nodes']/dimension_lenght)*(dimension_lenght+1))
            trace_predictor(predictor_combination['predictors'], predictor_combination[metric], round(nodes))
