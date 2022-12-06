import pandas as pd
import os
import math
from datetime import datetime
import numpy as np
from numpy import argmax

# Importing the Keras libraries and packages
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sys

# path to the dataset
# file = 'Shaleeza.Dataset.v1.csv'
file = 'IoT Network Intrusion Dataset.csv'



def data_preprocessing(targets_others):
    dataset = pd.read_csv(file, error_bad_lines=False, low_memory=False)
    dataset = dataset.drop(['Flow_ID', 'Src_IP', 'Dst_IP', 'Dst_Port', 'Protocol'], axis=1)
    dataset = dataset.drop(['Timestamp'], axis=1)

    # contain only single values
    dataset = dataset.drop(
        ['Fwd_PSH_Flags', 'Fwd_URG_Flags', 'Fwd_Byts/b_Avg', 'Fwd_Pkts/b_Avg', 'Fwd_Blk_Rate_Avg', 'Bwd_Byts/b_Avg',
         'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg', 'Init_Fwd_Win_Byts', 'Fwd_Seg_Size_Min'], axis=1)

    dataset['Flow_Byts/s'] = dataset.round({'Flow_Byts/s': 2})

    dataset = dataset.drop(targets_others, axis=1)

    dataset = dataset.reset_index()
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(inplace=True)

    # correlation
    correlated_features = set()
    correlation_matrix = dataset.corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >= 0.7:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    dataset.drop(labels=correlated_features, axis=1, inplace=True)

    return dataset


def run_ANN(target_name, target_type_count, target_others, units_selection, batch_size_selection, nb_epoch_selection, optimizer_selection, activation_selection, activation_output_selection):
    dataset = data_preprocessing(target_others)

    X = dataset.drop(labels=target_name, axis=1)
    y = dataset[target_name]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    enc = OneHotEncoder(handle_unknown='ignore')
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    enc.fit(y_train)
    y_train = enc.transform(y_train).toarray()
    y_test = enc.transform(y_test).toarray()

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=units_selection, kernel_initializer='normal', activation=activation_selection))

    # Adding the second hidden layer
    classifier.add(Dense(units=units_selection, kernel_initializer='normal', activation=activation_selection))

    # Adding the output layer
    classifier.add(Dense(units=target_type_count, kernel_initializer='normal', activation=activation_output_selection))

    # Compiling the ANN
    classifier.compile(optimizer=optimizer_selection, loss='categorical_crossentropy', metrics=[tensorflow.keras.metrics.AUC()])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size=batch_size_selection, epochs=nb_epoch_selection, verbose=1)

    # Part 3 - Making predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)


    #Output result
    output_result = ''

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    output_result += 'CM:\n'
    output_result += str(cm)
    output_result += '\n'

    # Accuracy
    _, accuracy = classifier.evaluate(X_test, y_test, verbose=0)
    print(accuracy)
    output_result += 'Accuracy:' + str(accuracy)
    output_result += '\n'

    from sklearn.metrics import precision_recall_fscore_support as score

    precision, recall, fscore, support = score(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    for i in range(target_type_count):  # 2,5,9 classes
        print('label: {}, precision: {}, recall: {}, fscore: {}, support: {}'.format(i + 1, precision[i], recall[i],
                                                                                     fscore[i], support[i]))
        output_result += ('label: {}, precision: {}, recall: {}, fscore: {}, support: {}'.format(i + 1, precision[i], recall[i],
                                                                                     fscore[i], support[i]))
        output_result += '\n'

    precision, recall, fscore, support = score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
    macro_results = 'Macro: precision: {}, recall: {}, fscore: {}, support: {}'.format(precision, recall, fscore, support)
    print(macro_results)
    output_result += macro_results
    output_result += '\n'

    precision, recall, fscore, support = score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
    weighted_results = 'Weighted: precision: {}, recall: {}, fscore: {}, support: {}'.format(precision, recall, fscore, support)
    print(weighted_results)
    output_result += weighted_results
    output_result += '\n'

    return output_result

def parameter_selection(target_name, target_type_count, targets_others,parameters):
    # parameters = {'batch_size': [10, 100],
                  # 'nb_epoch': [100, 200],
                  # 'optimizer': ['adam', 'rmsprop', 'SGD', 'adamax', 'adagrad'],
                  # 'activation': ['relu', 'tanh', 'sigmoid', 'softplus', 'softmax'],
                  # 'activation_output': ['relu', 'tanh', 'sigmoid', 'softplus', 'softmax'],
                  # 'units': [100, 200]}

    for batch_size_selection in parameters['batch_size']:
        for nb_epoch_selection in parameters['nb_epoch']:
            for optimizer_selection in parameters['optimizer']:
                for activation_selection in parameters['activation']:
                    for activation_output_selection in parameters['activation_output']:
                        for units_selection in parameters['units']:
                            try:
                                print('units: ' + str(units_selection))
                                print('batch_size: ' + str(batch_size_selection))
                                print('nb_epoch: ' + str(nb_epoch_selection))
                                print('optimizer: ' + optimizer_selection)
                                print('activation: ' + activation_selection)
                                print('activation_output: ' + activation_output_selection)

                                output_result = run_ANN(target_name, target_type_count, targets_others, units_selection, batch_size_selection, nb_epoch_selection, optimizer_selection, activation_selection, activation_output_selection)
                                with open(target_name + '_log.txt', 'a') as f:
                                    f.write('\n======================================')
                                    f.write('\nunits: ' + str(units_selection))
                                    f.write('\nbatch_size: ' + str(batch_size_selection))
                                    f.write('\nnb_epoch: ' + str(nb_epoch_selection))
                                    f.write('\noptimizer: ' + optimizer_selection)
                                    f.write('\nactivation: ' + activation_selection)
                                    f.write('\nactivation_output: ' + activation_output_selection)

                                    f.write('\noutput result:\n')
                                    f.write(output_result)
                                    f.write('\n\n\n')
                            except Exception as e:
                                with open(target_name + '_error.txt', 'a') as f:
                                    f.write('\n======================================')
                                    f.write('\nunits: ' + str(units_selection))
                                    f.write('\nbatch_size: ' + str(batch_size_selection))
                                    f.write('\nnb_epoch: ' + str(nb_epoch_selection))
                                    f.write('\noptimizer: ' + optimizer_selection)
                                    f.write('\nactivation: ' + activation_selection)
                                    f.write('\nactivation_output: ' + activation_output_selection)
                                    f.write('\nerror: ' + str(e))

#target_name = 'Sub_Cat'
#target_type_count = 9
#targets_others = ['Cat', 'Label']
# ['Label', 'Cat', 'Sub_Cat']
# [2, 5, 9]


# parameter tuning
parameters = {'batch_size': [10, 100],
              'nb_epoch': [100, 200],
              'optimizer': ['adam', 'rmsprop', 'SGD', 'adamax', 'adagrad'],
              'activation': ['relu', 'tanh', 'sigmoid', 'softplus', 'softmax'],
              'activation_output': ['relu', 'tanh', 'sigmoid', 'softplus', 'softmax'],
              'units': [100, 200]}
# parameters = {'batch_size': [int(sys.argv[1])],
              # 'nb_epoch': [int(sys.argv[2])],
              # 'optimizer': [sys.argv[3]],
              # 'activation': [sys.argv[4]],
              # 'activation_output': [sys.argv[5]],
              # 'units': [int(sys.argv[6])]}
# parameter_selection('Label', 2, ['Cat', 'Sub_Cat'],parameters) # binary classification based on grid search, results will be saved to Label_log.txt or Label_error.txt
# parameter_selection('Cat', 5, ['Label', 'Sub_Cat'],parameters) # 5-class classification based on grid search, results will be saved to Cat_log.txt or Cat_error.txt
parameter_selection('Sub_Cat', 9, ['Label', 'Cat'],parameters) # 9-class classification based on grid search, results will be saved to Sub_Cat_log.txt or Sub_Cat_error.txt