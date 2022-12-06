import numpy as np
import pandas as pd
import os
import math
from datetime import datetime
from numpy import argmax



# Importing the Keras libraries and packages
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# path to the dataset
file = 'Shaleeza.Dataset.v1.csv'

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


def build_classifier(optimizer,kernel_initializer, activation, units):
    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=units, kernel_initializer=kernel_initializer, activation=activation))

    # Adding the second hidden layer
    classifier.add(Dense(units=units, kernel_initializer=kernel_initializer, activation=activation))

    # Adding the output layer
    classifier.add(Dense(units=2, kernel_initializer=kernel_initializer, activation=activation))

    # Compiling the ANN
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[tensorflow.keras.metrics.AUC()])

    return classifier



def run_ANN(target_name, targets_others):

    dataset = data_preprocessing(targets_others)

    X = dataset.drop(labels=target_name, axis=1)
    y = dataset[target_name]

    le = LabelEncoder()
    y = le.fit_transform(y)

    from tensorflow import keras
    output_category = keras.utils.to_categorical(y, num_classes=None)

    X_train, X_test, y_train, y_test = train_test_split(X, output_category, test_size=0.25, random_state=0)

    y_train = y_train.argmax(axis = 1)
    y_test = y_test.argmax(axis = 1)

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

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier

    classifier = KerasClassifier(build_fn=build_classifier)

    from sklearn.model_selection import GridSearchCV
    parameters = {'batch_size': [10, 100],
                  'nb_epoch': [100, 200],
                  'optimizer': ['adam', 'rmsprop', 'SGD', 'adamax', 'adagrad'],
                  'kernel_initializer': ['uniform'],
                  'activation': ['relu', 'tanh', 'sigmoid', 'softplus', 'softmax'],
                  'units': [100, 200]}
    gridSearch = GridSearchCV(estimator=classifier,
                              param_grid=parameters,
                              cv=10,
                              n_jobs=-1,
                              return_train_score=True)

    gridSearch.fit(X, output_category)

    print('Grid Search Best score', gridSearch.best_score_)
    print('Grid Search Best Parameters', gridSearch.best_params_)
    print('Execution time', gridSearch.refit_time_)


    '''
    # Part 3 - Making predictions and evaluating the model
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    
    # Accuracy
    _, accuracy = classifier.evaluate(X_test, y_test, verbose=0)
    print(accuracy)

    from sklearn.metrics import precision_recall_fscore_support as score
    precision, recall, fscore, support = score(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    for i in range(target_type_count):  # 2,5,9 classes
        print('label: {}, precision: {}, recall: {}, fscore: {}, support: {}'.format(i + 1, precision[i], recall[i],
                                                                                     fscore[i], support[i]))

    precision, recall, fscore, support = score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
    print('Macro: precision: {}, recall: {}, fscore: {}, support: {}'.format(precision, recall, fscore, support))
    precision, recall, fscore, support = score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
    print('Weighted: precision: {}, recall: {}, fscore: {}, support: {}'.format(precision, recall, fscore, support))

'''


# Target
# ['Label', 'Cat', 'Sub_Cat']
# [2, 5, 9]

#run_ANN('Sub_Cat', 9, ['Label', 'Cat'])
#run_ANN('Cat', 5, ['Label', 'Sub_Cat'])
run_ANN('Label', ['Cat', 'Sub_Cat'])