#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PrimClasif_CV — Improved DNN
Based on original PrimClasif_CV.py by aguillenATC, jherrera (2018)

Changes from original CV_DNN.py:
  - kernel_initializer: Constant(0.025) -> glorot_uniform  [THE CORE FIX]
  - Added BatchNormalization after each hidden layer
  - Added Dropout (0.3 for units>=64, 0.2 for smaller)
  - Adam(lr=0.001) explicit
  - EarlyStopping patience=15, restore_best_weights=True
  - ReduceLROnPlateau factor=0.5 patience=7 (gentler than original 0.1)
  - 30 configs, 10 inner folds, 10 outer folds
  - Unit range: 32-256 (suits deeper improved architecture)
  - batch_size: 128 -> 256 (faster on full dataset)
  - Removed scikeras dependency -- uses native Keras directly
  - Tee class for guaranteed output save
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import keras.utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

elapsed_t = {}
prefijo = ''
path_dados   = ('../data'    + prefijo + '/')
path_results = ('../results' + prefijo + '/')
os.makedirs(path_results, exist_ok=True)

sys.stdout = Tee('/kaggle/working/RESULTS_CV_ImprovedDNN.txt')

random_st = 42
seed = 7
np.random.seed(seed)

X_df      = pd.read_csv(path_dados + 'XTrn.txt',   sep='  ', header=None, engine='python')
Y_df      = pd.read_csv(path_dados + 'YTrn.txt',   sep='  ', header=None, engine='python')
X_test_df = pd.read_csv(path_dados + 'XTest.txt',  sep='  ', header=None, engine='python')
Y_test_df = pd.read_csv(path_dados + 'YTest.txt',  sep='  ', header=None, engine='python')

X_df      = X_df[:][[2,1,4,0,3]]
X_test_df = X_test_df[:][[2,1,4,0,3]]

scalerX = StandardScaler()
scalerX.fit(X_df)

X_train = scalerX.transform(X_df)
Y_train = Y_df.values

X_testYval_norm = scalerX.transform(X_test_df)

X_TODO = np.concatenate((X_train, X_testYval_norm))
Y_TODO = np.concatenate((Y_train, Y_test_df))

X_test_norm, X_val, Y_test, Y_val = train_test_split(
    X_testYval_norm, Y_test_df.values, test_size=0.50, random_state=45)

n_folds    = 10
k_fold     = StratifiedKFold(n_splits=n_folds)
n_foldsOUT = 10
k_foldOUT  = StratifiedKFold(n_splits=n_foldsOUT)

print('""""""""""""""""""""""""""""""""""""""""""""""')
print('Improved DNN -- glorot_uniform + BatchNorm + Dropout + EarlyStopping')
print('""""""""""""""""""""""""""""""""""""""""""""""')

def clasif_model(individual):
    """
    Improved DNN. individual: list of (units, activ_f) tuples.
    glorot_uniform init + BatchNormalization + Dropout per layer.
    """
    activation_functions = {
        0: "relu", 1: "sigmoid", 2: "softmax", 3: "tanh",
        4: "selu", 5: "softplus", 6: "softsign", 7: "linear"
    }

    dimension = X_train.shape[1]
    model = Sequential()
    first_layer = True

    for units, activ_f in individual:
        if units > 5:
            if first_layer:
                model.add(Dense(units=int(units), input_dim=dimension,
                                kernel_initializer='glorot_uniform',
                                activation=activation_functions[int(activ_f)]))
                first_layer = False
            else:
                model.add(Dense(units=int(units),
                                kernel_initializer='glorot_uniform',
                                activation=activation_functions[int(activ_f)]))
            model.add(BatchNormalization())
            dropout_rate = 0.3 if int(units) >= 64 else 0.2
            model.add(Dropout(dropout_rate))

    model.add(Dense(units=5, activation="softmax",
                    kernel_initializer='glorot_uniform'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


DNN_perf_record_test_CV          = {}
DNN_perf_mean_record_test_CV     = {}
DNN_perf_mean_record_test_CV_std = {}

start_t = time.time()

# 30 configs, units 32-256
max_individuals = 30
max_depth = 6
individuals = {}
for i in range(0, max_individuals):
    individuals[i] = []
    layers = int(np.round(np.random.rand(1) * (max_depth - 1) + 1) + 1)
    units = np.ceil(np.random.rand(1, layers) * 224 + 32)[0]
    act_func = np.zeros((1, layers))[0]
    for j in range(0, layers):
        individuals[i].append((units[j], act_func[j]))

batch_size_list = np.array([256])

config_listDNN = []
for i in range(0, max_individuals):
    for j in batch_size_list:
        config_listDNN.append([individuals[i], j])

config_idx = -1
for config in config_listDNN:
    config_idx += 1
    fold = -1
    DNN_perf_record_test_CV[config_idx] = np.zeros(n_folds)

    for train_indices, test_indices in k_fold.split(X_train, Y_train):
        fold += 1
        start_time = time.time()

        X_train_CV, X_test_CV = X_train[train_indices], X_train[test_indices]
        Y_train_CV, Y_test_CV = Y_train[train_indices], Y_train[test_indices]

        hot_Y_train_CV = keras.utils.to_categorical(Y_train_CV, num_classes=5)
        hot_Y_test_CV  = keras.utils.to_categorical(Y_test_CV,  num_classes=5)

        model = clasif_model(config[0])

        early_stopping = callbacks.EarlyStopping(
            monitor='loss', min_delta=1e-03, patience=15,
            restore_best_weights=True, verbose=0, mode='min')
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=7,
            min_lr=1e-6, min_delta=1e-03, verbose=0)

        model.fit(X_train_CV, hot_Y_train_CV,
                  epochs=150, batch_size=int(config[1]),
                  callbacks=[early_stopping, reduce_lr],
                  verbose=0)

        Y_pred_test_CV = model.predict(X_test_CV, verbose=0)
        hot_Y_pred_test_CV = keras.utils.to_categorical(
            np.argmax(Y_pred_test_CV, axis=1), num_classes=5)

        DNN_perf_record_test_CV[config_idx][fold] = accuracy_score(hot_Y_test_CV, hot_Y_pred_test_CV)
        print("config %d fold %d acc=%.4f --- %.1fs" % (
            config_idx, fold,
            DNN_perf_record_test_CV[config_idx][fold],
            time.time() - start_time))

    DNN_perf_mean_record_test_CV[config_idx]     = np.mean(DNN_perf_record_test_CV[config_idx])
    DNN_perf_mean_record_test_CV_std[config_idx] = np.std(DNN_perf_record_test_CV[config_idx])
    print("config %d MEAN=%.4f STD=%.4f arch=%s" % (
        config_idx,
        DNN_perf_mean_record_test_CV[config_idx],
        DNN_perf_mean_record_test_CV_std[config_idx],
        config_listDNN[config_idx][0]))

elapsed_t['DNN_inner'] = time.time() - start_t

best_indexDNN = list(DNN_perf_mean_record_test_CV.keys())[
    np.argmax(list(DNN_perf_mean_record_test_CV.values()))]

print('\nBest config index: %d' % best_indexDNN)
print('Best inner CV accuracy: %.4f (std=%.4f)' % (
    DNN_perf_mean_record_test_CV[best_indexDNN],
    DNN_perf_mean_record_test_CV_std[best_indexDNN]))
print('Best architecture: %s' % config_listDNN[best_indexDNN][0])

# Outer 10-fold CV
config = config_listDNN[best_indexDNN]
DNN_perf_record_test    = np.zeros(n_foldsOUT)
DNN_perf_record_test_f1 = np.zeros(n_foldsOUT)

print('\n--- Outer 10-fold CV ---')
fold = -1
for train_indices, test_indices in k_foldOUT.split(X_TODO, Y_TODO):
    fold += 1
    start_time = time.time()

    X_train_CV2, X_test_CV2 = X_TODO[train_indices], X_TODO[test_indices]
    Y_train_CV2, Y_test_CV2 = Y_TODO[train_indices], Y_TODO[test_indices]

    hot_Y_train_CV2 = keras.utils.to_categorical(Y_train_CV2, num_classes=5)
    hot_Y_test_CV2  = keras.utils.to_categorical(Y_test_CV2,  num_classes=5)

    model = clasif_model(config[0])

    early_stopping = callbacks.EarlyStopping(
        monitor='loss', min_delta=1e-03, patience=15,
        restore_best_weights=True, verbose=0, mode='min')
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=7,
        min_lr=1e-6, min_delta=1e-03, verbose=0)

    model.fit(X_train_CV2, hot_Y_train_CV2,
              epochs=150, batch_size=int(config[1]),
              callbacks=[early_stopping, reduce_lr],
              verbose=0)

    Y_pred_test_CV2 = model.predict(X_test_CV2, verbose=0)
    hot_Y_pred_test_CV2 = keras.utils.to_categorical(
        np.argmax(Y_pred_test_CV2, axis=1), num_classes=5)

    DNN_perf_record_test_f1[fold] = f1_score(hot_Y_test_CV2, hot_Y_pred_test_CV2,
                                              labels=np.array(range(5)), average='macro')
    DNN_perf_record_test[fold]    = accuracy_score(hot_Y_test_CV2, hot_Y_pred_test_CV2)

    print("outer fold %d acc=%.4f f1=%.4f --- %.1fs" % (
        fold, DNN_perf_record_test[fold],
        DNN_perf_record_test_f1[fold],
        time.time() - start_time))

elapsed_t['DNN'] = time.time() - start_t

print('\nImproved DNN - Best test CV accuracy %f (std= %f ) for config %s \n' % (
    np.max(list(DNN_perf_mean_record_test_CV.values())),
    DNN_perf_mean_record_test_CV_std[best_indexDNN],
    config_listDNN[best_indexDNN]))
print('Improved DNN - Test accuracy %s , mean: %f (std= %f) \n' % (
    DNN_perf_record_test, np.mean(DNN_perf_record_test), np.std(DNN_perf_record_test)))
print('Improved DNN - Test f1_score %s , mean: %f (std= %f) \n' % (
    DNN_perf_record_test_f1, np.mean(DNN_perf_record_test_f1), np.std(DNN_perf_record_test_f1)))
print('Time elapsed for Improved DNN %f' % elapsed_t['DNN'])
