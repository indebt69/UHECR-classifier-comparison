#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PrimClasif_CV — KNN
Based on original PrimClasif_CV.py by aguillenATC, jherrera (2018)

Changes from original:
  - 30 configs (original: 50 k values)
  - 10 inner folds (original: 3), 10 outer folds
  - k range: 1-50 (same as original paper)
  - engine='python' added to read_csv
  - Tee class for guaranteed output save
"""

import numpy as np
import pandas as pd
import os
import sys
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

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

sys.stdout = Tee('/kaggle/working/RESULTS_CV_KNN.txt')

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
Y_train = Y_df.values.ravel()

X_testYval_norm = scalerX.transform(X_test_df)

X_TODO = np.concatenate((X_train, X_testYval_norm))
Y_TODO = np.concatenate((Y_train, Y_test_df.values.ravel()))

X_test_norm, X_val, Y_test, Y_val = train_test_split(
    X_testYval_norm, Y_test_df.values.ravel(), test_size=0.50, random_state=45)

n_folds    = 10
k_fold     = StratifiedKFold(n_splits=n_folds)
n_foldsOUT = 10
k_foldOUT  = StratifiedKFold(n_splits=n_foldsOUT)

print('""""""""""""""""""""""""""""""""""""""""""""""')
print('KNN')
print('""""""""""""""""""""""""""""""""""""""""""""""')

np.random.seed(seed)
k_values = sorted(np.random.choice(range(1, 51), size=30, replace=False).tolist())
config_listKNN = k_values

KNN_perf_record_test_CV          = {}
KNN_perf_mean_record_test_CV     = {}
KNN_perf_mean_record_test_CV_std = {}

start_t = time.time()

config_idx = -1
for k in config_listKNN:
    config_idx += 1
    fold = -1
    KNN_perf_record_test_CV[config_idx] = np.zeros(n_folds)

    for train_indices, test_indices in k_fold.split(X_train, Y_train):
        fold += 1
        start_time = time.time()

        X_train_CV, X_test_CV = X_train[train_indices], X_train[test_indices]
        Y_train_CV, Y_test_CV = Y_train[train_indices], Y_train[test_indices]

        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(X_train_CV, Y_train_CV)
        Y_pred = clf.predict(X_test_CV)

        KNN_perf_record_test_CV[config_idx][fold] = accuracy_score(Y_test_CV, Y_pred)
        print("config %d (k=%d) fold %d acc=%.4f --- %.1fs" % (
            config_idx, k, fold,
            KNN_perf_record_test_CV[config_idx][fold],
            time.time() - start_time))

    KNN_perf_mean_record_test_CV[config_idx]     = np.mean(KNN_perf_record_test_CV[config_idx])
    KNN_perf_mean_record_test_CV_std[config_idx] = np.std(KNN_perf_record_test_CV[config_idx])
    print("config %d (k=%d) MEAN=%.4f STD=%.4f" % (
        config_idx, k,
        KNN_perf_mean_record_test_CV[config_idx],
        KNN_perf_mean_record_test_CV_std[config_idx]))

elapsed_t['KNN_inner'] = time.time() - start_t

best_indexKNN = list(KNN_perf_mean_record_test_CV.keys())[
    np.argmax(list(KNN_perf_mean_record_test_CV.values()))]
best_k = config_listKNN[best_indexKNN]

print('\nBest config: k=%d' % best_k)
print('Best inner CV accuracy: %.4f (std=%.4f)' % (
    KNN_perf_mean_record_test_CV[best_indexKNN],
    KNN_perf_mean_record_test_CV_std[best_indexKNN]))

KNN_perf_record_test    = np.zeros(n_foldsOUT)
KNN_perf_record_test_f1 = np.zeros(n_foldsOUT)

print('\n--- Outer 10-fold CV ---')
fold = -1
for train_indices, test_indices in k_foldOUT.split(X_TODO, Y_TODO):
    fold += 1
    start_time = time.time()

    X_train_CV2, X_test_CV2 = X_TODO[train_indices], X_TODO[test_indices]
    Y_train_CV2, Y_test_CV2 = Y_TODO[train_indices], Y_TODO[test_indices]

    clf = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    clf.fit(X_train_CV2, Y_train_CV2)
    Y_pred2 = clf.predict(X_test_CV2)

    KNN_perf_record_test[fold]    = accuracy_score(Y_test_CV2, Y_pred2)
    KNN_perf_record_test_f1[fold] = f1_score(Y_test_CV2, Y_pred2,
                                              labels=np.array(range(5)), average='macro')
    print("outer fold %d acc=%.4f f1=%.4f --- %.1fs" % (
        fold, KNN_perf_record_test[fold],
        KNN_perf_record_test_f1[fold],
        time.time() - start_time))

elapsed_t['KNN'] = time.time() - start_t

print('\nKNN - Best test CV accuracy %f (std= %f ) for k=%d \n' % (
    KNN_perf_mean_record_test_CV[best_indexKNN],
    KNN_perf_mean_record_test_CV_std[best_indexKNN],
    best_k))
print('KNN - Test accuracy %s , mean: %f (std= %f) \n' % (
    KNN_perf_record_test, np.mean(KNN_perf_record_test), np.std(KNN_perf_record_test)))
print('KNN - Test f1_score %s , mean: %f (std= %f) \n' % (
    KNN_perf_record_test_f1, np.mean(KNN_perf_record_test_f1), np.std(KNN_perf_record_test_f1)))
print('Time elapsed for KNN %f' % elapsed_t['KNN'])
