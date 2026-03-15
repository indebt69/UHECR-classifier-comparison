#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PrimClasif_CV — SVM
Based on original PrimClasif_CV.py by aguillenATC, jherrera (2018)

Changes from original:
  - 30 configs (original: 81)
  - 10 inner folds (original: 3), 10 outer folds
  - Fixed: C range includes high values (paper found C=512 optimal)
  - C values: logarithmic scale 2^0 to 2^10 (1 to 1024)
  - gamma values: logarithmic scale 2^-4 to 2^4 (0.0625 to 16)
  - engine='python' added to read_csv
  - Tee class for guaranteed output save

WARNING: SVM is slow. 30 configs x 10 folds on 41,992 samples
         will take several hours. Run on Kaggle GPU/CPU overnight.
"""

import numpy as np
import pandas as pd
import os
import sys
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Tee
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

sys.stdout = Tee('/kaggle/working/RESULTS_CV_SVM.txt')

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

# ===========================================================================
# SVM
# ===========================================================================
print('""""""""""""""""""""""""""""""""""""""""""""""')
print('SVM (RBF kernel)')
print('""""""""""""""""""""""""""""""""""""""""""""""')

C_list     = [2**i for i in range(0, 11)]   # 1, 2, 4, ..., 1024
gamma_list = [2**i for i in range(-4, 5)]   # 0.0625, ..., 16

np.random.seed(seed)
config_listSVM = []
while len(config_listSVM) < 30:
    C     = float(np.random.choice(C_list))
    gamma = float(np.random.choice(gamma_list))
    cfg = (C, gamma)
    if cfg not in config_listSVM:
        config_listSVM.append(cfg)

SVM_perf_record_test_CV          = {}
SVM_perf_mean_record_test_CV     = {}
SVM_perf_mean_record_test_CV_std = {}

start_t = time.time()

config_idx = -1
for C, gamma in config_listSVM:
    config_idx += 1
    fold = -1
    SVM_perf_record_test_CV[config_idx] = np.zeros(n_folds)

    for train_indices, test_indices in k_fold.split(X_train, Y_train):
        fold += 1
        start_time = time.time()

        X_train_CV, X_test_CV = X_train[train_indices], X_train[test_indices]
        Y_train_CV, Y_test_CV = Y_train[train_indices], Y_train[test_indices]

        clf = SVC(C=C, kernel='rbf', gamma=gamma)
        clf.fit(X_train_CV, Y_train_CV)
        Y_pred = clf.predict(X_test_CV)

        SVM_perf_record_test_CV[config_idx][fold] = accuracy_score(Y_test_CV, Y_pred)

        print("config %d (C=%.1f gamma=%.4f) fold %d acc=%.4f --- %.1fs" % (
            config_idx, C, gamma, fold,
            SVM_perf_record_test_CV[config_idx][fold],
            time.time() - start_time))

    SVM_perf_mean_record_test_CV[config_idx]     = np.mean(SVM_perf_record_test_CV[config_idx])
    SVM_perf_mean_record_test_CV_std[config_idx] = np.std(SVM_perf_record_test_CV[config_idx])
    print("config %d MEAN=%.4f STD=%.4f" % (
        config_idx,
        SVM_perf_mean_record_test_CV[config_idx],
        SVM_perf_mean_record_test_CV_std[config_idx]))

elapsed_t['SVM_inner'] = time.time() - start_t

best_indexSVM = list(SVM_perf_mean_record_test_CV.keys())[
    np.argmax(list(SVM_perf_mean_record_test_CV.values()))]
best_C, best_gamma = config_listSVM[best_indexSVM]

print('\nBest config: C=%.1f gamma=%.4f' % (best_C, best_gamma))
print('Best inner CV accuracy: %.4f (std=%.4f)' % (
    SVM_perf_mean_record_test_CV[best_indexSVM],
    SVM_perf_mean_record_test_CV_std[best_indexSVM]))

# Outer 10-fold CV
SVM_perf_record_test    = np.zeros(n_foldsOUT)
SVM_perf_record_test_f1 = np.zeros(n_foldsOUT)

print('\n--- Outer 10-fold CV ---')
fold = -1
for train_indices, test_indices in k_foldOUT.split(X_TODO, Y_TODO):
    fold += 1
    start_time = time.time()

    X_train_CV2, X_test_CV2 = X_TODO[train_indices], X_TODO[test_indices]
    Y_train_CV2, Y_test_CV2 = Y_TODO[train_indices], Y_TODO[test_indices]

    clf = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
    clf.fit(X_train_CV2, Y_train_CV2)
    Y_pred2 = clf.predict(X_test_CV2)

    SVM_perf_record_test[fold]    = accuracy_score(Y_test_CV2, Y_pred2)
    SVM_perf_record_test_f1[fold] = f1_score(Y_test_CV2, Y_pred2,
                                              labels=np.array(range(5)), average='macro')

    print("outer fold %d acc=%.4f f1=%.4f --- %.1fs" % (
        fold, SVM_perf_record_test[fold],
        SVM_perf_record_test_f1[fold],
        time.time() - start_time))

elapsed_t['SVM'] = time.time() - start_t

print('\nSVM - Best test CV accuracy %f (std= %f ) for config C=%.1f gamma=%.4f \n' % (
    SVM_perf_mean_record_test_CV[best_indexSVM],
    SVM_perf_mean_record_test_CV_std[best_indexSVM],
    best_C, best_gamma))
print('SVM - Test accuracy %s , mean: %f (std= %f) \n' % (
    SVM_perf_record_test, np.mean(SVM_perf_record_test), np.std(SVM_perf_record_test)))
print('SVM - Test f1_score %s , mean: %f (std= %f) \n' % (
    SVM_perf_record_test_f1, np.mean(SVM_perf_record_test_f1), np.std(SVM_perf_record_test_f1)))
print('Time elapsed for SVM %f' % elapsed_t['SVM'])
