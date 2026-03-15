#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PrimClasif_CV — XGBoost
Based on original PrimClasif_CV.py by aguillenATC, jherrera (2018)

Changes from original:
  - 30 configs (original: 80)
  - 10 inner folds (original: 3), 10 outer folds
  - Fixed: num_round 2→300, eta range 0.05–1.0, max_depth 2–6
  - engine='python' added to read_csv
  - Tee class for guaranteed output save
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# ── Tee ──────────────────────────────────────────────────────────────────────
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

sys.stdout = Tee('/kaggle/working/RESULTS_CV_XGBoost.txt')

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

# ===========================================================================
# XGBoost
# ===========================================================================
print('""""""""""""""""""""""""""""""""""""""""""""""')
print('XGBoost')
print('""""""""""""""""""""""""""""""""""""""""""""""')

# 30 configs: max_depth 2-6, eta 0.05-1.0 (same ranges as paper)
max_depth_list = [2, 3, 4, 5, 6]
eta_list = np.round(np.arange(0.05, 1.05, 0.05), 2)  # 20 values

np.random.seed(seed)
config_listXGB = []
while len(config_listXGB) < 30:
    d   = int(np.random.choice(max_depth_list))
    eta = float(np.random.choice(eta_list))
    cfg = (d, eta)
    if cfg not in config_listXGB:
        config_listXGB.append(cfg)

XGB_perf_record_test_CV          = {}
XGB_perf_mean_record_test_CV     = {}
XGB_perf_mean_record_test_CV_std = {}

start_t = time.time()

config_idx = -1
for max_depth, eta in config_listXGB:
    config_idx += 1
    fold = -1
    XGB_perf_record_test_CV[config_idx] = np.zeros(n_folds)

    for train_indices, test_indices in k_fold.split(X_train, Y_train):
        fold += 1
        start_time = time.time()

        X_train_CV, X_test_CV = X_train[train_indices], X_train[test_indices]
        Y_train_CV, Y_test_CV = Y_train[train_indices], Y_train[test_indices]

        dtrain = xgb.DMatrix(X_train_CV, label=Y_train_CV)
        dtest  = xgb.DMatrix(X_test_CV)

        param = {
            'max_depth': max_depth, 'eta': eta,
            'objective': 'multi:softmax', 'num_class': 5,
            'eval_metric': 'merror', 'verbosity': 0
        }
        bst = xgb.train(param, dtrain, num_boost_round=300, verbose_eval=False)
        Y_pred = bst.predict(dtest)

        XGB_perf_record_test_CV[config_idx][fold] = accuracy_score(Y_test_CV, Y_pred)

        print("config %d (depth=%d eta=%.2f) fold %d acc=%.4f --- %.1fs" % (
            config_idx, max_depth, eta, fold,
            XGB_perf_record_test_CV[config_idx][fold],
            time.time() - start_time))

    XGB_perf_mean_record_test_CV[config_idx]     = np.mean(XGB_perf_record_test_CV[config_idx])
    XGB_perf_mean_record_test_CV_std[config_idx] = np.std(XGB_perf_record_test_CV[config_idx])
    print("config %d MEAN=%.4f STD=%.4f" % (
        config_idx,
        XGB_perf_mean_record_test_CV[config_idx],
        XGB_perf_mean_record_test_CV_std[config_idx]))

elapsed_t['XGB_inner'] = time.time() - start_t

best_indexXGB = list(XGB_perf_mean_record_test_CV.keys())[
    np.argmax(list(XGB_perf_mean_record_test_CV.values()))]
best_depth, best_eta = config_listXGB[best_indexXGB]

print('\nBest config: max_depth=%d eta=%.2f' % (best_depth, best_eta))
print('Best inner CV accuracy: %.4f (std=%.4f)' % (
    XGB_perf_mean_record_test_CV[best_indexXGB],
    XGB_perf_mean_record_test_CV_std[best_indexXGB]))

# ── Outer 10-fold CV ──────────────────────────────────────────────────────────
XGB_perf_record_test    = np.zeros(n_foldsOUT)
XGB_perf_record_test_f1 = np.zeros(n_foldsOUT)

print('\n--- Outer 10-fold CV ---')
fold = -1
for train_indices, test_indices in k_foldOUT.split(X_TODO, Y_TODO):
    fold += 1
    start_time = time.time()

    X_train_CV2, X_test_CV2 = X_TODO[train_indices], X_TODO[test_indices]
    Y_train_CV2, Y_test_CV2 = Y_TODO[train_indices], Y_TODO[test_indices]

    dtrain2 = xgb.DMatrix(X_train_CV2, label=Y_train_CV2)
    dtest2  = xgb.DMatrix(X_test_CV2)

    param = {
        'max_depth': best_depth, 'eta': best_eta,
        'objective': 'multi:softmax', 'num_class': 5,
        'eval_metric': 'merror', 'verbosity': 0
    }
    bst = xgb.train(param, dtrain2, num_boost_round=300, verbose_eval=False)
    Y_pred2 = bst.predict(dtest2)

    XGB_perf_record_test[fold]    = accuracy_score(Y_test_CV2, Y_pred2)
    XGB_perf_record_test_f1[fold] = f1_score(Y_test_CV2, Y_pred2,
                                              labels=np.array(range(5)), average='macro')

    print("outer fold %d acc=%.4f f1=%.4f --- %.1fs" % (
        fold, XGB_perf_record_test[fold],
        XGB_perf_record_test_f1[fold],
        time.time() - start_time))

elapsed_t['XGB'] = time.time() - start_t

print('\nXGBoost - Best\'s test CV accuracy %f (std= %f ) for config max_depth=%d eta=%.2f \n' % (
    XGB_perf_mean_record_test_CV[best_indexXGB],
    XGB_perf_mean_record_test_CV_std[best_indexXGB],
    best_depth, best_eta))
print('XGBoost - Test accuracy %s , mean: %f (std= %f) \n' % (
    XGB_perf_record_test, np.mean(XGB_perf_record_test), np.std(XGB_perf_record_test)))
print('XGBoost - Test f1_score %s , mean: %f (std= %f) \n' % (
    XGB_perf_record_test_f1, np.mean(XGB_perf_record_test_f1), np.std(XGB_perf_record_test_f1)))
print('Time elapsed for XGBoost %f' % elapsed_t['XGB'])
