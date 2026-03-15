#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PrimClasif_CV — Random Forest (corrected + optimized)

Fixes:
- Removed StandardScaler (not needed for Random Forest)
- Proper hyperparameter grid
- StratifiedKFold shuffle
- Faster Random Forest parameters
"""

import numpy as np
import pandas as pd
import os
import sys
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Tee logger
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

path_dados = ('../data' + prefijo + '/')
path_results = ('../results' + prefijo + '/')

os.makedirs(path_results, exist_ok=True)

#sys.stdout = Tee('/kaggle/working/RESULTS_CV_RandomForest.txt')


# Seeds
random_st = 42
seed = 7
np.random.seed(seed)


# Load data
X_df = pd.read_csv(path_dados + 'XTrn.txt', sep='  ', header=None, engine='python')
Y_df = pd.read_csv(path_dados + 'YTrn.txt', sep='  ', header=None, engine='python')

X_test_df = pd.read_csv(path_dados + 'XTest.txt', sep='  ', header=None, engine='python')
Y_test_df = pd.read_csv(path_dados + 'YTest.txt', sep='  ', header=None, engine='python')


# Feature ordering (same as original pipeline)
X_df = X_df[[2,1,4,0,3]]
X_test_df = X_test_df[[2,1,4,0,3]]

X_train = X_df.values
Y_train = Y_df.values.ravel()

X_test_full = X_test_df.values
Y_test_full = Y_test_df.values.ravel()


# Split validation/test
X_test_norm, X_val, Y_test, Y_val = train_test_split(
    X_test_full,
    Y_test_full,
    test_size=0.50,
    random_state=45
)


# Combine for outer CV
X_TODO = np.concatenate((X_train, X_test_full))
Y_TODO = np.concatenate((Y_train, Y_test_full))


# Cross-validation setup
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

n_foldsOUT = 10
k_foldOUT = StratifiedKFold(n_splits=n_foldsOUT, shuffle=True, random_state=seed)


print("======================================")
print("Random Forest (original contribution)")
print("======================================")


# Hyperparameter grid
n_estimators_list = [100, 200, 300, 400]
max_depth_list = [10, 20, 30]

config_listRF = []

for n_est in n_estimators_list:
    for depth in max_depth_list:
        config_listRF.append((n_est, depth))

print("Total configurations:", len(config_listRF))


RF_perf_record_test_CV = {}
RF_perf_mean_record_test_CV = {}
RF_perf_mean_record_test_CV_std = {}

start_t = time.time()


# Inner CV (hyperparameter selection)
config_idx = -1

for n_est, m_dept in config_listRF:

    config_idx += 1
    fold = -1

    RF_perf_record_test_CV[config_idx] = np.zeros(n_folds)

    print(f">>> Config {config_idx} (trees={n_est}, depth={m_dept})", flush=True)

    for train_indices, test_indices in k_fold.split(X_train, Y_train):

        fold += 1
        start_time = time.time()

        X_train_CV = X_train[train_indices]
        X_test_CV = X_train[test_indices]

        Y_train_CV = Y_train[train_indices]
        Y_test_CV = Y_train[test_indices]

        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=m_dept,
            max_features='sqrt',
            random_state=random_st,
            n_jobs=-1
        )

        clf.fit(X_train_CV, Y_train_CV)

        Y_pred = clf.predict(X_test_CV)

        RF_perf_record_test_CV[config_idx][fold] = accuracy_score(Y_test_CV, Y_pred)

        print(
            f"config {config_idx} fold {fold} acc={RF_perf_record_test_CV[config_idx][fold]:.4f}"
            f" --- {time.time()-start_time:.1f}s",
            flush=True
        )

    RF_perf_mean_record_test_CV[config_idx] = np.mean(RF_perf_record_test_CV[config_idx])
    RF_perf_mean_record_test_CV_std[config_idx] = np.std(RF_perf_record_test_CV[config_idx])

    print(
        f"config {config_idx} MEAN={RF_perf_mean_record_test_CV[config_idx]:.4f}"
        f" STD={RF_perf_mean_record_test_CV_std[config_idx]:.4f}",
        flush=True
    )


elapsed_t['RF_inner'] = time.time() - start_t


# Best configuration
best_indexRF = list(RF_perf_mean_record_test_CV.keys())[
    np.argmax(list(RF_perf_mean_record_test_CV.values()))
]

best_n_est, best_depth = config_listRF[best_indexRF]

print("\nBest config:", best_n_est, best_depth)


# Outer CV (final evaluation)
RF_perf_record_test = np.zeros(n_foldsOUT)
RF_perf_record_test_f1 = np.zeros(n_foldsOUT)

print("\n--- Outer 10-fold CV ---")

fold = -1

for train_indices, test_indices in k_foldOUT.split(X_TODO, Y_TODO):

    fold += 1
    start_time = time.time()

    X_train_CV2 = X_TODO[train_indices]
    X_test_CV2 = X_TODO[test_indices]

    Y_train_CV2 = Y_TODO[train_indices]
    Y_test_CV2 = Y_TODO[test_indices]

    clf = RandomForestClassifier(
        n_estimators=best_n_est,
        max_depth=best_depth,
        max_features='sqrt',
        random_state=random_st,
        n_jobs=-1
    )

    clf.fit(X_train_CV2, Y_train_CV2)

    Y_pred2 = clf.predict(X_test_CV2)

    RF_perf_record_test[fold] = accuracy_score(Y_test_CV2, Y_pred2)

    RF_perf_record_test_f1[fold] = f1_score(
        Y_test_CV2,
        Y_pred2,
        labels=np.array(range(5)),
        average='macro'
    )

    print(
        f"outer fold {fold} acc={RF_perf_record_test[fold]:.4f}"
        f" f1={RF_perf_record_test_f1[fold]:.4f}"
        f" --- {time.time()-start_time:.1f}s",
        flush=True
    )


elapsed_t['RF'] = time.time() - start_t


print("\nRandom Forest results")

print("Accuracy:", RF_perf_record_test,
      "mean:", np.mean(RF_perf_record_test),
      "std:", np.std(RF_perf_record_test))

print("F1:", RF_perf_record_test_f1,
      "mean:", np.mean(RF_perf_record_test_f1),
      "std:", np.std(RF_perf_record_test_f1))

print("Total time:", elapsed_t['RF'])

# Save results
with open('/kaggle/working/RESULTS_CV_RandomForest.txt', 'w') as f:
    f.write("RF mean acc: %.4f std: %.4f\n" % (
        np.mean(RF_perf_record_test), np.std(RF_perf_record_test)))
    f.write("RF mean f1:  %.4f std: %.4f\n" % (
        np.mean(RF_perf_record_test_f1), np.std(RF_perf_record_test_f1)))
    f.write("Best config: n_est=%d depth=%d\n" % (best_n_est, best_depth))
    f.write("Total time: %.1fs\n" % elapsed_t['RF'])
print("Results saved to /kaggle/working/RESULTS_CV_RandomForest.txt")
