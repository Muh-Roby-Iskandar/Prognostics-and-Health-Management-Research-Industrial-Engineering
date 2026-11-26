#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mulai Dibuat pada Kamis, 04 Juni 2026 pada 21:42:00 WIB

@author : Muhammad Roby Iskandar, 16012200194, Fakultas Sains dan Teknologi, Teknik Industri

Universitas Bina Bangsa, Serang
"""

import os
import sys
import platform
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, cross_val_score

import scipy
from scipy.stats import uniform

import joblib

print(platform.system())
print(platform.release())
print(platform.machine())
print(sys.version.split()[0])
print(sys.executable)

print(np.__version__)
print(pd.__version__)
print(matplotlib.__version__)

print(sklearn.__version__)

print(scipy.__version__)

print(joblib.__version__)

RANDOM_STATE = 42
RUL_CAP = 125
N_OP_CONDITIONS = 6
SENSOR_SELECTED = 21

sensor_mapping = {
    'op_setting_1': 'altitude',
    'op_setting_2': 'mach_num',
    'op_setting_3': 'sea_lv_temp',
    'sensor_1': 'T2',
    'sensor_2': 'T24',
    'sensor_3': 'T30',
    'sensor_4': 'T50',
    'sensor_5': 'P2',
    'sensor_6': 'P15',
    'sensor_7': 'P30',
    'sensor_8': 'Nf',
    'sensor_9': 'Nc',
    'sensor_10': 'epr',
    'sensor_11': 'Ps30',
    'sensor_12': 'phi',
    'sensor_13': 'NRf',
    'sensor_14': 'NRc',
    'sensor_15': 'BPR',
    'sensor_16': 'farB',
    'sensor_17': 'htBleed',
    'sensor_18': 'Nf_dmd',
    'sensor_19': 'PCNfR_dmd',
    'sensor_20': 'W31',
    'sensor_21': 'W32'
}

cols_data = (['unit_id', 'cycle'] +
        [f'op_setting_{i}' for i in range(1,4)] +
        [f'sensor_{i}' for i in range(1,22)])

sensor_cols = ['T2', 'T24', 'T30', 'T50',
          'P2', 'P15', 'P30', 'Nf',
          'Nc', 'epr', 'Ps30', 'phi',
          'NRf', 'NRc', 'BPR', 'farB',
          'htBleed', 'Nf_dmd', 'PCNfR_dmd',
          'W31', 'W32']

op_cols = ['altitude','mach_num','sea_lv_temp']

meta_cols = ['unit_id','cycle','op_condition','RUL']

train_acq = pd.read_csv(
    r"/home/roby/Documents/Thesis Defense/Results and Discussion/C-MAPSS/train_FD004.txt",
    sep = ' ', header=None
)

test_acq = pd.read_csv(
    r"/home/roby/Documents/Thesis Defense/Results and Discussion/C-MAPSS/test_FD004.txt",
    sep = ' ', header=None
)

rul_acq = pd.read_csv(
    r"/home/roby/Documents/Thesis Defense/Results and Discussion/C-MAPSS/RUL_FD004.txt",
    header=None, names = ['RUL_init']
)

train_acq = train_acq.dropna(axis=1)
test_acq = test_acq.dropna(axis=1)
rul_acq = rul_acq.dropna().reset_index(drop=True)

train_acq.columns = cols_data
test_acq.columns = cols_data

train_acq = train_acq.rename(columns=sensor_mapping)
test_acq = test_acq.rename(columns=sensor_mapping)

max_cycle = train_acq.groupby('unit_id')['cycle'].max().reset_index()
max_cycle.columns = ['unit_id','max_cycle']

train_acq = train_acq.merge(max_cycle, on ='unit_id')

train_acq['RUL'] = train_acq['max_cycle'] - train_acq['cycle']

train_acq.drop(columns=['max_cycle'], inplace = True)

test_acq.columns = test_acq.columns.astype(str)
cols_to_drop = [
    c for c in test_acq.columns if 'max_cycle' in c.lower() or 'rul' in c.lower()
    ]

test_acq = test_acq.drop(columns=cols_to_drop)

max_cycle = test_acq.groupby('unit_id')['cycle'].max().reset_index(name='max_cycle')
test_acq = test_acq.merge(max_cycle, on = 'unit_id')
test_acq = test_acq.merge(rul_acq, left_on='unit_id', right_index=True)

test_acq['RUL'] = test_acq['RUL_init'] + (test_acq['max_cycle'] - test_acq['cycle'])

test_acq = test_acq.drop(columns = ['max_cycle', 'RUL_init'])

train_s = train_acq.copy()
test_s = test_acq.copy()

for df in [train_s, test_s]:
    df['altitude'] = df['altitude'].round(1)
    df['mach_num'] = df['mach_num'].round(2)
    df['sea_lv_temp'] = df['sea_lv_temp'].round(0)

op_condition_map = (
    train_s[op_cols]
    .drop_duplicates()
    .sort_values(op_cols)
    .reset_index(drop=True)
)

op_condition_map['op_condition'] = op_condition_map.index

train_s = train_s.merge(op_condition_map, on = op_cols, how = 'left')
test_s = test_s.merge(op_condition_map, on = op_cols, how = 'left')

train_sc = train_s.copy()
test_sc = test_s.copy()

scaler_sensor_per_cond = {}

for cond in sorted(train_s['op_condition'].unique()):

    idx_train = train_s['op_condition'] == cond
    idx_test = test_s['op_condition'] == cond

    scaler = MinMaxScaler()
    scaler.fit(train_sc.loc[idx_train, sensor_cols])

    train_sc.loc[idx_train, sensor_cols] = scaler.transform(train_s.loc[idx_train, sensor_cols])
    if idx_test.sum() > 0:
        test_sc.loc[idx_test, sensor_cols] = scaler.transform(test_s.loc[idx_test, sensor_cols])
    scaler_sensor_per_cond[cond] = scaler

scaler_op_global = MinMaxScaler()
scaler_op_global.fit(train_s[op_cols])

train_sc[op_cols] = scaler_op_global.transform(train_s[op_cols])
test_sc[op_cols] = scaler_op_global.transform(test_s[op_cols])

train_rul_cap = train_sc.copy()
test_rul_cap = test_sc.copy()

def apply_rul_clipping(df, rul_max=RUL_CAP):
    df = df.copy()
    df['RUL'] = df['RUL'].clip(upper=RUL_CAP)
    return df

train_rul_cap = apply_rul_clipping(train_rul_cap, rul_max=RUL_CAP)
test_rul_cap = apply_rul_clipping(test_rul_cap,  rul_max=RUL_CAP)

def build_temporal_features(df, sensor_cols, unit_col='unit_id', cycle_col='cycle'):
    df = df.sort_values([unit_col, cycle_col]).copy()
    new_features = {}

    for sensor in sensor_cols:
        g = df.groupby(unit_col)[sensor]
        new_features[f'{sensor}_lag1']    = g.shift(1)
        new_features[f'{sensor}_rmean10'] = g.transform(lambda x: x.rolling(10, min_periods=1).mean())
        new_features[f'{sensor}_ewm3']    = g.transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
        new_features[f'{sensor}_rstd10']  = g.transform(lambda x: x.rolling(10, min_periods=1).std())
        new_features[f'{sensor}_max10']  = g.transform(lambda x: x.rolling(10, min_periods=1).max())
        new_features[f'{sensor}_min10']  = g.transform(lambda x: x.rolling(10, min_periods=1).min())
    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

train_model = build_temporal_features(train_rul_cap, sensor_cols)
test_model  = build_temporal_features(test_rul_cap,  sensor_cols)

train_model = train_model.fillna(0)
test_model  = test_model.fillna(0)

groups = train_model['unit_id'] 

drop_cols = ['unit_id', 'cycle', 'op_condition', 'RUL']

X_train = train_model.drop(columns=drop_cols)
y_train = train_model['RUL']

X_test  = test_model.drop(columns=drop_cols)
y_test  = test_model['RUL']

cv = GroupKFold(n_splits=5)

param_svm = {
    'C'      : uniform(0.1, 50),
    'epsilon': uniform(0.01, 2),
    'gamma'  : ['scale', 'auto'],
}

search_svr = RandomizedSearchCV(
    SVR(),
    param_svm,
    n_iter=50,
    cv=cv,                                      
    scoring='neg_root_mean_squared_error',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

search_svr.fit(X_train, y_train, groups=groups)

print(f"Best params : {search_svr.best_params_}")
print(f"Best CV RMSE: {-search_svr.best_score_:.4f}")

best_svr = search_svr.best_estimator_
y_pred_svr_train = best_svr.predict(X_train)
y_pred_svr_test  = best_svr.predict(X_test)

rmse_train_svr = root_mean_squared_error(y_train, y_pred_svr_train)
mae_train_svr = mean_absolute_error(y_train, y_pred_svr_train)
rmse_test_svr = root_mean_squared_error(y_test, y_pred_svr_test)
mae_test_svr = mean_absolute_error(y_test, y_pred_svr_test)

def nasa_score(y_true, y_pred):
    diff = y_pred - y_true
    score = np.where(diff < 0,
                     np.sum(np.exp(-diff/13) - 1),
                     np.sum(np.exp(diff/10) - 1))
    return np.sum(score)

print(nasa_score(y_test, y_pred_svr_test))

print(rmse_train_svr)
print(mae_train_svr)
print(rmse_test_svr)
print(mae_test_svr)

print(best_svr.n_support_[0])
print(best_svr.n_support_[0]/len(X_train)*100)

cv_scores = cross_val_score(
    best_svr, X_train, y_train,
    cv=GroupKFold(n_splits=5),
    scoring='neg_root_mean_squared_error',
    groups=groups, n_jobs=2
)
print(f"CV RMSE tiap fold : {(-cv_scores).round(4)}")
print(f"CV RMSE mean      : {-cv_scores.mean():.4f}")
print(f"CV RMSE std       : {cv_scores.std():.4f}")

plot_dir = r"/home/roby/Documents/Thesis Defense/Results and Discussion/Plots"
os.makedirs(plot_dir, exist_ok=True)

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

results_svr = pd.DataFrame(search_svr.cv_results_)

results_svr = results_svr[[
    'rank_test_score',
    'mean_test_score',
    'std_test_score',
    'param_C',
    'param_epsilon',
    'param_gamma',
]].sort_values('rank_test_score')

results_svr['mean_rmse'] = -results_svr['mean_test_score']
results_svr['std_rmse']  = results_svr['std_test_score']

results_svr = results_svr.drop(columns=['mean_test_score','std_test_score'])

print(results_svr.to_string())

results_svr.to_csv(os.path.join(plot_dir, 'cv_results_svr.csv'), index=False)

fig, ax = plt.subplots()
residuals = y_test - y_pred_svr_test
plt.scatter(y_pred_svr_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("RUL Predicted")
plt.ylabel("Residual")
plt.title("Residual vs Predicted")
save_plot("SVR Residual vs Predicted Plot.png")
plt.show()

fig, ax = plt.subplots()
plt.scatter(y_test, y_pred_svr_test, alpha=0.5)
plt.plot([0,125],[0,125], 'r--')
plt.xlabel("RUL Actual")
plt.ylabel("RUL Predicted")
plt.title("Actual vs Predicted RUL")
save_plot("SVR Residual vs Predicted.png")
plt.show()

fig, ax = plt.subplots()
plt.scatter(y_train, y_pred_svr_train, alpha=0.5)
plt.plot([0,125],[0,125], 'r--')
plt.xlabel("RUL Actual")
plt.ylabel("RUL Predicted")
plt.title("Actual vs Predicted RUL")
save_plot("SVR BVTO.png")
plt.show()

fig, ax = plt.subplots()
plt.plot(y_test[:400], label='RUL Actual')
plt.plot(y_pred_svr_test[:400], label='RUL Predicted')
plt.legend()
plt.title("Actual vs Predicted RUL")
save_plot("SVR Residual vs Predicted RUL Plot.png")
plt.show()

save_path = r"/home/roby/Documents/Thesis Defense/Results and Discussion/model/best_svr.pkl"

joblib.dump(best_svr, save_path)
