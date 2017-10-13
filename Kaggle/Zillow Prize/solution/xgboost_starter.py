import gc

import numpy as np
import pandas as pd
import xgboost as xgb

print('Loading data ...')
'''
properties2016 = pd.read_csv('../input/properties_2016.csv', low_memory=False)
properties2017 = pd.read_csv('../input/properties_2017.csv', low_memory=False)
train2016 = pd.read_csv('../input/train_2016_v2.csv')
train2017 = pd.read_csv('../input/train_2017.csv')

# sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory=False)
train2016 = pd.merge(train2016, properties2016, how='left', on='parcelid')
train2017 = pd.merge(train2017, properties2017, how='left', on='parcelid')
'''
# train_2017 = pd.read_csv('../input/train_2017.csv', parse_dates=["transactiondate"])
train = pd.read_csv('../input/train_2016_v2.csv', parse_dates=["transactiondate"])
# train = pd.concat([train_2016, train_2017])
# del train_2016, train_2017;
gc.collect()
prop = pd.read_csv('../input/properties_2016.csv', low_memory=False)
# properties2017 = pd.read_csv('../input/properties_2017.csv', low_memory=False)
# prop = pd.concat([properties2016, properties2017])
# del properties2016, properties2017;
gc.collect()
sample = pd.read_csv('../input/sample_submission.csv', low_memory=False)

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'],
                        axis=1)  # XGboost is good at dealing with numbers but definitely not good when dealing with string. So the solution is neither we drop them or transform them.
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:  # The columns which are "object" types have NaN values and True values. I he's converting the NaNs to False for easier processing.
    x_train[c] = (x_train[c] == True)

del df_train;
gc.collect()

split = 80000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid;
gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

del prop;
gc.collect()

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test, sample;
gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test;
gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test;
gc.collect()

sub = pd.read_csv('../input/sample_submission.csv')
print(p_test.shape, sub.shape)
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

from datetime import datetime

print('Writing csv ...')
sub.to_csv('../submission/xgb_starter{}.csv.gz'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4g', compression='gzip')  # Thanks to @inversion
