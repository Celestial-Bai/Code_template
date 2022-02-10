import lightgbm as lgb
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

random.seed(0)
train_data = pd.read_csv('X_train_pca.csv')
y = pd.read_csv('y_train.csv')
y = y - 1
X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.33, random_state=0)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    'task': 'train',
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'num_class': 6,
    'num_leaves': 51,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'boost_from_average': True
}
gbm = lgb.train(params,lgb_train,num_boost_round=800,valid_sets=lgb_val,early_stopping_rounds=30)
test_data = pd.read_csv('X_test_pca.csv')
preds = gbm.predict(test_data)
pred = np.nanargmax(preds, axis=1) + 1

f = open('lgb_pca_att1.txt',mode='w')
for i in range(len(pred)):
    print(pred[i], file=f)
f.close()