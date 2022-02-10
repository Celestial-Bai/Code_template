import xgboost as xgb
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

random.seed(0)
train_data = pd.read_csv('X_train.csv')
y = pd.read_csv('y_train.csv')
y = y - 1
X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.33, random_state=0)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
params = {
    'booster': 'gbtree',
    'objective': 'multi:softprob',
    'num_class': 6,
    'max_depth': 5,
    'eta': 0.05,
    'subsample': 0.7,
    'seed': 0
}
plst = params
evallist = [(dtrain, 'train'), (dval, 'eval')]
num_round = 695
bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=30)
test_data = pd.read_csv('X_test.csv')
dtest = xgb.DMatrix(test_data)
preds = bst.predict(dtest)
pred = np.nanargmax(preds, axis=1) + 1

f = open('xgb_full_att1.txt',mode='w')
for i in range(len(pred)):
    print(pred[i], file=f)
f.close()
