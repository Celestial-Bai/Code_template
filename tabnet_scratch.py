from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random

random.seed(0)
train_data = pd.read_csv('X_train.csv')
y = pd.read_csv('y_train.csv')
y = y - 1
X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.33, random_state=0)

X_train = np.array(X_train, dtype = np.float32)
X_val = np.array(X_val, dtype = np.float32)
y_train = np.array(y_train['y'])
y_val = np.array(y_val['y'])

clf = TabNetClassifier(optimizer_params=dict(lr=2e-2))
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_name=['train', 'valid'],
    eval_metric=['logloss'],
    max_epochs=300,
    patience=30,
    num_workers=0,
    drop_last=False,
)

test_data = pd.read_csv('X_test.csv')
test_data = np.array(test_data, dtype = np.float32)
pred = clf.predict(test_data) + 1

f = open('tabnet_full_att2.txt',mode='w')
for i in range(len(pred)):
    print(pred[i], file=f)
f.close()