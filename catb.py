from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

random.seed(0)
train_data = pd.read_csv('X_train.csv')
y = pd.read_csv('y_train.csv')
y = y - 1
X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.33, random_state=0)

params = {
    'iterations':2000,
    'learning_rate':0.05,
    'depth':6,
    'classes_count': 6,
    'loss_function':'MultiClass',
    'eval_metric': 'Accuracy',
    'logging_level':'Verbose'
}
model = CatBoostClassifier(**params)
model.fit(X_train,y_train,eval_set=(X_val, y_val))
test_data = pd.read_csv('X_test.csv')
pred = model.predict(test_data) + 1

f = open('catb_full_att1.txt',mode='w')
for i in range(len(pred)):
    print(int(pred[i]), file=f)
f.close()