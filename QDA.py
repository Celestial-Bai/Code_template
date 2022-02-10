from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np
import random

random.seed(0)
train_data = pd.read_csv('X_train.csv')
y = pd.read_csv('y_train.csv')
y = np.array(y)[:, 0] - 1
clf = QuadraticDiscriminantAnalysis()
clf.fit(train_data, y)

test_data = pd.read_csv('X_test.csv')
pred = clf.predict(test_data) + 1

f = open('qda_full_att1.txt',mode='w')
for i in range(len(pred)):
    print(pred[i] , file=f)
f.close()