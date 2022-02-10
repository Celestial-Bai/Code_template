from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

random.seed(0)
train_data = pd.read_csv('X_train.csv')
y = pd.read_csv('y_train.csv')
y = np.array(y) - 1
# X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.33, random_state=0)
model = LinearSVC(random_state=0, multi_class='crammer_singer', penalty = 'l2')
model.fit(train_data, y.ravel())
test_data = pd.read_csv('X_test.csv')
pred = model.predict(test_data) + 1

f = open('lsvm_full_att3.txt',mode='w')
for i in range(len(pred)):
    print(pred[i] , file=f)
f.close()