from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import random

random.seed(0)
train_data = pd.read_csv('X_train.csv')
X_train = np.array(train_data, dtype='float32')
y = pd.read_csv('y_train.csv')
y = np.array(y) - 1

#X_embedded = PCA(n_components=2).fit_transform(X_train)
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(X_train)

tsne=pd.DataFrame(np.concatenate((X_embedded,y), axis=1))

import matplotlib.pyplot as plt

d=tsne[tsne[2] == 0]
plt.scatter(d[0],d[1], c='#FFB6C1', marker='.', label=u'1')

d=tsne[tsne[2] == 1]
plt.scatter(d[0],d[1], c='#BA55D3', marker='.', label=u'2')

d=tsne[tsne[2] == 2]
plt.scatter(d[0],d[1], c='#00BFFF', marker='.', label=u'3')

d=tsne[tsne[2] == 3]
plt.scatter(d[0],d[1], c='#FFD700', marker='.', label=u'4')

d=tsne[tsne[2] == 4]
plt.scatter(d[0],d[1], c='#A0522D', marker='.', label=u'5')

d=tsne[tsne[2] == 5]
plt.scatter(d[0],d[1], c='#98FB98', marker='.', label=u'6')

plt.legend()
plt.savefig('tsne.png')
plt.show()