from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import torch

random.seed(0)
train_data = pd.read_csv('X_pretrain.csv')
X_train, X_val = train_test_split(train_data, test_size=0.33, random_state=0)

X_train = np.array(X_train, dtype = np.float32)
X_val = np.array(X_val, dtype = np.float32)

unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type="sparsemax"
)

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_val],
    max_epochs=1000 ,
    patience=50,
    pretraining_ratio=0.5,
)

unsupervised_model.save_model('./pretrain_full_att2')
