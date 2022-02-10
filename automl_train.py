import pandas as pd
import numpy as np
import autogluon.text
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('X_train.csv')
label = TabularDataset('y_train.csv')
train_data['y'] = label
label = 'y'
predictor = TabularPredictor(label=label, problem_type = 'multiclass', eval_metric = 'accuracy').fit(train_data)#, excluded_model_types = ['NN'])
predictor.leaderboard()