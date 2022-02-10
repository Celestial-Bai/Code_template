import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np

test_data = TabularDataset('X_test.csv')
predictor_new = TabularPredictor.load('/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zeheng/DS/rp2/AutogluonModels/ag-20220112_030612')
label = 'y'
preds = predictor_new.predict(test_data)
submission = pd.DataFrame({label:preds})
submission.to_csv('automl_full_att1.txt', index=False)