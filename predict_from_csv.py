# source CAT/bin/activate
# conda activate CAT

# python predict_from_csv.py -d data-naive.csv -p naive
# python predict_from_csv.py -d data-cat.csv -p catboost

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.naive_bayes import GaussianNB
import argparse
import pickle

parser = argparse.ArgumentParser(description="Пример использования консольного интерфейса")
parser.add_argument("-d", "--data", help="Введите (путь и/или) имя data.csv", required=True)
parser.add_argument("-p", "--predict", help="Введите имя предсказателя: naive или catboost", required=True)
 
args = parser.parse_args()
 
test_csv = args.data
name_model = args.predict

X_test = pd.read_csv(test_csv)

iso_reg = IsotonicRegression(y_min = 0, y_max = 1, out_of_bounds = 'clip')

if name_model == "naive":
	model = GaussianNB()      # parameters not required.
	with open('naive_bayes.pkl', 'rb') as f:
		model = pickle.load(f)
	with open('naive_iso_reg.pkl', 'rb') as f:
		iso_reg = pickle.load(f)
else:
	model = CatBoostClassifier()      # parameters not required.
	model.load_model('catboost_classification_space_luck_ID.model')
	with open('catboost_iso_reg.pkl', 'rb') as f:
		iso_reg = pickle.load(f)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
y_pred_proba_positive = y_pred_proba[:, 1]
print(f'y_predict = \n{y_pred}')

proba_test_forest_isoreg = iso_reg.predict(model.predict_proba(X_test)[:, 1])

X_test['predicted'] = y_pred
X_test['probability space %'] = np.round_(y_pred_proba_positive*100, decimals = 2)
X_test['calibration isoreg probability space %'] = np.round_(proba_test_forest_isoreg*100, decimals = 2)
print('\nX_test+predicted\n', X_test.head(11), '\n')

X_test.to_csv(f'data-predicted-{name_model}.csv')