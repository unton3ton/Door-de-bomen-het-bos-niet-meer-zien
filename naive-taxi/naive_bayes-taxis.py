# https://fritz.ai/naive-bayes-classifier-in-python-using-scikit-learn/
# https://www.geeksforgeeks.org/gaussian-naive-bayes-using-sklearn/

# python naive_bayes-taxis.py -d taxis.csv

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import argparse
from sklearn.metrics import (accuracy_score, 
							confusion_matrix, 
							classification_report, 
							matthews_corrcoef,
							roc_curve, 
							roc_auc_score,
							auc,
							cohen_kappa_score)
 
parser = argparse.ArgumentParser(description="Пример использования консольного интерфейса")
parser.add_argument("-d", "--data", help="Введите (путь и/или) имя data.csv", required=True)
 
# args = parser.parse_args()
 
# no_space_csv = args.nospace # 'noSpace-v7.csv'
# space_csv = args.space # 'space-v7.csv'

# df = pd.read_csv(space_csv)
# n_balance = df.shape[0] # 3176 - количество строк в space.csv

# df = pd.read_csv(no_space_csv)
# '''Выберите n строк случайным образом, используя sample(n) или sample(n=n). 
# Каждый раз, когда вы выполняете это, вы получаете n разных строк.'''
# df = df.sample(n=n_balance)
# df.to_csv('notspace-v7.csv', index=False)

# # merging two csv files 
# dataframe = pd.concat( 
# 	map(pd.read_csv, [space_csv, 'notspace-v7.csv']), ignore_index=True)

parser = argparse.ArgumentParser(description="Пример использования консольного интерфейса")
parser.add_argument("-d", "--data", help="Введите (путь и/или) имя data.csv", required=True)
 
args = parser.parse_args()
 
taxi_csv = args.data # 'taxis.csv'

dataframe = pd.read_csv(taxi_csv) 

print(dataframe.head(), '\n')
# print(dataframe.shape, '\n')
print(dataframe.info(), '\n')
print(dataframe.describe(), '\n')

#specify that all columns should be shown
pd.set_option('display.max_columns', None)
print(dataframe.describe(include="all"), '\n') 

# for columnname in list(dataframe.columns):
# 	print(dataframe[columnname].describe(), '\n')

print(dataframe['payment'].value_counts())

target = 'payment'

'''luck: 
Теперь мы можем перебирать каждый столбец с помощью apply и sample с заменой из не пропущенных значений.
https://stackoverflow.com/questions/46384934/pandas-replace-nan-using-random-sampling-of-column-values'''
dataframe = dataframe.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))

dataframe = dataframe.dropna()
print(dataframe.shape) # diff = 218721-201671 = 17050

print(dataframe.isna().sum(), '\n')
print(dataframe['payment'].value_counts())

categorical_features = ['pickup', 
						'dropoff',
						'color', 
						'pickup_zone', 
						'dropoff_zone', 
						'pickup_borough',
                        'dropoff_borough']
dataframe = dataframe.drop(categorical_features, axis=1)

X = dataframe.drop(target, axis=1)
y = dataframe[target]
y = y.map({'cash': 1, 'credit card': 0}).astype(int) # для построения ROC-кривой и использования CrossEntropy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
y_pred_proba_positive = y_pred_proba[:, 1]
print(f'y_predict = \n{y_pred}')

iso_reg = IsotonicRegression(y_min = 0, y_max = 1, out_of_bounds = 'clip').fit(y_pred_proba_positive, y_test)
proba_test_forest_isoreg = iso_reg.predict(model.predict_proba(X_test)[:, 1])

X_test['predicted'] = y_pred
X_test['probability cash %'] = np.round_(y_pred_proba_positive*100, decimals = 2)
X_test['calibration isoreg probability cash %'] = np.round_(proba_test_forest_isoreg*100, decimals = 2)
print('\nX_test+predicted\n', X_test.head(11), '\n')
X_test.to_csv('cash-naive-predicted.csv')

# reliability diagram
fop, mpv = calibration_curve(y_test, y_pred_proba_positive, n_bins=10)
# plot perfectly calibrated
plt.figure(figsize=(7, 4))
plt.plot([0, 1], [0, 1], linestyle='--', label = 'Perfect calibration')
# plot model reliability
plt.plot(mpv, fop, marker='.', label = 'Real calibration')
plt.title('Calibration curve')
plt.xlabel('Cреднее значение прогнозируемой вероятности для ячейки')
plt.ylabel('Cреднее значение целевой переменной для ячейки')
plt.grid()
plt.legend()
plt.savefig('calibration_curve_naive.png')

fop, mpv = calibration_curve(y_test, proba_test_forest_isoreg, n_bins=10)
# plot perfectly calibrated
plt.figure(figsize=(7, 4))
plt.plot([0, 1], [0, 1], linestyle='--', label = 'Perfect calibration')
# plot model reliability
plt.plot(mpv, fop, marker='.', label = 'Real calibration')
plt.title('Calibration curve with Isotonic Regression')
plt.xlabel('Cреднее значение прогнозируемой вероятности для ячейки')
plt.ylabel('Cреднее значение целевой переменной для ячейки')
plt.grid()
plt.legend()
plt.savefig('calibration_curve_forest_isoreg_naive.png')

def expected_calibration_error(y, proba, bins = 'fd'): 
	'''с правилом Фридмана-Диакониса по умолчанию
	(статистическое правило, предназначенное для определения количества ячеек, 
	которое делает гистограмму максимально близкой к теоретическому 
	распределению вероятностей)'''
	bin_count, bin_edges = np.histogram(proba, bins = bins)
	n_bins = len(bin_count)
	bin_edges[0] -= 1e-8 # because left edge is not included
	bin_id = np.digitize(proba, bin_edges, right = True) - 1
	bin_ysum = np.bincount(bin_id, weights = y, minlength = n_bins)
	bin_probasum = np.bincount(bin_id, weights = proba, minlength = n_bins)
	bin_ymean = np.divide(bin_ysum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
	bin_probamean = np.divide(bin_probasum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
	ece = np.abs((bin_probamean - bin_ymean) * bin_count).sum() / len(proba)
	return ece

'''
ожидаемая ошибка калибровки (ЕСЕ) представляет собой средневзвешенное значение ошибок 
калибровки отдельных ячеек, где каждая ячейка весит пропорционально количеству наблюдений, 
которые она содержит
'''

print(f'Oжидаемая ошибка калибровки (ЕСЕ) naive_bayes = {expected_calibration_error(y_test, y_pred_proba_positive)}')
print(f'Oжидаемая ошибка калибровки (ЕСЕ) naive_bayes + Isotonic Regression = {expected_calibration_error(y_test, proba_test_forest_isoreg)}\n')

accuracy = accuracy_score(y_test,y_pred)*100
print(f'accuracy = {accuracy:.2f}%') # = 90.19%
Kappa = cohen_kappa_score(y_test, y_pred)
'''
Kappa - это фактический показатель соответствия между фактическими метками 
предсказания и фактическими метками в "получено". Но здесь также упоминается, 
что не следует забывать о другом вероятном результате – точном предсказании 
благодаря чистой случайности. Это также означало, что чем выше или чем ближе 
значение Kappa к 1, тем лучше соответствие между прогнозируемыми значениями 
и фактическими метками.
'''
print(f'Kappa = {Kappa:.2f}', '\n') # 0.97

# Plot the confusion matrix as a heatmap
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
# [[597   7]		True Positive (TP)	False Negative (FN)
#  [  4 298]]		False Positive (FP)	True Negative (TN)

# Коэффициент корреляции Мэтьюса (MCC) — это показатель, который мы можем использовать 
# для оценки эффективности модели классификации. 
# Он рассчитывается как: MCC = (TP*TN – FP*FN) / √ (TP+FP) (TP+FN) (TN+FP) (TN+FN) 
# TP: количество истинных положительных результатов 
# TN: количество истинных отрицательных результатов 
# FP: Количество ложных срабатываний 
# FN: количество ложноотрицательных результатов.

'''
True Positive, True Negative - истинное утверждение и отрицание соответственно; 
False Nagative - если факт отрицается, а на самом деле есть; 
False Positive - факт утверждается, на самом деле ничего не произошло.
'''

TP, FN, FP, TN = confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]
try:
	MCC = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
	print(f'TP = {TP}, FN = {FN}, FP = {FP}, TN = {TN}')
	print(f"Коэффициент корреляции Мэтьюса (MCC) = {MCC:.2f}") # = 0.97
except ZeroDivisionError:
	print("MCC = NaN")

print(f"Коэффициент корреляции Мэтьюса (sklearn.metrics) = {matthews_corrcoef(y_test, y_pred):.2f}") # 0.9727766584990545


plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', 
			xticklabels=['noSpace', 'Space'], 
			yticklabels=['noSpace', 'Space'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for cash')
# plt.show()
plt.savefig('Confusion_Space_Matrix_naive.png')


# Print the classification report
print("Classification Report for cash:")
print(classification_report(y_test, y_pred))

# Plot ROC curves for each class
class_labels = np.unique(y) # Get unique class labels
plt.figure(figsize=(8, 6))
all_fpr, all_tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
plt.plot(all_fpr, all_tpr, 'orange')

plt.xlabel(r'False Positive Rate ($FPR = \frac{FP}{FP + TN}$)')
'''
FPR представляет собой долю случаев, когда модель предсказала 
положительный исход, но фактический исход был отрицательным.
False positive rate является важным показателем при разработке 
и оценке моделей машинного обучения, особенно в ситуациях, где 
последствия ложного положительного предсказания могут быть серьёзными.

На высокую false positive rate в моделях машинного обучения могут 
влиять следующие факторы:
* Качество и баланс обучающих данных.
* Тип используемой модели.

Для снижения false positive rate важно тщательно выбирать и предварительно 
обрабатывать обучающие данные, выбирать подходящую модель для конкретной 
задачи и изменять порог модели для прогнозирования благоприятного исхода.
'''
plt.ylabel(r'True Positive Rate ($TPR = \frac{TP}{TP + FN}$)')
'''
TPR представляет собой пропорцию реальных положительных случаев, 
которые были правильно идентифицированы или классифицированы 
моделью как положительные. TPR также известен как чувствительность, 
отзывчивость или частота попадания.
'''
roc_auc = auc(all_fpr, all_tpr)
plt.title(f'ROC curves for taxi Naive Bayes-classification (area = {roc_auc:.2f})')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1.05)
# plt.show()
plt.savefig('ROC-taxi.png')
# https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/
print(f'ROC-AUC area = {roc_auc:.2f}')

'''
Результат без замены пустых ячеек (с их удалением)

payment
credit card    4577
cash           1812
Name: count, dtype: int64
(6341, 14)
pickup             0
dropoff            0
passengers         0
distance           0
fare               0
tip                0
tolls              0
total              0
color              0
payment            0
pickup_zone        0
dropoff_zone       0
pickup_borough     0
dropoff_borough    0
dtype: int64

payment
credit card    4546
cash           1795
Name: count, dtype: int64
y_predict =
[1 1 0 ... 0 1 0]

X_test+predicted
       passengers  distance   fare   tip  tolls  total  predicted  \
742            1      1.32   8.50  0.00   0.00  11.80          1
4824           1      2.90  17.00  0.00   0.00  20.30          1
3108           1      4.17  14.50  3.66   0.00  21.96          0
4985           1      4.03  21.50  2.50   0.00  27.30          0
219            1      1.40   7.00  2.25   0.00  13.55          0
4154           1      1.10   7.50  2.00   0.00  13.80          0
2280           1      1.00   6.00  0.70   0.00  10.00          0
5456           1     15.78  42.82  0.00   5.76  49.08          0
1684           6      2.39  10.50  2.96   0.00  17.76          0
4889           2      0.95   5.00  1.66   0.00   9.96          0
5395           1      3.37  18.50  3.00   0.00  24.80          0

      probability space %  calibration isoreg probability space %
742                 99.99                                   93.17
4824                99.99                                   85.90
3108                 0.00                                    0.00
4985                 0.00                                    0.00
219                  0.00                                    0.00
4154                 0.00                                    0.00
2280                 0.00                                    0.00
5456                11.03                                   20.00
1684                 0.00                                    0.00
4889                 0.00                                    0.00
5395                 0.00                                    0.00

Oжидаемая ошибка калибровки (ЕСЕ) naive_bayes = 0.05713218345806964
Oжидаемая ошибка калибровки (ЕСЕ) naive_bayes + Isotonic Regression = 9.376310252690274e-16

accuracy = 94.17%
Kappa = 0.87

[[833  71]
 [  3 362]]
TP = 833, FN = 71, FP = 3, TN = 362
Коэффициент корреляции Мэтьюса (MCC) = 0.87
Коэффициент корреляции Мэтьюса (sklearn.metrics) = 0.87
Classification Report for cash:
              precision    recall  f1-score   support

           0       1.00      0.92      0.96       904
           1       0.84      0.99      0.91       365

    accuracy                           0.94      1269
   macro avg       0.92      0.96      0.93      1269
weighted avg       0.95      0.94      0.94      1269

ROC-AUC area = 0.96
'''

'''
со случайной заменой

payment
credit card    4577
cash           1812
Name: count, dtype: int64
(6433, 14)
pickup             0
dropoff            0
passengers         0
distance           0
fare               0
tip                0
tolls              0
total              0
color              0
payment            0
pickup_zone        0
dropoff_zone       0
pickup_borough     0
dropoff_borough    0
dtype: int64

payment
credit card    4602
cash           1831
Name: count, dtype: int64
y_predict =
[0 1 1 ... 0 1 1]

X_test+predicted
       passengers  distance  fare   tip  tolls  total  predicted  \
4092           0      1.30   9.0  2.66    0.0  15.96          0
6282           1      1.40   8.5  0.00    0.0   9.80          1
3237           2      2.39  14.0  0.00    0.0  17.80          1
1891           1      1.90  10.5  2.85    0.0  17.15          0
5010           1      2.40  11.5  1.70    0.0  17.00          0
2168           1      4.33  25.5  7.20    0.0  36.00          0
5154           1      0.84   5.5  1.76    0.0  10.56          0
5202           1      2.80  17.5  4.15    0.0  24.95          0
247            1      2.75  13.0  3.46    0.0  20.76          0
198            1      0.20   3.0  0.00    0.0   7.30          1
2977           1      1.30   5.5  0.00    0.0   8.80          1

      probability space %  calibration isoreg probability space %
4092                 0.00                                    0.00
6282                99.99                                   88.89
3237                99.98                                   75.68
1891                 0.00                                    0.00
5010                 0.00                                    0.00
2168                 0.00                                    0.00
5154                 0.00                                    0.00
5202                 0.00                                    0.00
247                  0.00                                    0.00
198                 99.99                                   90.77
2977                99.99                                   89.38

Oжидаемая ошибка калибровки (ЕСЕ) naive_bayes = 0.08039630214074049
Oжидаемая ошибка калибровки (ЕСЕ) naive_bayes + Isotonic Regression = 2.2592650361253186e-16

accuracy = 91.84%
Kappa = 0.82

[[809 101]
 [  4 373]]
TP = 809, FN = 101, FP = 4, TN = 373
Коэффициент корреляции Мэтьюса (MCC) = 0.83
Коэффициент корреляции Мэтьюса (sklearn.metrics) = 0.83
Classification Report for cash:
              precision    recall  f1-score   support

           0       1.00      0.89      0.94       910
           1       0.79      0.99      0.88       377

    accuracy                           0.92      1287
   macro avg       0.89      0.94      0.91      1287
weighted avg       0.93      0.92      0.92      1287

ROC-AUC area = 0.94
'''