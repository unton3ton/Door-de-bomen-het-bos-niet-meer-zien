# source CAT/bin/activate
# conda activate CAT

# python .\taxi-katboost-all_metrics.py -d .\taxis.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, 
							confusion_matrix, 
							classification_report, 
							matthews_corrcoef,
							roc_curve, 
							roc_auc_score,
							auc,
							cohen_kappa_score)
from math import sqrt
from catboost import CatBoostClassifier, Pool

import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

# no_space_csv = 'noSpace-v7.csv'
# space_csv = 'space-v7.csv'

# df = pd.read_csv(space_csv)
# n_balance = df.shape[0] # 3176 - количество строк в space.csv

# df = pd.read_csv(no_space_csv)
# df = df.iloc[0:n_balance, :] # выбираем количество строк = в space.csv для сбалансированности
# df.to_csv('notspace-v7.csv', index=False)

# # merging two csv files 
# dataframe = pd.concat( 
# 	map(pd.read_csv, [space_csv, 'notspace-v7.csv']), ignore_index=True) 


'''
python .\taxi-katboost-all_metrics.py -h
usage: taxi-katboost-all_metrics.py [-h] -d DATA

Пример использования консольного интерфейса

options:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Введите (путь и/или) имя data.csv
'''

 
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
# dataframe = dataframe.dropna()
# print(dataframe['payment'].value_counts())
# # до dropna()
# payment
# credit card    4577
# cash           1812
# # после dropna()
# payment
# credit card    4546
# cash           1795



corr_matrix = dataframe[['passengers', 'distance', 'fare', 'tip', 'tolls', 'total']]
print(corr_matrix.corr(), '\n')

plt.figure(figsize=(13, 13))
sns.heatmap(corr_matrix.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Taxi Matrix')
plt.savefig('corr_taxi_matrix.png')
# plt.show()


# dataframe['radiusGL'] = dataframe['radiusGL'].fillna(dataframe['radiusGL'].mean()) # замена NaN на среднее по столбцу
# dataframe['limitDown'] = dataframe['limitDown'].fillna(dataframe['limitDown'].min())
# dataframe['limitUp'] = dataframe['limitUp'].fillna(dataframe['limitUp'].max())
# dataframe['long'] = dataframe['long'].fillna(dataframe['long'].mode()[0]) # замена NaN на моду по столбцу
# dataframe['scope'] = dataframe['scope'].fillna(" ") # замена NaN на пустую строку (или любое заданное значение)


print(dataframe.isna().sum(), '\n')
# pickup              0
# dropoff             0
# passengers          0
# distance            0
# fare                0
# tip                 0
# tolls               0
# total               0
# color               0
# payment            44
# pickup_zone        26
# dropoff_zone       45
# pickup_borough     26
# dropoff_borough    45


target = 'payment'

dataframe = dataframe.dropna()
# print(dataframe.shape) # diff = 218721-201671 = 17050


# Create the feature matrix (X) and target vector (y)
X = dataframe.drop(target, axis=1)

y = dataframe[target]
y = y.map({'cash': 1, 'credit card': 0}).astype(int) # для построения ROC-кривой и использования CrossEntropy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# specifying categorical features
categorical_features = ['pickup', 
						'dropoff',
						'color', 
						'pickup_zone', 
						'dropoff_zone', 
						'pickup_borough',
                        'dropoff_borough']
# create and train the CatBoostClassifier
# https://www.geeksforgeeks.org/catboost-tree-parameters/
# model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.1, cat_features=categorical_features,
#                            loss_function='CrossEntropy', # or 'LogLoss'
#                            custom_metric=['Accuracy', 'AUC'], random_seed=42)
# model.fit(X_train, y_train)


train_data = Pool(data=X_train, label=y_train, cat_features=categorical_features)
test_data = Pool(data=X_test, label=y_test, cat_features=categorical_features)
class_counts = np.bincount(y_train)
 
model = CatBoostClassifier(iterations=500,  # Number of boosting iterations
                           learning_rate=0.1,  # Learning rate
                           depth=8,  # Depth of the tree
                           verbose=100,  # Print training progress every 50 iterations
                           early_stopping_rounds=10,  # stops training if no improvement in 10 consequtive rounds
                           loss_function='CrossEntropy',
                           custom_metric=['Accuracy', 'AUC'], random_seed=42)  # used for Multiclass classification tasks
 
# Train the CatBoost model and collect training progress data
model.fit(train_data, eval_set=test_data)

'''
0:      learn: 0.4782965        test: 0.4736305 best: 0.4736305 (0)     total: 162ms    remaining: 1m 21s
Stopped by overfitting detector  (10 iterations wait)

bestTest = 0.09514272028
bestIteration = 68
'''
 
# Extract the loss values from the evals_result_ dictionary
evals_result = model.get_evals_result()
train_loss = evals_result['learn']['CrossEntropy']
test_loss = evals_result['validation']['CrossEntropy']

# Plot the training progress
iterations = np.arange(1, len(train_loss) + 1)
 
plt.figure(figsize=(7, 4))
plt.plot(iterations, train_loss, label='Training Loss', color='blue')
plt.plot(iterations, test_loss, label='Validation Loss', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Loss (CrossEntropy)')
plt.title('CatBoost Taxi Training Progress')
plt.legend()
plt.grid()
# plt.show()
plt.savefig('Training_Progress_Taxi_Matrix.png')


model.save_model('catboost_classification_taxi.model')


model_name = CatBoostClassifier()      # parameters not required.
model_name.load_model('catboost_classification_taxi.model')


y_pred = model_name.predict(X_test) # predicting accuracy
print('\nX_test\n', X_test.head(11), '\n')
X_test['predicted'] = y_pred
y_pred = model_name.predict(X_test) # predicting accuracy
y_pred_proba = model_name.predict_proba(X_test)
y_pred_proba_positive = y_pred_proba[:, 1] # is used to keep the probability for positive outcome only
# print(y_pred_proba)
X_test['predicted'] = y_pred
X_test['probability space %'] = np.round_(y_pred_proba_positive*100, decimals = 2)
'''
калибровка – это функция, которая преобразует одномерный вектор (некалиброванных вероятностей) 
в другой одномерный вектор (калиброванных вероятностей).
'''
log_reg = LogisticRegression().fit(y_pred_proba_positive.reshape(-1, 1), y_test)
proba_test_forest_logreg = log_reg.predict_proba(model_name.predict_proba(X_test)[:, 1].reshape(-1, 1))[:, 1]
X_test['calibration logreg probability space %'] = np.round_(proba_test_forest_logreg*100, decimals = 2)
'''
Изотоническая регрессия.
Непараметрический алгоритм, который подгоняет неубывающую линию свободной 
формы под данные. 
Тот факт, что линия неубывающая, является основополагающим, поскольку так 
учитывается исходная сортировка.
'''
iso_reg = IsotonicRegression(y_min = 0, y_max = 1, out_of_bounds = 'clip').fit(y_pred_proba_positive, y_test)
proba_test_forest_isoreg = iso_reg.predict(model_name.predict_proba(X_test)[:, 1])

# X_test['calibration'] = proba_test_forest_isoreg == y_pred
# X_test['calibration predicted'] = proba_test_forest_isoreg
X_test['calibration isoreg probability space %'] = np.round_(proba_test_forest_isoreg*100, decimals = 2)

print('\nX_test+predicted\n', X_test.head(11), '\n')


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
plt.savefig('calibration_curve.png')

fop, mpv = calibration_curve(y_test, proba_test_forest_logreg, n_bins=10)
# plot perfectly calibrated
plt.figure(figsize=(7, 4))
plt.plot([0, 1], [0, 1], linestyle='--', label = 'Perfect calibration')
# plot model reliability
plt.plot(mpv, fop, marker='.', label = 'Real calibration')
plt.title('Calibration curve with Logistic Regression')
plt.xlabel('Cреднее значение прогнозируемой вероятности для ячейки')
plt.ylabel('Cреднее значение целевой переменной для ячейки')
plt.grid()
plt.legend()
plt.savefig('calibration_curve_forest_logreg.png')

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
plt.savefig('calibration_curve_forest_isoreg.png')

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

print(f'Oжидаемая ошибка калибровки (ЕСЕ) catboost = {expected_calibration_error(y_test, y_pred_proba_positive)}')
print(f'Oжидаемая ошибка калибровки (ЕСЕ) catboost + Logistic Regression = {expected_calibration_error(y_test, proba_test_forest_logreg)}')
print(f'Oжидаемая ошибка калибровки (ЕСЕ) catboost + Isotonic Regression = {expected_calibration_error(y_test, proba_test_forest_isoreg)}\n')

# saving the dataframe
X_test.to_csv('taxi-predicted.csv')
'''
                        dropoff_zone pickup_borough dropoff_borough  predicted
742                      Murray Hill      Manhattan       Manhattan          1 cash
4824                 Lenox Hill East      Manhattan       Manhattan          1
3108                  Midtown Center      Manhattan       Manhattan          0 credit card
4985             Little Italy/NoLiTa      Manhattan       Manhattan          0
219                    Alphabet City      Manhattan       Manhattan          0
4154    Penn Station/Madison Sq West      Manhattan       Manhattan          0
2280                  Yorkville East      Manhattan       Manhattan          0
5456                     Marble Hill         Queens       Manhattan          0
1684  Long Island City/Hunters Point      Manhattan          Queens          0
4889                Manhattan Valley      Manhattan       Manhattan          0
5395                        Union Sq      Manhattan       Manhattan          0
'''


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") # Accuracy: 0.99

Kappa = cohen_kappa_score(y_test, y_pred)
'''
Kappa - это фактический показатель соответствия между фактическими метками 
предсказания и фактическими метками в "получено". Но здесь также упоминается, 
что не следует забывать о другом вероятном результате – точном предсказании 
благодаря чистой случайности. Это также означало, что чем выше или чем ближе 
значение Kappa к 1, тем лучше соответствие между прогнозируемыми значениями 
и фактическими метками.
'''
print(f'Kappa = {Kappa:.2f}', '\n') # 0.93 = 0.9262691828096996

# Plot the confusion matrix as a heatmap
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
# [[356   9]		True Positive (TP)	False Negative (FN)
#  [ 31 873]]		False Positive (FP)	True Negative (TN)

# Коэффициент корреляции Мэтьюса (MCC) — это показатель, который мы можем использовать 
# для оценки эффективности модели классификации. 
# Он рассчитывается как: MCC = (TP*TN – FP*FN) / √ (TP+FP) (TP+FN) (TN+FP) (TN+FN) 
# TP: количество истинных положительных результатов 
# TN: количество истинных отрицательных результатов 
# FP: Количество ложных срабатываний 
# FN: количество ложноотрицательных результатов.

'''
TP = 356, FN = 9, FP = 31, TN = 873

True Positive, True Negative - истинное утверждение и отрицание соответственно; 
False Nagative - если факт отрицается, а на самом деле есть; 
False Positive - факт утверждается, на самом деле ничего не произошло.
'''

TP, FN, FP, TN = confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]
try:
	MCC = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
	print(f'TP = {TP}, FN = {FN}, FP = {FP}, TN = {TN}')
	print(f"Коэффициент корреляции Мэтьюса (MCC) = {MCC:.2f}") # = 0.93
except ZeroDivisionError:
	print("MCC = NaN")

print(f"Коэффициент корреляции Мэтьюса (sklearn.metrics) = {matthews_corrcoef(y_test, y_pred):.2f}") # 0.9727766584990545


plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
			xticklabels=['credit card', 'cash'], 
			yticklabels=['credit card', 'cash'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for taxi')
# plt.show()
plt.savefig('Confusion_Taxi_Matrix.png')



importances = model_name.get_feature_importance()
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]
 
plt.figure(figsize=(15, 12))
plt.bar(range(len(feature_names)), importances[sorted_indices])
plt.xticks(range(len(feature_names)), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importance for taxi (%)")
# plt.show()
plt.grid()
plt.savefig('Feature_taxi.png')


print(f'\n\nsum(importances[sorted_indices]) = {importances[sorted_indices].sum():.0f}%\n\n')


print("Classification Report for taxi:")
print(classification_report(y_test, y_pred))
# https://habr.com/ru/articles/821547/#classification_metrics
# Classification Report for taxi:
#               precision    recall  f1-score   support

#         cash       0.92      0.98      0.95       365
#  credit card       0.99      0.97      0.98       904

#     accuracy                           0.97      1269
#    macro avg       0.95      0.97      0.96      1269
# weighted avg       0.97      0.97      0.97      1269

'''
0. Precision :
Характеризует долю правильно предсказанных положительных классов среди 
всех образцов, которые модель спрогнозировала как положительный класс.

1. Точность : процент правильных положительных прогнозов по отношению 
к общему количеству положительных прогнозов.

2. Отзыв recall : процент правильных положительных прогнозов по отношению 
к общему количеству фактических положительных результатов.

3. Оценка F1 : средневзвешенное гармоническое значение точности и полноты. 
Чем ближе к 1, тем лучше модель.

Оценка F1: 2 * (Точность * Отзыв) / (Точность + Отзыв)

4. Поддержка support: эти значения просто говорят нам, сколько "игроков" принадлежало 
к каждому классу в тестовом наборе данных.

5. Макро-усреднение (macro-averaging) представляет собой среднее арифметическое 
подсчитанной метрики для каждого класса и используется при дисбалансе классов, 
когда важен каждый класс. В таком случае все классы учитываются равномерно 
независимо от их размера.

6. Взвешенное усреднение (weighted averaging) рассчитывается как взвешенное среднее 
и также применяется в случае дисбаланса классов, но только когда важность класса 
учитывается в зависимости от количества объектов с таким классом, то есть когда 
важны наибольшие классы. При таком подходе важность каждого класса учитывается с 
присвоением им весов. Вес класса w_k может устанавливаться по-разному, например, 
как доля примеров этого класса в обучающей выборке.
'''

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
plt.title(f'ROC curves for taxi CatBoost-classification (area = {roc_auc:.2f})')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1.05)
# plt.show()
plt.savefig('ROC-taxi.png')
# https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/
print(f'ROC-AUC area = {roc_auc:.2f}') # = 0.99

'''
В идеальном случае ROC-кривая будет стремиться в верхний левый угол (TPR=1 и FPR=0), 
а площадь под ней (AUC) будет равна 1. 
При значении площади 0.5 качество прогнозов модели будет сопоставимо случайному 
угадыванию, ну а если это значение меньше 0.5, то, модель лучше предсказывает 
результаты, противоположные истинным — в таком случае нужно просто поменять 
целевые метки местами для получения площади больше 0.5.
'''

'''
                pickup              dropoff  passengers  ...           dropoff_zone  pickup_borough  dropoff_borough
0  2019-03-23 20:21:09  2019-03-23 20:27:24           1  ...    UN/Turtle Bay South       Manhattan        Manhattan
1  2019-03-04 16:11:55  2019-03-04 16:19:00           1  ...  Upper West Side South       Manhattan        Manhattan
2  2019-03-27 17:53:01  2019-03-27 18:00:25           1  ...           West Village       Manhattan        Manhattan
3  2019-03-10 01:23:59  2019-03-10 01:49:51           1  ...         Yorkville West       Manhattan        Manhattan
4  2019-03-30 13:27:42  2019-03-30 13:37:14           3  ...         Yorkville West       Manhattan        Manhattan

[5 rows x 14 columns]

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6433 entries, 0 to 6432
Data columns (total 14 columns):
 #   Column           Non-Null Count  Dtype
---  ------           --------------  -----
 0   pickup           6433 non-null   object
 1   dropoff          6433 non-null   object
 2   passengers       6433 non-null   int64
 3   distance         6433 non-null   float64
 4   fare             6433 non-null   float64
 5   tip              6433 non-null   float64
 6   tolls            6433 non-null   float64
 7   total            6433 non-null   float64
 8   color            6433 non-null   object
 9   payment          6389 non-null   object
 10  pickup_zone      6407 non-null   object
 11  dropoff_zone     6388 non-null   object
 12  pickup_borough   6407 non-null   object
 13  dropoff_borough  6388 non-null   object
dtypes: float64(5), int64(1), object(8)
memory usage: 703.7+ KB
None

        passengers     distance         fare         tip        tolls        total
count  6433.000000  6433.000000  6433.000000  6433.00000  6433.000000  6433.000000
mean      1.539251     3.024617    13.091073     1.97922     0.325273    18.517794
std       1.203768     3.827867    11.551804     2.44856     1.415267    13.815570
min       0.000000     0.000000     1.000000     0.00000     0.000000     1.300000
25%       1.000000     0.980000     6.500000     0.00000     0.000000    10.800000
50%       1.000000     1.640000     9.500000     1.70000     0.000000    14.160000
75%       2.000000     3.210000    15.000000     2.80000     0.000000    20.300000
max       6.000000    36.700000   150.000000    33.20000    24.020000   174.820000

                     pickup              dropoff   passengers     distance  \
count                  6433                 6433  6433.000000  6433.000000
unique                 6414                 6425          NaN          NaN
top     2019-03-13 10:57:06  2019-03-04 18:08:13          NaN          NaN
freq                      2                    2          NaN          NaN
mean                    NaN                  NaN     1.539251     3.024617
std                     NaN                  NaN     1.203768     3.827867
min                     NaN                  NaN     0.000000     0.000000
25%                     NaN                  NaN     1.000000     0.980000
50%                     NaN                  NaN     1.000000     1.640000
75%                     NaN                  NaN     2.000000     3.210000
max                     NaN                  NaN     6.000000    36.700000

               fare         tip        tolls        total   color  \
count   6433.000000  6433.00000  6433.000000  6433.000000    6433
unique          NaN         NaN          NaN          NaN       2
top             NaN         NaN          NaN          NaN  yellow
freq            NaN         NaN          NaN          NaN    5451
mean      13.091073     1.97922     0.325273    18.517794     NaN
std       11.551804     2.44856     1.415267    13.815570     NaN
min        1.000000     0.00000     0.000000     1.300000     NaN
25%        6.500000     0.00000     0.000000    10.800000     NaN
50%        9.500000     1.70000     0.000000    14.160000     NaN
75%       15.000000     2.80000     0.000000    20.300000     NaN
max      150.000000    33.20000    24.020000   174.820000     NaN

            payment     pickup_zone           dropoff_zone pickup_borough  \
count          6389            6407                   6388           6407
unique            2             194                    203              4
top     credit card  Midtown Center  Upper East Side North      Manhattan
freq           4577             230                    245           5268
mean            NaN             NaN                    NaN            NaN
std             NaN             NaN                    NaN            NaN
min             NaN             NaN                    NaN            NaN
25%             NaN             NaN                    NaN            NaN
50%             NaN             NaN                    NaN            NaN
75%             NaN             NaN                    NaN            NaN
max             NaN             NaN                    NaN            NaN

       dropoff_borough
count             6388
unique               5
top          Manhattan
freq              5206
mean               NaN
std                NaN
min                NaN
25%                NaN
50%                NaN
75%                NaN
max                NaN

payment
credit card    4577
cash           1812
Name: count, dtype: int64
            passengers  distance      fare       tip     tolls     total
passengers    1.000000  0.009411  0.007637  0.021099 -0.002903  0.015708
distance      0.009411  1.000000  0.920108  0.452589  0.635267  0.904676
fare          0.007637  0.920108  1.000000  0.488612  0.609307  0.974358
tip           0.021099  0.452589  0.488612  1.000000  0.413619  0.646186
tolls        -0.002903  0.635267  0.609307  0.413619  1.000000  0.683142
total         0.015708  0.904676  0.974358  0.646186  0.683142  1.000000

pickup              0
dropoff             0
passengers          0
distance            0
fare                0
tip                 0
tolls               0
total               0
color               0
payment            44
pickup_zone        26
dropoff_zone       45
pickup_borough     26
dropoff_borough    45
dtype: int64

0:      learn: 0.4782965        test: 0.4736305 best: 0.4736305 (0)     total: 162ms    remaining: 1m 20s
Stopped by overfitting detector  (10 iterations wait)

bestTest = 0.09514272028
bestIteration = 68

Shrink model to first 69 iterations.

X_test
                    pickup              dropoff  passengers  distance   fare  \
742   2019-03-05 09:52:36  2019-03-05 10:03:47           1      1.32   8.50
4824  2019-03-17 13:16:13  2019-03-17 13:40:32           1      2.90  17.00
3108  2019-03-14 01:33:26  2019-03-14 01:45:58           1      4.17  14.50
4985  2019-03-05 11:41:34  2019-03-05 12:12:45           1      4.03  21.50
219   2019-03-08 18:08:03  2019-03-08 18:15:42           1      1.40   7.00
4154  2019-03-11 18:57:33  2019-03-11 19:06:21           1      1.10   7.50
2280  2019-03-17 12:10:05  2019-03-17 12:15:02           1      1.00   6.00
5456  2019-03-12 21:11:03  2019-03-12 21:41:36           1     15.78  42.82
1684  2019-03-28 19:53:03  2019-03-28 20:04:10           6      2.39  10.50
4889  2019-03-28 12:24:30  2019-03-28 12:28:44           2      0.95   5.00
5395  2019-03-02 19:36:34  2019-03-02 20:05:06           1      3.37  18.50

       tip  tolls  total   color              pickup_zone  \
742   0.00   0.00  11.80  yellow  Greenwich Village North
4824  0.00   0.00  20.30  yellow             Clinton East
3108  3.66   0.00  21.96  yellow          Lower East Side
4985  2.50   0.00  27.30  yellow             Clinton West
219   2.25   0.00  13.55  yellow                 Kips Bay
4154  2.00   0.00  13.80  yellow             Midtown East
2280  0.70   0.00  10.00  yellow          Lenox Hill East
5456  0.00   5.76  49.08   green                  Maspeth
1684  2.96   0.00  17.76  yellow           Midtown Center
4889  1.66   0.00   9.96  yellow    Upper West Side North
5395  3.00   0.00  24.80  yellow    Upper East Side South

                        dropoff_zone pickup_borough dropoff_borough
742                      Murray Hill      Manhattan       Manhattan
4824                 Lenox Hill East      Manhattan       Manhattan
3108                  Midtown Center      Manhattan       Manhattan
4985             Little Italy/NoLiTa      Manhattan       Manhattan
219                    Alphabet City      Manhattan       Manhattan
4154    Penn Station/Madison Sq West      Manhattan       Manhattan
2280                  Yorkville East      Manhattan       Manhattan
5456                     Marble Hill         Queens       Manhattan
1684  Long Island City/Hunters Point      Manhattan          Queens
4889                Manhattan Valley      Manhattan       Manhattan
5395                        Union Sq      Manhattan       Manhattan


X_test+predicted
                    pickup              dropoff  passengers  distance   fare  \
742   2019-03-05 09:52:36  2019-03-05 10:03:47           1      1.32   8.50
4824  2019-03-17 13:16:13  2019-03-17 13:40:32           1      2.90  17.00
3108  2019-03-14 01:33:26  2019-03-14 01:45:58           1      4.17  14.50
4985  2019-03-05 11:41:34  2019-03-05 12:12:45           1      4.03  21.50
219   2019-03-08 18:08:03  2019-03-08 18:15:42           1      1.40   7.00
4154  2019-03-11 18:57:33  2019-03-11 19:06:21           1      1.10   7.50
2280  2019-03-17 12:10:05  2019-03-17 12:15:02           1      1.00   6.00
5456  2019-03-12 21:11:03  2019-03-12 21:41:36           1     15.78  42.82
1684  2019-03-28 19:53:03  2019-03-28 20:04:10           6      2.39  10.50
4889  2019-03-28 12:24:30  2019-03-28 12:28:44           2      0.95   5.00
5395  2019-03-02 19:36:34  2019-03-02 20:05:06           1      3.37  18.50

       tip  tolls  total   color              pickup_zone  \
742   0.00   0.00  11.80  yellow  Greenwich Village North
4824  0.00   0.00  20.30  yellow             Clinton East
3108  3.66   0.00  21.96  yellow          Lower East Side
4985  2.50   0.00  27.30  yellow             Clinton West
219   2.25   0.00  13.55  yellow                 Kips Bay
4154  2.00   0.00  13.80  yellow             Midtown East
2280  0.70   0.00  10.00  yellow          Lenox Hill East
5456  0.00   5.76  49.08   green                  Maspeth
1684  2.96   0.00  17.76  yellow           Midtown Center
4889  1.66   0.00   9.96  yellow    Upper West Side North
5395  3.00   0.00  24.80  yellow    Upper East Side South

                        dropoff_zone pickup_borough dropoff_borough  \
742                      Murray Hill      Manhattan       Manhattan
4824                 Lenox Hill East      Manhattan       Manhattan
3108                  Midtown Center      Manhattan       Manhattan
4985             Little Italy/NoLiTa      Manhattan       Manhattan
219                    Alphabet City      Manhattan       Manhattan
4154    Penn Station/Madison Sq West      Manhattan       Manhattan
2280                  Yorkville East      Manhattan       Manhattan
5456                     Marble Hill         Queens       Manhattan
1684  Long Island City/Hunters Point      Manhattan          Queens
4889                Manhattan Valley      Manhattan       Manhattan
5395                        Union Sq      Manhattan       Manhattan

      predicted  probability space %  calibration logreg probability space %  \
742           1                92.29                                   93.86
4824          1                87.55                                   91.40
3108          0                 0.02                                    1.24
4985          0                 0.03                                    1.24
219           0                 0.01                                    1.24
4154          0                 0.01                                    1.24
2280          0                 0.07                                    1.24
5456          0                 9.28                                    2.49
1684          0                 0.20                                    1.25
4889          0                 0.01                                    1.24
5395          0                 0.02                                    1.24

      calibration isoreg probability space %
742                                    93.78
4824                                   93.78
3108                                    0.00
4985                                    0.00
219                                     0.00
4154                                    0.00
2280                                    0.00
5456                                    4.76
1684                                    0.00
4889                                    0.00
5395                                    0.00

Oжидаемая ошибка калибровки (ЕСЕ) catboost = 0.01692384317037953
Oжидаемая ошибка калибровки (ЕСЕ) catboost + Logistic Regression = 0.016988213514373282
Oжидаемая ошибка калибровки (ЕСЕ) catboost + Isotonic Regression = 1.2642019468741932e-16

Accuracy: 0.97
Kappa = 0.93

[[872  32]
 [  5 360]]
TP = 872, FN = 32, FP = 5, TN = 360
Коэффициент корреляции Мэтьюса (MCC) = 0.93
Коэффициент корреляции Мэтьюса (sklearn.metrics) = 0.93


sum(importances[sorted_indices]) = 100%


Classification Report for taxi:
              precision    recall  f1-score   support

           0       0.99      0.96      0.98       904
           1       0.92      0.99      0.95       365

    accuracy                           0.97      1269
   macro avg       0.96      0.98      0.97      1269
weighted avg       0.97      0.97      0.97      1269

ROC-AUC area = 0.98
'''