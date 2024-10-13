# python luck-katboost-imba-for_alles.py -s space-october.csv -n noSpace-october.csv
# python luck-katboost-imba-for_alles.py -n noSpace-all.csv -s space-all.csv

'''
Вызов справочной инфы:

python luck-katboost-imba.py -h

usage: luck-katboost-imba.py [-h] -n NOSPACE -s SPACE

Пример использования консольного интерфейса

options:
  -h, --help            show this help message and exit
  -n NOSPACE, --nospace NOSPACE
                        Введите (путь и/или) имя nospace.csv
  -s SPACE, --space SPACE
                        Введите (путь и/или) имя space.csv
'''

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
							cohen_kappa_score,
							balanced_accuracy_score,
							f1_score)
from math import sqrt
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import argparse
import pickle


ID_test = 'id' # 'notamNumber' # название нужного столбца с id natam'а 
 
parser = argparse.ArgumentParser(description="Пример использования консольного интерфейса")
parser.add_argument("-n", "--nospace", help="Введите (путь и/или) имя nospace.csv", required=True)
parser.add_argument("-s", "--space", help="Введите (путь и/или) имя space.csv", required=True)
 
args = parser.parse_args()
 
no_space_csv = args.nospace 
space_csv = args.space 

df = pd.read_csv(space_csv)
n_balance = df.shape[0] # 3176 - количество строк в space.csv

df = pd.read_csv(no_space_csv)

'''Выберите n строк случайным образом, используя sample(n) или sample(n=n). 
Каждый раз, когда вы выполняете это, вы получаете n разных строк.'''
df = df.sample(n=n_balance)
df.to_csv('notspace-old.csv', index=False)

dataframe = pd.concat( 
	map(pd.read_csv, [space_csv, no_space_csv]), ignore_index=True) 
	# map(pd.read_csv, [space_csv, 'notspace-old.csv']), ignore_index=True)

print(dataframe.head(), '\n')
print(dataframe.info(), '\n')
print(dataframe.describe(), '\n')

# pd.set_option('display.max_columns', None)
# print(dataframe.describe(include="all"), '\n') 

print(dataframe['isSpace'].value_counts())

corr_matrix = dataframe[['countointmore3',
                         'limitDown',
						 'limitUp',
						 'long',
						 'lat',
						 'radiusGL',
						 'dist',
						 'azm1',
						 'incline1',
                         'idSpaceport',
						 'isSpace']]
print(corr_matrix.corr(), '\n')

plt.figure(figsize=(13, 13))
sns.heatmap(corr_matrix.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Space Matrix')
plt.savefig('corr_spase_matrix_ID.png')

print(dataframe.isna().sum(), '\n')
'''luck: 
Теперь мы можем перебирать каждый столбец с помощью apply и sample с заменой из не пропущенных значений.
https://stackoverflow.com/questions/46384934/pandas-replace-nan-using-random-sampling-of-column-values'''
dataframe = dataframe.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))


print(dataframe.isna().sum(), '\n')

target = 'isSpace'

dataframe = dataframe.dropna()
print(dataframe.shape) 
print(dataframe.isna().sum(), '\n')
print(dataframe['isSpace'].value_counts())

X = dataframe.drop(target, axis=1) # Create the feature matrix (X) and target vector (y)
y = dataframe[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

flag_id = False
if ID_test in X_test.columns:
	flag_id = True
	ID = X_test[ID_test]
	X_train = X_train.drop(ID_test, axis=1)
	X_test = X_test.drop(ID_test, axis=1)

print('\nX_train \n', X_train.head(), '\n')

categorical_features = ['fir',
						'aerodrome',
						'subject',
						'condition',
						'traffic',
						'purpose',
						'scope',
                        'codeF',
                        'codeG']

train_data = Pool(data=X_train, label=y_train, cat_features=categorical_features)
test_data = Pool(data=X_test, label=y_test, cat_features=categorical_features)
class_counts = np.bincount(y_train)

# filename = 'search_the_best_learning-rate_and_depth-balanced_dataset.txt'
filename = 'search_the_best_learning-rate_and_depth.txt'

with open(filename, 'w') as f:

	for lrs in [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
		for depths in range(1,15):

			f.write(f'learning_rate={lrs}, depth={depths}\n\n')
 
			model = CatBoostClassifier(iterations=500,  # Number of boosting iterations
			                           learning_rate=lrs,  # Learning rate
			                           depth=depths,  # =8 Depth of the tree - Глубина деревьев. Регулирует сложность модели. Глубокие деревья могут лучше выявлять сложные зависимости, но также рискуют переобучиться.
			                           verbose=100,  # Print training progress every 50 iterations
			                           early_stopping_rounds=10,  # stops training if no improvement in 10 consequtive rounds
			                           loss_function='Logloss',
			                           custom_metric=['Accuracy', 'AUC'],
			                           use_best_model=True, 
			                           random_seed=42,
			                           auto_class_weights='Balanced')  # https://www.geeksforgeeks.org/handling-imbalanced-classes-in-catboost-techniques-and-solutions/?ysclid=m1ezogfwm1990689887


			model.fit(train_data, eval_set=test_data) # Train the CatBoost model and collect training progress data
			print(f"class_weights = {model.get_all_params()['class_weights']}")
			f.write(f"class_weights = {model.get_all_params()['class_weights']}\n")

			evals_result = model.get_evals_result() # Extract the loss values from the evals_result_ dictionary
			train_loss = evals_result['learn']['Logloss']
			test_loss = evals_result['validation']['Logloss']

			metrics = model.eval_metrics(test_data, 
			                             metrics = ['Logloss', 'CrossEntropy', 'Accuracy', 'PRAUC', 'AUC'], 
			                             plot = False)
			 
			logloss = metrics['Logloss'][-1] # Print the evaluation metrics
			evalAccuracy = metrics['Accuracy'][-1]
			auc_pr = metrics['PRAUC'][-1]
			eval_auc = metrics['AUC'][-1] # AUC измеряет, насколько хорошо модель способна различать классы.
			print(f'Log Loss (Cross-Entropy): {logloss:.2f}') # 0.03
			f.write(f'Log Loss (Cross-Entropy): {logloss:.2f}\n')
			print(f'Eval Accuracy: {evalAccuracy:.2f}') # 0.99
			f.write(f'Eval Accuracy: {evalAccuracy:.2f}\n')
			print(f'AUC-PR: {auc_pr:.2f}') # 1.00
			f.write(f'AUC-PR: {auc_pr:.2f}\n')
			print(f"AUC-ROC: {eval_auc:.2f}") # 1.00
			f.write(f"AUC-ROC: {eval_auc:.2f}\n")

			y_pred = model.predict(X_test) # predicting accuracy
			y_pred_proba = model.predict_proba(X_test)
			y_pred_proba_positive = y_pred_proba[:, 1] # is used to keep the probability for positive outcome only

			iso_reg = IsotonicRegression(y_min = 0, y_max = 1, out_of_bounds = 'clip').fit(y_pred_proba_positive, y_test)
			proba_test_forest_isoreg = iso_reg.predict(model.predict_proba(X_test)[:, 1])

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
			f.write(f'ECE = {expected_calibration_error(y_test, y_pred_proba_positive)}\n')
			print(f'Oжидаемая ошибка калибровки (ЕСЕ) catboost + Isotonic Regression = {expected_calibration_error(y_test, proba_test_forest_isoreg)}\n')
			f.write(f'ECE + Isotonic Regression = {expected_calibration_error(y_test, proba_test_forest_isoreg)}\n')

			accuracy = accuracy_score(y_test, y_pred)
			print(f"Accuracy: {accuracy*100:.4f}%")
			f.write(f"Accuracy: {accuracy*100:.4f}%\n")

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
			f.write(f'Kappa = {Kappa:.2f}\n')

			print(f"Коэффициент корреляции Мэтьюса (sklearn.metrics) = {matthews_corrcoef(y_test, y_pred):.2f}") # 0.9727766584990545
			f.write(f"matthews_corrcoef = {matthews_corrcoef(y_test, y_pred):.2f}\n")

			print("Classification Report for space:")
			print(classification_report(y_test, y_pred))
			f.write(f'{classification_report(y_test, y_pred)}\n')

			'''
			Ниже приведены некоторые общие рекомендации:

			* Высокая точность (accuracy) и показатель F1 указывают на то, 
			что модель в целом работает хорошо.

			* Высокая точность (precision) и низкая повторяемость (recall) 
			указывают на то, что модель консервативна в своих прогнозах и 
			упускает некоторые истинные положительные результаты.

			* Высокая повторяемость (recall) и низкая точность (precision) 
			указывают на то, что модель агрессивна в своих прогнозах, что 
			приводит к большему количеству ложных срабатываний.

			* Высокий показатель AUC-ROC указывает на то, что модель хорошо 
			различает положительные и отрицательные классы.

			* Низкие логарифмические потери и перекрестная энтропия указывают 
			на то, что модель уверена в своих прогнозах.
			'''

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


			# class_labels = np.unique(y) # Get unique class labels
			# plt.figure(figsize=(8, 6))
			all_fpr, all_tpr, _ = roc_curve(y_test, y_pred)
			# plt.plot(all_fpr, all_tpr, 'orange')
			# plt.xlabel(r'False Positive Rate ($FPR = \frac{FP}{FP + TN}$)')
			# '''
			# FPR представляет собой долю случаев, когда модель предсказала 
			# положительный исход, но фактический исход был отрицательным.
			# False positive rate является важным показателем при разработке 
			# и оценке моделей машинного обучения, особенно в ситуациях, где 
			# последствия ложного положительного предсказания могут быть серьёзными.

			# На высокую false positive rate в моделях машинного обучения могут 
			# влиять следующие факторы:
			# * Качество и баланс обучающих данных.
			# * Тип используемой модели.

			# Для снижения false positive rate важно тщательно выбирать и предварительно 
			# обрабатывать обучающие данные, выбирать подходящую модель для конкретной 
			# задачи и изменять порог модели для прогнозирования благоприятного исхода.
			# '''
			# plt.ylabel(r'True Positive Rate ($TPR = \frac{TP}{TP + FN}$)')
			# '''
			# TPR представляет собой пропорцию реальных положительных случаев, 
			# которые были правильно идентифицированы или классифицированы 
			# моделью как положительные. TPR также известен как чувствительность, 
			# отзывчивость или частота попадания.
			# '''

			roc_auc = auc(all_fpr, all_tpr)
			# plt.title(f'ROC curves for Spase-or-Nospace CatBoost-classification (area = {roc_auc:.2f})')
			# plt.grid()
			# plt.xlim(0, 1)
			# plt.ylim(0, 1.05)
			# # plt.show()
			# plt.savefig('ROC-spase_luck_ID.png')
			# # https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/
			print(f'ROC-AUC area = {roc_auc:.2f}') # = 0.99
			f.write(f'ROC-AUC area = {roc_auc:.4f}\n')

			print(f'accuracy = {accuracy*100:.2f}%, balanced_accuracy = {balanced_accuracy_score(y_test, y_pred)*100:.2f}%, f1 = {f1_score(y_test, y_pred,average='binary')*100:.2f}%\n\n')
			f.write(f'accuracy = {accuracy*100:.2f}%, balanced_accuracy = {balanced_accuracy_score(y_test, y_pred)*100:.2f}%, f1 = {f1_score(y_test, y_pred,average='binary')*100:.2f}%\n\n')

			'''
			Есть некоторые области, где использование ROC-AUC может быть не идеальным. 
			В случаях, когда набор данных сильно несбалансирован,  кривая ROC может дать 
			слишком оптимистичную оценку производительности модели . Эта предвзятость 
			оптимизма возникает из-за того, что уровень ложноположительных результатов 
			(FPR) кривой ROC может стать очень маленьким, когда количество фактических 
			отрицательных результатов велико.
			'''