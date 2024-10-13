# python alles_klassik.py -n noSpace-october.csv -s space-october.csv
# python alles_klassik.py -n noSpace-all.csv -s space-all.csv

from catboost import CatBoostClassifier

from sklearn.ensemble import (AdaBoostClassifier,
						HistGradientBoostingClassifier,
						RandomForestClassifier,
						ExtraTreesClassifier,
						GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import (SGDClassifier,
						    RidgeClassifier)
from sklearn.svm import SVC
from sklearn.naive_bayes import (GaussianNB,
						   BernoulliNB)
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb # pip install lightgbm xgboost
import lightgbm as lgb

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import (StandardScaler, 
                                   OneHotEncoder)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
						precision_score,
						recall_score,
						f1_score,
						confusion_matrix,
						classification_report,
						matthews_corrcoef,
						roc_curve,
						auc,
						cohen_kappa_score,
                              balanced_accuracy_score,
                              f1_score)
import pandas as pd
import numpy as np
import argparse

import warnings
warnings.filterwarnings("ignore")

ID_test = 'id'

parser = argparse.ArgumentParser(description="Пример использования консольного интерфейса")
parser.add_argument("-n", "--nospace", help="Введите (путь и/или) имя nospace.csv", required=True)
parser.add_argument("-s", "--space", help="Введите (путь и/или) имя space.csv", required=True)
 
args = parser.parse_args()
 
no_space_csv = args.nospace # 'noSpace-v7.csv'
space_csv = args.space # 'space-v7.csv'

df = pd.read_csv(space_csv)
n_balance = df.shape[0] # 3176 - количество строк в space.csv

df = pd.read_csv(no_space_csv)
'''Выберите n строк случайным образом, используя sample(n) или sample(n=n). 
Каждый раз, когда вы выполняете это, вы получаете n разных строк.'''
df = df.sample(n=n_balance)
df.to_csv('notspace.csv', index=False)

# merging two csv files 
dataframe = pd.concat(
	# map(pd.read_csv, [space_csv, no_space_csv]), ignore_index=True)  
	map(pd.read_csv, [space_csv, 'notspace.csv']), ignore_index=True) 

# print(dataframe.head(), '\n')
# print(dataframe.shape, '\n')
# print(dataframe.info(), '\n')
# print(dataframe.describe(), '\n')

categorical_features = ['fir',
					'aerodrome',
					'subject',
					'condition',
					'traffic',
					'purpose',
					'scope',
                        'codeF',
                        'codeG']

'''
## OneHotEncoder : [5 rows x 991 columns]
#для imbalanceddataset - numpy.core._exceptions._ArrayMemoryError: Unable to allocate 4.06 GiB for an array with shape (138178, 3942) and data type float64
categorical_features = dataframe.select_dtypes(include=['object']).columns.tolist()
#Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(dataframe[categorical_features])
#Create a DataFrame with the one-hot encoded columns
#We use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_features))
# Concatenate the one-hot encoded dataframe with the original dataframe
dataframe = pd.concat([dataframe, one_hot_df], axis=1)
'''
dataframe = dataframe.drop(categorical_features, axis=1)

# pd.set_option('display.max_columns', None)
print(dataframe.head(), '\n')

flag_id = False
if ID_test in dataframe.columns:
	flag_id = True
	ID = dataframe[ID_test]
	dataframe = dataframe.drop(ID_test, axis=1)

target = 'isSpace'

dataframe1 = dataframe.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x)) # замена случайными из имеющихся

from sklearn.impute import KNNImputer # замена ближайщими соседями
imputer = KNNImputer(n_neighbors=3) # дольше работает при той же точности
dataframe2 = pd.DataFrame(imputer.fit_transform(dataframe),columns = dataframe.columns)

dataframe3 = dataframe.fillna(dataframe.mean()) ## fill NaNs with column means in each column # значения NaN в каждом столбце были заполнены средним значением столбца

#define the model
clf0 = CatBoostClassifier(iterations=500,  # Number of boosting iterations
                           learning_rate=0.1,  # Learning rate
                           depth=8,  # =8 Depth of the tree - Глубина деревьев. Регулирует сложность модели. Глубокие деревья могут лучше выявлять сложные зависимости, но также рискуют переобучиться.
                           verbose=100,  # Print training progress every 50 iterations
                           early_stopping_rounds=10,  # stops training if no improvement in 10 consequtive rounds
                           loss_function='Logloss',
                           custom_metric=['Accuracy', 'AUC'],
                           use_best_model=False, 
                           random_seed=42,
                           auto_class_weights='Balanced') 

clf1 = CatBoostClassifier(auto_class_weights='Balanced')


clf2 = AdaBoostClassifier(n_estimators=300,
                         learning_rate=1.4, # требуется "ручной" подбор скорости обучения для максимизации точности
                         random_state=42) # 

clf3 = HistGradientBoostingClassifier(max_iter=100) 

clf4 = RandomForestClassifier(random_state=0, n_jobs=-1) 

clf5 = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0) 

clf6 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0) 

clf7 = DecisionTreeClassifier(random_state=0) 

clf8 = SGDClassifier(class_weight='balanced') 

clf9 = RidgeClassifier(class_weight='balanced') 

clf10 = SVC(class_weight='balanced') 

clf11 = GaussianNB() 

clf12 = BernoulliNB() 

clf13 = CalibratedClassifierCV() 

clf14 = xgb.XGBClassifier() 

clf15 = lgb.LGBMClassifier(metric='auc') 

clf16 = MLPClassifier(random_state=1, max_iter=300) 

clf17 = KNeighborsClassifier(n_neighbors=3) 

clfs = [clf0,clf1,clf2,clf3,clf4,clf5,clf6,clf7,
	  clf8,clf9,clf10,clf11,clf12,clf13,clf14,
	  clf15,clf16,clf17]

# filename = 'results-balanced-random.txt'
# filename = 'results-balanced-neighbors.txt'
# filename = 'results-balanced-mean.txt'

# filename = 'results-balanced-random-OneHotEncoder.txt';features = dataframe1.drop(target, axis=1);target = dataframe1[target]
# filename = 'results-balanced-neighbors-OneHotEncoder.txt';features = dataframe2.drop(target, axis=1);target = dataframe2[target]
# filename = 'results-balanced-mean-OneHotEncoder.txt';features = dataframe3.drop(target, axis=1);target = dataframe3[target]

# filename = 'results-imbalanced-random.txt';features = dataframe1.drop(target, axis=1);target = dataframe1[target]
# filename = 'results-imbalanced-neighbors.txt';features = dataframe2.drop(target, axis=1);target = dataframe2[target]
filename = 'results-imbalanced-mean.txt';features = dataframe3.drop(target, axis=1);target = dataframe3[target]

# import os
# try:
# 	os.remove(filename)
# except OSError:
# 	pass
# with open(filename, 'a') as f:

with open(filename, 'w') as f:

	# features = dataframe1.drop(target, axis=1)
	# target = dataframe1[target]

	# features = dataframe2.drop(target, axis=1)
	# target = dataframe2[target]

	# features = dataframe3.drop(target, axis=1)
	# target = dataframe3[target]

	# Standardize features
	scaler = StandardScaler()
	features_standardized = scaler.fit_transform(features)

	X_train, X_test, y_train, y_test = train_test_split(features_standardized, target, test_size=0.2, random_state=42)

	# for clf in clfs:
	for i, clf in enumerate(clfs):
		
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		# print(y_pred)

		accuracy = accuracy_score(y_test, y_pred)
		# print("Accuracy:", accuracy)

		# confusion = confusion_matrix(y_test, y_pred)
		# print(confusion)
		# f.write(confusion,'\n')

		Kappa = cohen_kappa_score(y_test, y_pred)
		print(f'\nKappa = {Kappa:.2f}\n')
		f.write(f'Model number: {i}\n')
		f.write(f'Kappa = {Kappa:.2f}\n')

		print(f"matthews_corrcoef = {matthews_corrcoef(y_test, y_pred):.2f}")
		f.write(f"matthews_corrcoef = {matthews_corrcoef(y_test, y_pred):.2f}")

		print(classification_report(y_test, y_pred),'\n')
		f.write(f'{classification_report(y_test, y_pred)}\n')

		all_fpr, all_tpr, _ = roc_curve(y_test, y_pred)
		roc_auc = auc(all_fpr, all_tpr)
		print(f'ROC-AUC area = {roc_auc:.2f}\n')
		f.write(f'ROC-AUC area = {roc_auc:.2f}\n')

		# print(f'accuracy = {accuracy}, balanced_accuracy = {balanced_accuracy_score(y_test, y_pred)}')
		print(f'accuracy = {accuracy*100:.2f}%, \
			balanced_accuracy = {balanced_accuracy_score(y_test, y_pred)*100:.2f}%, \
			f1 = {f1_score(y_test, y_pred,average='binary')*100:.2f}%\n')
		f.write(f'\naccuracy = {accuracy*100:.2f}%, \
			balanced_accuracy = {balanced_accuracy_score(y_test, y_pred)*100:.2f}%, \
			f1 = {f1_score(y_test, y_pred,average='binary')*100:.2f}%\n\n' + '\n')

