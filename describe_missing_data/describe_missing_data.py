# python describe_missing_data.py -s space-old.csv -n noSpace-old.csv
# python describe_missing_data.py -s space-october.csv -n noSpace-october.csv

# https://www.dmitrymakarov.ru/data-analysis/nan-06/

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse

import missingno as msno # pip install missingno   ;   https://pypi.org/project/missingno/
sns.set()

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
	map(pd.read_csv, [space_csv, no_space_csv]), ignore_index=True)  
	# map(pd.read_csv, [space_csv, 'notspace.csv']), ignore_index=True) 

print(dataframe.head(), '\n')
print(dataframe.shape, '\n')
print(dataframe.info(), '\n')
print(dataframe.describe(), '\n')
print(dataframe['isSpace'].value_counts())

'''
skew() в Python pandas — это метод для вычисления коэффициента асимметрии распределения.
Он определяет, насколько распределение отличается от нормального (колоколообразная кривая), 
где значение 0 указывает на идеальную симметрию. 

Метод вычисляет коэффициент асимметрии для каждой строки или каждого столбца данных, представленных в объекте DataFrame.

Асимметрией называется смещение распределения относительно ее моды. 
Отрицательная асимметрия, или левое смещение кривой, указывает на то, 
что площадь под графиком больше на левой стороне моды. 
Положительная асимметрия, или правое смещение кривой, указывает на то, 
что площадь под графиком больше на правой стороне моды.
'''
categorical_features_etc = ['fir',
						'aerodrome',
						'subject',
						'condition',
						'traffic',
						'purpose',
						'scope',
                        'codeF',
                        'codeG',
                        'id',
                        'idSpaceport',
                        ]
# dataframe = dataframe.drop(categorical_features_etc, axis=1)

# dataframe = dataframe[(dataframe['isSpace'] == 1)]
# print(dataframe.head(), '\n')
# print(dataframe.describe(), '\n')
# print(dataframe['isSpace'].value_counts())

# print('Ассиметрия\n', dataframe.skew(axis = 0, skipna = True),'\n')
# print('Среднее\n', dataframe.mean(axis = 0, skipna = True),'\n')
# print('Median\n', dataframe.median(axis = 0, skipna = True),'\n')
# print('Mode\n', dataframe.dropna().mode(axis = 0),'\n')

# # .isna() выдает True или 1, если есть пропуск,
# # .sum() суммирует единицы по столбцам
print('isna().sum()\n', dataframe.isna().sum(), '\n')

# # Процент пропущенных значений
# # для этого разделим сумму пропусков в каждом столбце на количество наблюдений,
# # округлим результат и умножим его на 100
print(f'Процент пропущенных значений \n{(dataframe.isna().sum() / len(dataframe)).round(4) * 100} \n')


# #Еще один интересный инструмент — матрица корреляции пропущенных значений (nullity correlation matrix).
# #По сути она показывает, насколько сильно присутствие или отсутствие значений одного признака влияет на присутствие значений другого.

df = dataframe.iloc[:, [i for i, n in enumerate(np.var(dataframe.isnull(), axis = 'rows')) if n > 0]]
print(df.isnull().corr())

plt.figure(figsize=(13, 13))
sns.heatmap(df.isnull().corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Матрица корреляции пропущенных значений')
# plt.savefig('nullity_correlation_matrix.png')
plt.savefig('nullity_correlation_matrix_imba.png')
plt.close()

msno.bar(dataframe, figsize=(18,10), fontsize=12)
plt.title('Столбчатая диаграмма пропущенных значений', fontsize=15)
# plt.savefig('nullity_bar.png')
plt.savefig('nullity_bar_imba.png')
# plt.show()
plt.close()

sns.displot(
    data=dataframe.isnull().melt(value_name='missing'),
    y='variable',
    hue='missing',
    multiple='fill',
    height=8,
    aspect=1.1
)
# specifying a threshold value
# plt.axvline(0.34, color='r')
# plt.savefig('nullity_bar_sns.png')
plt.axvline(0.071, color='r')
plt.savefig('nullity_bar_sns_imba.png')
# plt.show()
plt.close()

msno.matrix(dataframe, figsize=(18,15))
plt.title('Mатрица пропущенных значений', fontsize=18)
# plt.savefig('nullity_matrix.png')
plt.savefig('nullity_matrix_imba.png')
# plt.show()
plt.close()

msno.heatmap(dataframe)
plt.title('Mатрица корреляции пропущенных значений', fontsize=20)
# plt.savefig('nullity_correlation_matrix_msno.png')
plt.savefig('nullity_correlation_matrix_msno_imba.png')
# plt.show()
plt.close()

msno.dendrogram(dataframe, figsize=(18,15))
plt.title('Дендрограмма пропущенных значений', fontsize=17)
# plt.savefig('nullity_dendrogram.png')
plt.savefig('nullity_dendrogram_imba.png')
# plt.show()
plt.close()


print(len(dataframe[(dataframe['fir'] == dataframe['aerodrome'])]), '\n')
print(f"fir = aerodrome: {len(dataframe[(dataframe['fir'] == dataframe['aerodrome'])])/dataframe.shape[0]*100} %\n")
# print(dataframe[(dataframe['fir'] == dataframe['aerodrome'])], '\n')

print(f"Количество уникальных значений в столбце 'aerodrome' = {dataframe['aerodrome'].nunique(dropna=True)}")
print(dataframe['aerodrome'].value_counts()) # "Мы все уникальны, такие же уникальные как все остальные."
value = dataframe.shape[0] - dataframe.isna().sum()['aerodrome']
print(f'value = {value}')
# вероятность "правильного" (fir = aerodrome) заполнения пропусков для каждого уникального значения в столбце 'aerodrome'
print(dataframe['aerodrome'].value_counts().div(value).mul(100))
# '''
# LRBB    10.667634 %
# UUWV     6.313498 %
# USTV     5.926463 %
# RJJJ     5.805515 %
# YMMM     5.152395 %
#           ...
# UHPP     0.024190 %
# LAAA     0.024190 %
# FACT     0.024190 %
# CZEG     0.024190 %
# CZQX     0.024190 %
# Name: count, Length: 139
# '''


sum_proc_fir_aero = 0
N = 10
for i in range(N):
	dataframe_current = dataframe.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))
	sum_proc_fir_aero += len(dataframe_current[(dataframe_current['fir'] == dataframe_current['aerodrome'])])/dataframe_current.shape[0]*100

# print(f'Процент пропущенных значений \n{(dataframe_current.isna().sum() / len(dataframe_current)).round(4) * 100} \n')
proc_fir_aero = sum_proc_fir_aero / N
print(f"fir = aerodrome: {proc_fir_aero} %\n")

