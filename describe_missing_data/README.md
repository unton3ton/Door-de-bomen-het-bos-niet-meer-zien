from sklearn.impute import KNNImputer # замена ближайщими соседями
 
imputer = KNNImputer(n_neighbors=1) # дольше работает при той же точности
 
dataframe = pd.DataFrame(imputer.fit_transform(dataframe),columns = dataframe.columns)

# Sources 
* https://www.kaggle.com/code/anupamshah/how-to-handle-missing-categorical-features
* https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
