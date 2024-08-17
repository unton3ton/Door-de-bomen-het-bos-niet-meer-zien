# Sources

* [Save classifier to disk in scikit-learn](https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn)
* [Gaussian Naive Bayes using Sklearn](https://www.geeksforgeeks.org/gaussian-naive-bayes-using-sklearn/)
* [The naive forgive and forget; the wise forgive but do not forget](https://fritz.ai/naive-bayes-classifier-in-python-using-scikit-learn/)

  
  Now we can loop through each column with apply and sample with replacement from the non-missing values.

  
* [df.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))](https://stackoverflow.com/questions/46384934/pandas-replace-nan-using-random-sampling-of-column-values)


 
Select n numbers of rows randomly using sample(n) or sample(n=n). Each time you run this, you get n different rows. 


* [df.sample(n=3)](https://www.geeksforgeeks.org/how-to-randomly-select-rows-from-pandas-dataframe/) 
* [Моя шпаргалка по pandas](https://habr.com/ru/companies/ruvds/articles/494720/)
