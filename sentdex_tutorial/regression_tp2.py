import math
import pandas as pd
import quandl
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import warnings


df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#some important data
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
#new df with important features
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# print(df.head())

# What we want to predict
forecast_col = 'Adj. Close'
# fills empty data
df.fillna(-99999, inplace=True)

# gets to ceiling of days out, predict 10% of data frame
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
# print(df.head())
#features are everything except label
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
#scal x before feeding through classifier
X = preprocessing.scale(X)

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test,y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#video 4

