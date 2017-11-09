from __future__ import print_function
from sklearn import datasets
import numpy as np
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

#diabetes dataset cosits of 10 pysiological variables on 442 patients
#and an indication of disease prograssion after 1 year
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_y_train = diabetes.target[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_test = diabetes.target[-20:]

from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr)
print(regr.coef_)

#the mean square error
print(np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2))
#1 is perfect score, 0 means no linear relationship
print(regr.score(diabetes_X_test, diabetes_y_test))

#shrinkage, if there are a few datapoints per dimension, noise in the observations case high variance
X = np.c_[.5, 1].T
y = [.5, 1]
test = np.c_[0,2].T
regr = linear_model.LinearRegression()

import matplotlib.pyplot as plt
plt.figure()

np.random.seed(0)
for _ in range(6):
    this_X = .1*np.random.normal(size=(2,1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s=3)

plt.show()

#a solution in high dimension stat learning is to shrink the regression coefficients to zero
#any two randomly chosen set of observations are likely to be uncorrelated, this is ridge regression

regr = linear_model.Ridge(alpha=.1)
plt.figure()
np.random.seed(0)
for _ in range(6):
    this_X = .1*np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s=3)
plt.show()
#the larger the ridge alpha parameter the higher the boas and lower the variance

#we can choose alpha to minimize left out the rror, this time using diabetes dataset rather than synthtic data

alphas = np.logspace(-4,-1,6)

print([regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train,).score(diabetes_X_test, diabetes_y_test) for alpha in alphas])

#pick up on sparcity
# http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
