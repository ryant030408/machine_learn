#supervised learning consists of learning the link between two datasets
#the observed data x and the external variable y that we are trying ot predict
#usually y is called a target or labels. most often y is a 1d array
#of length n_samples

import numpy as np
from sklearn import datasets

#if the prediction is a label then we are doing a classification
#if the prediction will be a continous target variable the we are doing regression


#this dataset consits of three types of irises, Setosa, Versicolor and Viginica
# it uses petal and sepal length and width
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
#the three type of iris
print(np.unique(iris_y))

#split iris data into train and test data
#a random permutation to split the data randomly

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
#create and fit a nearest neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
print(knn)
#print estimated answers then print actual answers
print(knn.predict(iris_X_test))
print(iris_y_test)

