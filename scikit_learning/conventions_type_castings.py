import numpy as np
from sklearn import random_projection
from sklearn import datasets
from sklearn.svm import SVC
#unless otherwise stated input will be cast to float64

#setting dtype to float32
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
print(X.dtype)
#auto set to float64 with casting
transformer = random_projection.GaussianRandomProjection()
#here X is cast to float64
X_new = transformer.fit_transform(X)
print(X_new.dtype)


#regression targets are cast to float64, classification targets are maintained
iris = datasets.load_iris()
clf = SVC()
clf.fit(iris.data, iris.target)
print(clf)

#returns integer array because iris.target was used to fit
print(list(clf.predict(iris.data[:3])))

clf.fit(iris.data, iris.target_names[iris.target])
print(clf)
#returns string aray because target_names was used to fit
print(list(clf.predict(iris.data[:3])))





