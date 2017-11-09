import numpy as np
from sklearn.svm import SVC

#hyper parameters of an estimator can be updated after it has been constructed
#calling fit more than once wil overwrite what was learned by any previous fit

rng = np.random.RandomState(0)
X = rng.rand(100,10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

clf = SVC()
#default is rbf(radical basis function) but we change to linear
clf.set_params(kernel='linear').fit(X,y)
print(clf)

print(clf.predict(X_test))
#now we re enable rbf and re fit
clf.set_params(kernel='rbf').fit(X,y)
print(clf)

print(clf.predict(X_test))