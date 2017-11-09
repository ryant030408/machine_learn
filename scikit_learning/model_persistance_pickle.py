from sklearn import svm
from sklearn import datasets

#define classifier
clf = svm.SVC()
#open dataset and save data to X and target to y
iris = datasets.load_iris()
X, y = iris.data, iris.target
#fit dataset
clf.fit(X,y)
print(clf)

#this saves to a string
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[0:1]))
print(y[0])

#but scikit has its own persistance maintainer
from sklearn.externals import joblib
#save with this
joblib.dump(clf, 'test_dump.pkl')
#load with this
clf = joblib.load('test_dump.pkl')