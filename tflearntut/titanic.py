#
# Neural Network Classifier from http://tflearn.org/tutorials/quickstart.html
#

import numpy as np
import tflearn

# download the titanic set
from tflearn.datasets import titanic

titanic.download_dataset('titanic_dataset.csv')

# load csv file
from tflearn.data_utils import load_csv

data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)


# preprocessor function
def preprocess(data, columns_to_ignore):
    # sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        # converting 'sex' field to float(id is 1 after removing label columns
        data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)


# ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore = [1, 6]

# preprocess data
data = preprocess(data, to_ignore)

# build neural network
net = tflearn.input_data(shape=[None, 6])  # 6 features
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# define model
model = tflearn.DNN(net)
# start training (apply gradient descent)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# create data to test
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# predict survivibg chances
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])
