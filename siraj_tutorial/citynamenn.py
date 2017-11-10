from __future__ import absolute_import, division, print_function
import os
# LSTM(long short term memory network) to generate city names
import ssl

import sys
import tflearn
from six import moves
from tflearn.data_utils import *


# step 1 - retrieve data
path = "US_cities.txt"

if not os.path.isfile(path):
    print("NOPE")
    sys.exit()

# city name max length
maxlen = 20

# vector-ize text file, redundant step means we will go through 3 sequences to make it happen
X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

# create LSTM
# saying make our inout layer, 20 cahracters
g = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
# now create lstm later, 512 nodes(neurons)
g = tflearn.lstm(g, 512, return_seq=True)
# create dropout, when we train nn data is flowing, this drops over fitting(cant generalize data to new info) so
# nodes are randomly turned off, break out of 'groves'
g = tflearn.dropout(g, 0.5)
# create another lstm layer
g = tflearn.lstm(g, 512)
# add more dropout for every layer, 0.5 is just a measure of how random we want to get
g = tflearn.dropout(g, 0.5)
# last layer, softmax is a type of logistic regression
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

# generate cities
# checkpoints save model status
m = tflearn.SequenceGenerator(g, dictionary=char_idx, seq_maxlen=maxlen, clip_gradients=5.0,
                              checkpoint_path='model_us_cities')

#training
for i in range(40):
    # seed helps us start from the same time every time we generate
    seed = random_sequence_from_textfile(path, maxlen)
    # this is the training
    m.fit(X, Y, validation_set=0.1, batch_size=128, n_epoch=1, run_id='us_cities')
    print("TESTING")
    print(m.generate(30, temperature=1.2, seq_seed=seed))
    print("TESTING")
    print(m.generate(30, temperature=1.0, seq_seed=seed))
    print("TESTING")
    print(m.generate(30, temperature=0.5, seq_seed=seed))
