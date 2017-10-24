#!env python2

# Bibtex dataset -- convert arff format to pkl (please run with python2)

import arff  # pip install liac-arff
import os
import pickle

data_dir = '../data'

ftrain = os.path.join(data_dir, 'bibtex/bibtex-train.arff')
ftest = os.path.join(data_dir, 'bibtex/bibtex-test.arff')

pkltrain = os.path.join(data_dir, 'bibtex/bibtex-train.pkl')
pkltest = os.path.join(data_dir, 'bibtex/bibtex-test.pkl')

data_train = arff.load(open(ftrain, 'rb'))  # works in python2 but not in python3
pickle.dump(data_train, open(pkltrain, 'wb'))

data_test = arff.load(open(ftest, 'rb'))
pickle.dump(data_test, open(pkltest, 'wb'))

#from scipy.io import arff
#data = arff.loadarff('data/bibtex/bibtex-test.arff')  # errors

ftrain = os.path.join(data_dir, 'delicious/delicious-train.arff')
ftest = os.path.join(data_dir, 'delicious/delicious-test.arff')

pkltrain = os.path.join(data_dir, 'delicious/delicious-train.pkl')
pkltest = os.path.join(data_dir, 'delicious/delicious-test.pkl')

data_train = arff.load(open(ftrain, 'rb'))  # works in python2 but not in python3
pickle.dump(data_train, open(pkltrain, 'wb'))

data_test = arff.load(open(ftest, 'rb'))
pickle.dump(data_test, open(pkltest, 'wb'))
