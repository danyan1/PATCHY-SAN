import numpy as np
import data
import util
import networkx as nx
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from receptive_field import ReceptiveField
from pscn import PSCN

test_size = 0.2

## MUTAG Dataset
dataset = data.load_compact_data('MUTAG')
X, y = zip(*dataset)

# Set $w$ to be average number of nodes in input graphs
w = util.compute_average_node_count(X)
num_classes = 2
num_attr = 1

# Initialize model
pscn = PSCN(w, num_attr=num_attr, num_classes=num_classes, attribute_name='node_label', epochs=10)
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
# Model training
pscn.fit(X_train, y_train)
#### Model Evaluation
pscn.evaluate(X_test, y_test)


## COX2 Dataset
dataset = data.load_compact_data('COX2')
X, y = zip(*dataset)

w = util.compute_average_node_count(X)
num_classes = 2
num_attr = 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=13)
pscn = PSCN(w, num_attr=num_attr, num_classes=num_classes, attribute_name='node_attributes', epochs=10)
pscn.fit(X_train, y_train)
pscn.evaluate(X_test, y_test)


## AIDS Dataset
dataset = data.load_compact_data('AIDS')
X, y = zip(*dataset)

w = util.compute_average_node_count(X)
num_classes = 2
num_attr = 4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
pscn = PSCN(w, num_attr=num_attr, num_classes=num_classes, attribute_name='node_attributes', epochs=10)
pscn.fit(X_train, y_train)
pscn.evaluate(X_test, y_test)


## Fingerprint Dataset
dataset = data.load_compact_data('Fingerprint')
X, y = zip(*dataset)

w = util.compute_average_node_count(X)
num_classes = 15
num_attr = 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=23)
#### Convert labels to categorical array
y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
pscn = PSCN(w, num_attr=num_attr, num_classes=num_classes, attribute_name='node_attributes', epochs=10)
pscn.fit(X_train, y_train)
pscn.evaluate(X_test, y_test)


## Letter Dataset
dataset = data.load_compact_data('Letter-high')
X, y = zip(*dataset)

w = util.compute_average_node_count(X)
num_classes = 15
num_attr = 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
pscn = PSCN(w, num_attr=num_attr, num_classes=num_classes, attribute_name='node_attributes', epochs=10)
pscn.fit(X_train, y_train)
pscn.evaluate(X_test, y_test)


## SYNTHETIC Dataset
dataset = data.load_compact_data('SYNTHETIC')
X, y = zip(*dataset)

w = util.compute_average_node_count(X)
num_classes = 2
num_attr = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
pscn = PSCN(w, num_attr=num_attr, num_classes=num_classes, attribute_name='node_attributes', epochs=10)
pscn.fit(X_train, y_train)
pscn.evaluate(X_test, y_test)
