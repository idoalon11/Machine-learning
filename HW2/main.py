import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, node):
        self.children.append(node)

# load dataset
data = pd.read_csv('agaricus-lepiota.csv')
data

data = data.dropna(axis=1)

from sklearn.model_selection import train_test_split
# Making sure the last column will hold the labels
X, y = data.drop('class', axis=1), data['class']
X = np.column_stack([X,y])
# split dataset using random_state to get the same split each time
X_train, X_test = train_test_split(X, random_state=99)

print("Training dataset shape: ", X_train.shape)
print("Testing dataset shape: ", X_test.shape)

from hw2 import calc_gini, calc_entropy

##### Your tests here #####

print(calc_gini(X), calc_entropy(X))

from hw2 import goodness_of_split

##### Your tests here #####

# python support passing a function as arguments to another function.

goodness_gini, split_values_gini = goodness_of_split(X, 0, calc_gini)
goodness_entropy, split_values_entropy = goodness_of_split(X, 0, calc_entropy)

print(goodness_gini, goodness_entropy)

from hw2 import build_tree

##### Your tests here #####

tree_gini = build_tree(data=X_train, impurity=calc_gini) # gini and goodness of split
tree_entropy = build_tree(data=X_train, impurity=calc_entropy) # entropy and goodness of split
tree_entropy_gain_ratio = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True) # entropy and gain ratio


from hw2 import calc_accuracy, predict

##### Your tests here #####

print('gini', calc_accuracy(tree_gini, X_train), calc_accuracy(tree_gini, X_test))
print('entropy', calc_accuracy(tree_entropy, X_train), calc_accuracy(tree_entropy, X_test))
print('entropy gain ratio', calc_accuracy(tree_entropy_gain_ratio, X_train),
      calc_accuracy(tree_entropy_gain_ratio, X_test))


def print_tree(node, depth=0, parent_feature='ROOT', feature_val='ROOT'):
    '''
    prints the tree according to the example above

    Input:
    - node: a node in the decision tree

    This function has no return value
    '''
    if node.terminal == False:
        if node.depth == 0:
            print('[ROOT, feature=X{}]'.format(node.feature))
        else:
            print('{}[X{}={}, feature=X{}], Depth: {}'.format(depth*'  ', parent_feature, feature_val,
                                                              node.feature, node.depth))
        for i, child in enumerate(node.children):
            print_tree(child, depth+1, node.feature, node.children_values[i])
    else:
        classes_count = {}
        labels, counts = np.unique(node.data[:, -1], return_counts=True)
        for l, c in zip(labels, counts):
            classes_count[l] = c
        print('{}[X{}={}, leaf]: [{}], Depth: {}'.format(depth*'  ', parent_feature, feature_val,
                                                         classes_count, node.depth))

print_tree(tree_gini)

