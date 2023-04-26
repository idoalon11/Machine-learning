import queue

import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = ['e', 'p']
    s = len(data)
    sigma = 0

    # we did loop to make the code generic and suitable to any number of labels
    for c in labels:
        sigma = sigma + (np.count_nonzero(data[:, -1] == c)/s) ** 2

    gini = 1 - sigma
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = ['e', 'p']
    s = len(data)
    sigma = 0

    # we did loop to make the code generic and suitable to any number of labels
    for c in labels:
        s_i = np.count_nonzero(data[:, -1] == c)
        if not s_i == 0:
            sigma = sigma + ((s_i / s) * np.log2(s_i / s))
    entropy = -sigma
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {}  # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    sigma = 0
    s = len(data)
    splitInformation = 1

    if gain_ratio:
        for value in np.unique(data[:, feature]):
            s_a = np.count_nonzero(data[:, feature] == value)
            if not s_a == 0:
                sigma = sigma + ((s_a / s) * np.log2(s_a / s))
            splitInformation = - sigma

    sigma = 0

    for value in np.unique(data[:, feature]):
        groups[value] = data[data[:, feature] == value]

    for key, value in groups.items():
        sigma = sigma + (len(value) / len(data)) * impurity_func(value)

    if not splitInformation == 0:
        goodness = (impurity_func(data) - sigma) / splitInformation
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups


class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        unique, counts = np.unique(self.data[:, -1], return_counts=True)
        if len(unique) == 1:
            pred = unique[0]
        else:
            max_index = np.argmax(counts)
            pred = unique[max_index]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        feature_dict = {}
        for i in range(self.data.shape[1] - 1):
            feature_dict[i] = goodness_of_split(self.data, i, impurity_func, self.gain_ratio)

        self.feature = max(feature_dict, key=lambda x: feature_dict[x][0])

        if feature_dict[self.feature][0] == 0 or self.max_depth == self.depth:
            self.terminal = True
            return

        if not self.chi == 1:
            unique, counts = np.unique(self.data[:, -1], return_counts=True)
            p_0 = counts[0] / np.sum(counts)
            p_1 = counts[1] / np.sum(counts)

            unique1, counts1 = np.unique(self.data[:, self.feature], return_counts=True)
            list_of_values = list(zip(unique1.tolist(), counts1.tolist()))
            sigma = 0
            for value in list_of_values:
                d_f = value[1]
                p_f = 0
                n_f = 0
                unique2, counts2 = np.unique(feature_dict[self.feature][1][value[0]][:, -1], return_counts=True)

                if not len(counts2) == 1:
                    p_f = counts2[0]
                    n_f = counts2[1]
                elif unique2[0] == 'e':
                    p_f = counts2[0]
                else:
                    n_f = counts2[0]

                e_0 = d_f * p_0
                e_1 = d_f * p_1
                sigma = sigma + (((p_f - e_0) ** 2) / e_0) + (((n_f - e_1) ** 2) / e_1)

            if sigma <= chi_table[len(list_of_values) - 1][self.chi]:
                self.terminal = True
                return

        # create the corresponding children
        for key, value in feature_dict[self.feature][1].items():
            new_child = DecisionNode(value, depth=self.depth + 1, gain_ratio=self.gain_ratio, max_depth=self.max_depth, chi=self.chi)
            self.add_child(new_child, key)

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    nodes_queue = queue.Queue()
    root = DecisionNode(data, gain_ratio=gain_ratio, max_depth=max_depth, chi=chi)
    nodes_queue.put(root)

    while not nodes_queue.empty():
        n = nodes_queue.get()
        n.split(impurity)
        for child in n.children:
            nodes_queue.put(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root


def perfectlyClassified(n):
    pred = n.data[0, -1]
    return not np.count_nonzero(n.data[:, -1] != pred) > 0


def predict(root, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    while not root.terminal:
        value = instance[root.feature]
        if value in root.children_values:
            index_of_value = root.children_values.index(value)
            root = root.children[index_of_value]
        else:
            break

    pred = root.pred

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    counter = 0

    for row in dataset:
        if predict(node, row) == row[-1]:
            counter = counter + 1

    accuracy = counter/dataset.shape[0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy * 100

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output: the training and testing accuracies per max depth
    """
    training = []
    testing = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, max_depth=max_depth)
        training.append(calc_accuracy(tree, X_train))
        testing.append(calc_accuracy(tree, X_test))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    tree = None

    p_values_cut_off = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for p in p_values_cut_off:
        tree = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, chi=p)
        chi_training_acc.append(calc_accuracy(tree, X_train))
        chi_testing_acc.append(calc_accuracy(tree, X_test))
        # find the depth of the tree
        depth.append(find_tree_depth(tree))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth


def find_tree_depth(root):
    if root.terminal:
        return 0

    depth_arr = []
    for child in root.children:
        depth_arr.append(find_tree_depth(child))

    max_depth = np.max(depth_arr) + 1
    return max_depth


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    if node.terminal:
        return 0

    array = []

    for child in node.children:
        array.append(count_nodes(child))

    n_nodes = np.sum(array) + 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






