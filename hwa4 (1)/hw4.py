import math

import numpy as np

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# make matplotlib figures appear inline in the notebook
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = self.apply_bias_trick(X)
        theta = np.random.random(X.shape[1])
        self.theta, _ = self.efficient_gradient_descent(X, y, theta, self.eta, self.n_iter)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


    def apply_bias_trick(self, X):
        """
        Applies the bias trick to the input data.

        Input:
        - X: Input data (m instances over n features).

        Returns:
        - X: Input data with an additional column of ones in the
            zeroth position (m instances over n+1 features).
        """
        ###########################################################################
        # TODO: Implement the bias trick by adding a column of ones to the data. #
        ###########################################################################
        new_ones = np.ones((len(X), 1))
        X = np.column_stack((new_ones, X))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return X

    def compute_cost(self, X, y, theta):
        h = self.sigmoid(np.dot(X, theta))
        J = np.sum(np.dot(-y, np.log(h)) - np.dot(1 - y, np.log(1 - h))) / len(X)
        return J

    def efficient_gradient_descent(self, X, y, theta, alpha, num_iters):
        self.Js = []
        J = -1

        for i in range(num_iters):
            hypothesis = self.sigmoid(np.dot(X, theta))
            theta = theta - alpha * np.dot(X.T, hypothesis - y)

            J_previous = J
            J = self.compute_cost(X, y, theta)
            self.Js.append(J)

            if np.abs(J - J_previous) < self.eps:
                break

        return theta, self.Js

    def sigmoid(self, X):
        sigmoid = 1.0 / (1 + np.exp(-X))
        return sigmoid

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = self.apply_bias_trick(X)
        preds = []
        for instance in X:
            h = 1 / (1 + np.exp(np.dot(-self.theta.T, instance)))
            preds.append(1 if h > 0.5 else 0)

        preds = np.array(preds)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    accuracies = []
    merged_data = np.column_stack((X, y))
    np.random.shuffle(merged_data)
    splited_data = np.split(merged_data, folds, axis=0)
    for i in range(folds):
        newX, newy = splited_data[i][:, :-1], splited_data[i][:, -1]
        algo.fit(newX, newy)
        accuracies.append(calc_accuracy(algo, newX, newy))

    cv_accuracy = np.average(accuracies)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def calc_accuracy(algo, X, y):
    counter = 0
    preds = algo.predict(X)
    for i in range(len(preds)):
        if preds[i] == y[i]:
            counter = counter + 1

    return counter/len(preds)

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = (1 / np.sqrt(2 * math.pi * (sigma ** 2))) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.mus = np.random.random(self.k)
        self.sigmas = np.random.random(self.k)
        self.weights = [1 / self.k] * self.k
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.responsibilities = np.zeros((len(data), self.k))

        for i, instance in enumerate(data):
            for j in range(self.k):
                self.responsibilities[i, j] = self.weights[j] * norm_pdf(instance, self.mus[j], self.sigmas[j])

            self.responsibilities[i] = self.responsibilities[i] / np.sum(self.responsibilities[i], axis=0)
            ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.weights = np.average(self.responsibilities, axis=0)

        for j in range(self.k):
            self.mus[j] = (1 / (self.weights[j] * len(data))) * np.sum(np.dot(self.responsibilities[:, j], data))
            self.sigmas[j] = np.sqrt((1 / (self.weights[j] * len(data))) * np.sum(np.dot(self.responsibilities[:, j], np.square(data - self.mus[j]))))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.costs = []
        cost = -1
        self.init_params(data)

        for i in range(1, self.n_iter):
            self.expectation(data)
            self.maximization(data)

            cost_previous = cost
            cost = self.compute_cost(data)
            self.costs.append(cost)

            if np.abs(cost - cost_previous) < self.eps:
                break

            self.costs.append(cost)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def compute_cost(self, data):
        sigma = 0
        for x in data:
            likelihood = [w * norm_pdf(x, mu, sigma) for w, mu, sigma in zip(self.weights, self.mus, self.sigmas)]
            sigma += -np.log(sum(likelihood))
        return sigma

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pdf = 0
    for i in range(len(weights)):
        pdf = pdf + np.sum(weights[i] * norm_pdf(data, mus[i], sigmas[i]))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

        # additional attributes
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.classes = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.classes, counts = np.unique(y, return_counts=True)
        # self.weights = np.ones(self.k) / self.k
        # self.mus = np.random.rand(self.k)
        # self.sigmas = np.random.rand(self.k)
        #
        # for i in range(len(self.classes)):
        #     selected_data = X[np.where(y == self.classes[i])]
        #     for j in range(X.shape[1]):
        #         em = EM()
        #         em.fit(selected_data[:, j])
        #         self.weights, self.mus, self.sigmas = em.get_dist_params()
        self.prior = self.get_prior(y)

        num_of_classes = len(self.classes)
        num_of_features = X.shape[1]

        self.weights = np.empty([num_of_classes, num_of_features, self.k])
        self.mus = np.empty([num_of_classes, num_of_features, self.k])
        self.sigmas = np.empty([num_of_classes, num_of_features, self.k])

        for i in range(num_of_classes):
            data_by_class = X[np.where(y == self.classes[i])]
            for j in range(X.shape[1]):
                feature_by_class = data_by_class[:, j]
                class_EM = EM(self.k, random_state=self.random_state)
                class_EM.fit(feature_by_class)
                self.weights[i][j], self.mus[i][j], self.sigmas[i][j] = class_EM.get_dist_params()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        num_classes = len(self.classes)
        preds = np.empty([num_classes, len(X)])

        for i in range(num_classes):
            prior = self.prior[i]
            likelihood = self.get_likelihood(X, i)
            preds[i] = prior * likelihood

        preds = preds[0] < preds[1]
        preds = preds.astype(int)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

    def get_prior(self, y):
        _, counts = np.unique(y, return_counts=True)
        prior_sol = (counts / len(y))
        return prior_sol

    def get_likelihood(self, data, class_value):
        likelihood = 1
        iterations = self.mus.shape[1]
        for i in range(iterations):
            prob = 0
            for j in range(self.k):
                norm = norm_pdf(data[:, i], self.mus[class_value][i][j], self.sigmas[class_value][i][j])
                prob = prob + self.weights[class_value][i][j] * norm
            likelihood = likelihood * prob
            # norm = gmm_pdf(data, self.weights[class_value][i], self.mus[class_value][i], self.sigmas[class_value][i])
            # likelihood = likelihood * norm
        return likelihood

def get_accuracy(pred, data):
    accuracy = pred == data
    accuracy = accuracy.astype(int)
    return np.sum(accuracy) / accuracy.shape[0]

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    logistic_regression = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_regression.fit(x_train, y_train)

    logistic_train_predict = logistic_regression.predict(x_train)
    logistic_test_predict = logistic_regression.predict(x_test)

    lor_train_acc = get_accuracy(logistic_train_predict, y_train)
    lor_test_acc = get_accuracy(logistic_test_predict, y_test)

    naive_bayes = NaiveBayesGaussian(k)
    naive_bayes.fit(x_train, y_train)

    naive_train_predict = naive_bayes.predict(x_train)
    naive_test_predict = naive_bayes.predict(x_test)

    bayes_train_acc = get_accuracy(naive_train_predict, y_train)
    bayes_test_acc = get_accuracy(naive_test_predict, y_test)

    title_lor = "Logistic regression " + str(len(x_train))

    plot_decision_regions(x_train, y_train, classifier=logistic_regression, title=title_lor)
    logistic_regression = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_regression.fit(x_train, y_train)
    plt.plot(range(len(logistic_regression.Js)), logistic_regression.Js)
    plt.title("Logistic regression - first 1000 points - cost as function of iterations")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    title_naive = "Naive bayes " + str(len(x_train))
    plot_decision_regions(x_train, y_train, classifier=naive_bayes, title=title_naive)

    # plot_decision_regions(X_training, y_training, classifier=lor, title="Logistic regression - Full Data")
    #
    # # plot_decision_regions(X_training, y_training, classifier=naive_bayes, title="Naive bayes - Full Data")
    #
    # logistic_regression = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    # logistic_regression.fit(X_training, y_training)
    # plt.plot(range(len(logistic_regression.Js)), logistic_regression.Js)
    # plt.title("Logistic regression - all data - cost as function of iterations")
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.show()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    samples = np.array([500, 500, 500, 500])
    mu = [[20, -20, 0], [-20, 20, 0], [20, 20, 0], [-20, -20, 0]]
    sigma = [[[16, 0, 0],
              [0, 16, 0],
              [0, 0, 16]],

             [[16, 0, 0],
              [0, 16, 0],
              [0, 0, 16]],

             [[16, 0, 0],
              [0, 16, 0],
              [0, 0, 16]],

             [[16, 0, 0],
              [0, 16, 0],
              [0, 0, 16]]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ["b", "b", "y", "y"]

    y0 = np.zeros((500, 1))
    y1 = np.ones((500, 1))
    dataset_a_labels = [y0, y0, y1, y1]

    for i in range(len(samples)):
        generated_data = np.random.multivariate_normal(mu[i], sigma[i], samples[i])
        x = generated_data[:, 0]
        y = generated_data[:, 1]
        z = generated_data[:, 2]
        x1 = np.reshape(x, (x.shape[0], 1))
        x2 = np.reshape(y, (x.shape[0], 1))
        x3 = np.reshape(z, (x.shape[0], 1))

        dataset_a_features = [x1, x2, x3]

        ax.scatter(x, y, z, color=colors[i])
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('feature 1')
        ax.set_ylabel('feature 2')
        ax.set_zlabel('feature 3')

    plt.show()

    y0 = np.zeros((500, 1))
    x_feature_class0 = np.random.normal(20, 1, 500)
    y_feature_class0 = x_feature_class0 * 2 + np.random.normal(20, 20, 500)
    z_feature_class0 = y_feature_class0 * 4 + np.random.normal(10, 3, 500)
    x_feature_class0 = np.reshape(x_feature_class0, (x_feature_class0.shape[0], 1))
    y_feature_class0 = np.reshape(y_feature_class0, (y_feature_class0.shape[0], 1))
    z_feature_class0 = np.reshape(z_feature_class0, (z_feature_class0.shape[0], 1))

    y1 = np.ones((500, 1))
    x_feature_class1 = np.random.normal(10, 1, 500)
    y_feature_class1 = x_feature_class1 * 2 + np.random.normal(20, 20, 500) + 20
    z_feature_class1 = y_feature_class1 * 4 + np.random.normal(10, 3, 500) + 40
    x_feature_class1 = np.reshape(x_feature_class1, (x_feature_class1.shape[0], 1))
    y_feature_class1 = np.reshape(y_feature_class1, (y_feature_class1.shape[0], 1))
    z_feature_class1 = np.reshape(z_feature_class1, (z_feature_class1.shape[0], 1))

    dataset_b_labels = [y0, y1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_feature_class0, y_feature_class0, z_feature_class0, c="b")
    ax.scatter(x_feature_class1, y_feature_class1, z_feature_class1, c="y")
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')
    ax.set_zlabel('feature 3')
    plt.show()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }