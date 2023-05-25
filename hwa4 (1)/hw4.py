import math

import numpy as np

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
        X, y = splited_data[i][:, :-1], splited_data[i][:, -1]
        algo.fit(X, y)
        accuracies.append(calc_accuracy(algo, X, y))

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

    return counter/X.shape[0]

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
        self.mus = [np.mean(data)] * self.k
        self.sigmas = [np.std(data)] * self.k
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
        p = np.zeros((len(data), self.k))

        for i, instance in enumerate(data):
            for j in range(self.k):
                r = self.weights[j] * norm_pdf(instance, self.mus[j], self.sigmas[j])
                p[i, j] = r

            sumP = np.sum(p[i], axis=0)
            self.responsibilities[i] = p[i] / sumP
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
        for j in range(self.k):
            self.weights[j] = np.average(self.responsibilities[:, j])
            self.mus[j] = (1 / (self.weights[j] * len(data))) * np.sum(self.responsibilities[:, j] * data)
            self.sigmas[j] = (1 / (self.weights[j] * len(data))) * np.sum(self.responsibilities[:, j] * (data - self.mus[j]) ** 2)
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

        for i in range(1, self.n_iter):
            self.init_params(data)
            self.expectation(data)
            self.maximization(data)
            cost = self.compute_cost(data)
            if self.costs[i - 1] - cost < self.eps:
                break
            self.costs.append(cost)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def compute_cost(self, data):
        sigma = 0
        for i in range(self.k):
            sigma = sigma + np.sum(-np.log(self.weights[i] * norm_pdf(data, self.mus[i], self.sigmas[i])))
        self.costs.append(sigma)

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

        self.X = X
        self.y = y
        self.X_zero = X[y == 0]
        self.X_one = X[y == 1]
        self.n_examples = X.shape[0]
        self.n_features = X.shape[1]
        self.zero_dist_params = []
        self.one_dist_params = []
        self.cls_to_dist_params = {0: self.zero_dist_params, 1: self.one_dist_params}

        for i in range(self.n_features):
          # zero
          em = EM(self.k)
          em.fit(self.X_zero[:, i])
          self.zero_dist_params.append(em.get_dist_params())

          # one
          em = EM(self.k)
          em.fit(self.X_one[:, i])
          self.one_dist_params.append(em.get_dist_params())
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_prior(self, cls):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return len(self.X[self.y == cls]) / len(self.X)

    def get_instance_likelihood(self, x, cls):
        likelihood = 1
        for i in range(self.n_features):
          feature_likelihood = 0
          for j in range(self.k):
            dist_params = self.cls_to_dist_params[cls][i]
            mu = dist_params[0][j]
            sigma = dist_params[1][j]
            w = dist_params[2][j]
            feature_likelihood += w * norm_pdf(x[i], mu, sigma)

          likelihood *= feature_likelihood
        return likelihood

    def get_instance_posterior(self, x, cls):
        return self.get_instance_likelihood(x, cls) * self.get_prior(cls)

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
        # preds = []
        # for x in X:
        #     pred = 0 if self.get_instance_posterior(x) > self.get_instance_posterior(x) else 1
        #     preds.append(pred)

        number_of_predictions = len(X)
        preds = np.zeros(number_of_predictions)
        for i, x in enumerate(X): # loop over all instances
            prediction = 0 if self.get_instance_posterior(x, 0) > self.get_instance_posterior(x, 1) else 1
            preds[i] = prediction

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

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
    pass
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
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }