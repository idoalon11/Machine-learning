###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X,axis=0))
    y = (y - np.mean(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y


def apply_bias_trick(X):
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


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    J = np.sum(((np.dot(X, theta) - y) ** 2)) / (2 * len(X))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    for i in range(num_iters):
        theta = theta - alpha * (np.dot(X.T, (np.dot(X, theta) - y))) / len(X)
        J = compute_cost(X, y, theta)
        J_history.append(J)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudo-inverse algorithm.                            #
    ###########################################################################
    pinv_x = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    pinv_theta = np.dot(pinv_x, y)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    theta = theta - alpha * (np.dot(X.T, (np.dot(X, theta) - y))) / len(X)
    J = compute_cost(X, y, theta)
    J_history.append(J)
    for i in range(1, num_iters):
        theta = theta - alpha * (np.dot(X.T, (np.dot(X, theta) - y))) / len(X)
        J = compute_cost(X, y, theta)
        if J_history[i-1] - J < 1e-8:
            break
        J_history.append(J)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    for alpha in alphas:
        np.random.seed(42)
        theta = np.random.random(size=2)
        theta, J_history = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    feature_dict = {}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    while len(selected_features) < 5:
        for feature_idx in range(X_train.shape[1]):
            if feature_idx not in selected_features:
                selected_features.append(feature_idx)
                selected_X_train = apply_bias_trick(X_train[:, selected_features])
                selected_X_val = apply_bias_trick(X_val[:, selected_features])

                np.random.seed(42)
                theta = np.random.random(size=len(selected_features) + 1)

                theta, _ = efficient_gradient_descent(selected_X_train, y_train, theta, best_alpha, iterations)
                feature_dict[feature_idx] = compute_cost(selected_X_val, y_val, theta)
                selected_features.remove(feature_idx)

        best_feature = min(feature_dict, key=feature_dict.get)
        selected_features.append(best_feature)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    for i, feature in enumerate(df.columns):
        for j in range(i, len(df.columns)):
            second_feature = df.columns[j]
            if feature != second_feature:
                column_feature_name = feature + '*' + second_feature
            else:
                column_feature_name = feature + '^2'

            if column_feature_name not in df_poly.columns:
                new_values_feature = df[feature] * df[second_feature]
                new_column = pd.DataFrame({column_feature_name: new_values_feature})
                df_poly = pd.concat([df_poly] + [new_column], axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly
