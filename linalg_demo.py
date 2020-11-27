"""
References:
    Https://www.geeksforgeeks.org/linear-regression-python-implementation/
    https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/

"""

import numpy as np
import matplotlib.pyplot as plt

def plot_regression_line(x, y, b):
    """
    ripped from Https://www.geeksforgeeks.org/linear-regression-python-implementation/
    """
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m", marker = "o", s = 30)
    # predicted response vector
    y_pred = b[0] + b[1]*x 
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    # function to show plot
    plt.show()


def normal_equation(X, y):
    """
    from my brain but with help from https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/ 
    """
    # theta(min cost coeficients) = (Xt dot X)^-1 dot (Xt dot y)
    # lets do it in pieces
    # Xt is transpose of X, bassically flip and reverse of matrix
    Xt = np.transpose(X)
    # Xt dot X is the dot product (matrix mutiplication) of the two. remember that for this to work, they must be (m,n) * (n,m) dimentions and therefor Xt dot X != X dot Xt
    Xt_dot_X = np.dot(Xt, X)
    # we can get our other side, Xt dot y, while we are in the mood for dots
    Xt_dot_y = np.dot(Xt, X)
    # now for the inversion of Xt_dot_X, this is the most computationaly heavy step, im told its O(n^3), so over about n = 10,000 gradient decent, 
    # might be a better choice. n is the number of features
    try:
        inv_Xt_dot_X = np.linalg.inv(Xt_dot_X)
    except np.linalg.SigualarMatixError as e:
        print("a sigular matrix is not invertable, it will be div zero error")
        print("but numpy.linalg.pinv gives us a way to fake it out by replacing zero with really small floats")
        inv_Xt_dot_X = np.linalg.pinv(Xt_dot_X)
    # so theta is
    theta = np.dot(inv_Xt_dot_X, Xt_dot_y)
    print("theta calcutated to be: ".format(theta))
    return theta

def main():
    """     ripped from Https://www.geeksforgeeks.org/linear-regression-python-implementation/     """
    # observations
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    # estimating coefficients
    b = estimate_coef(x, y)   
    print("Estimated coefficients:\nb_0 = {}\nb_1 = {}".format(b[0], b[1])) 
    # plotting regression line   
    plot_regression_line(x, y, b) 
    
    
if __name__ == "__main__": 
    main()

