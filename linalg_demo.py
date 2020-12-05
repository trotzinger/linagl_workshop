"""
References:
    Https://www.geeksforgeeks.org/linear-regression-python-implementation/
    https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/
    https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1
    https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
    https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho
"""
import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

H = lambda x : 3 + 2*x

def grad_desc_simple():
    # theta_0 = theta_0 - alpha*(sum(H(x_i) - y_i)))
    # theta_1 = theta_0 - alpha*(sum(H(x_i) - y_i))*x_i)
    """ ripped from 
    https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1
    Just steps through a graph to minima
    """
    f = lambda x: (x+5)**2
    perfect_samples = np.array([(x, f(x)) for x in range(-10, 10)])
    print(perfect_samples)
    plot_regression_line(perfect_samples[:, 0], perfect_samples[:, 1])
    cur_x = 3 # The algorithm starts at x=3 
    rate = 0.2 # Learning rate 
    precision = 0.000001 #This tells us when to stop the algorithm 
    previous_step_size = 1 
    max_iters = 5000 # maximum number of iterations 
    iters = 0 #iteration counter 
    df = lambda x: 2*(x+5) #Gradient of our function

    while previous_step_size > precision and iters < max_iters:    
        prev_x = cur_x   #Store current x value in prev_x   
        cur_x = cur_x - rate * df(prev_x)  #Grad descent  
        # when rate*df(prev_x) == 0 we have found min
        previous_step_size = abs(cur_x - prev_x) #Change in x    

        iters = iters+1 
        #iteration count  
        print("Iteration",iters,"\nX value is",cur_x) 
        plt.scatter(cur_x, f(cur_x), color="b")

    #Print iterations
    print("The local minimum occurs at", cur_x)
    plot_regression_line(perfect_samples[:, 0], perfect_samples[:, 1])

def grad_desc_less_simple():
    # theta_0 = theta_0 - alpha*(sum(H(x_i) - y_i)))
    # theta_1 = theta_0 - alpha*(sum(H(x_i) - y_i))*x_i)
    """ ripped from 
    https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1
    Just steps through a graph to minima
    """
    f = lambda x, z: (3*z) + (x+5)**2 
    perfect_samples = np.array([(x, z, f(x, z)) for x,z in two_range(-10,10)])
    print(perfect_samples)
    plot_regression_line(perfect_samples[:, 0], perfect_samples[:, 1], perfect_samples[:, 2])
    cur_x = 3 # The algorithm starts at x=3 
    cur_z = -40 # The algorithm starts at z=.5 
    rate = 0.1 # Learning rate 
    precision = 0.00001 #This tells us when to stop the algorithm 
    previous_step_size = 1 
    max_iters = 1000 # maximum number of iterations 
    iters = 0 #iteration counter 
    dfz = lambda z: 3 #derivative with respect to z
    dfx = lambda x: 2*(x+5) #derivative with respect to x

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    while previous_step_size > precision and iters < max_iters:    
        prev_x = cur_x   #Store current x value in prev_x   
        prev_z = cur_z   #Store current x value in prev_x   
        cur_x = cur_x - rate * dfx(prev_x)  #Grad descent on x
        cur_z = cur_z - rate * dfz(prev_z)  #Grad descent on z
        
        previous_step_size = abs(cur_x - prev_x) + abs(cur_z - prev_z) #Change in x + change in z

        iters = iters+1 
        #iteration count  
        print("Iteration",iters,"\nX value is",cur_x) 
        print("Iteration",iters,"\nZ value is",cur_z) 
        ax.scatter(cur_x, f(cur_x, cur_z), cur_z, color="b")

    #Print iterations
    print("The local minimum occurs at", cur_x)
    plot_regression_line(perfect_samples[:, 0], perfect_samples[:, 1], perfect_samples[:, 2])



def plot_regression_line(x, y, z=None, b=None):
    """
    refrence from Https://www.geeksforgeeks.org/linear-regression-python-implementation/
    """
    # plotting the actual points as scatter plot
    if z is not None:
        fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        ax = fig.gca(projection='3d')
        ax.scatter(x, y, z, color = "m", marker = "o", s = 30)
    else:
        plt.scatter(x, y, color = "m", marker = "o", s = 30)
    if b:
        # predicted response vector
        y_pred = b[0] + b[1]*x 
        # plotting the regression line
        plt.plot(x, y_pred, color = "g")
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    #if z is not None:
    #    fig.zlabel('z')
    # function to show plot
    plt.show()

def two_range(start, stop):
    for i in range(start, stop):
        yield (i, random.random())
###########################
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
    except np.linalg.LinAlgError as e:
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
    #x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #X = np.array([[1],[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    x = np.array([4, 2, 3, 4, 5, 7, 7, 8, 12, 10])
    X = np.array([[4], [2], [3], [4], [5], [7], [7], [8], [12], [10]])
    print(f"x: {x}")
    print(f"X: {X}")
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    print(f"y: {y}")
    plot_regression_line(x, y) 
    # estimating coefficients
    #b = estimate_coef(x, y)  
    #return
    theta = normal_equation(X, y)
    print(f"theta: {theta}")
    print("Estimated coefficients:\ntheta_0 = {}\ntheta_1 = {}".format(theta[0], theta[1])) 
    # plotting regression line 
    plot_regression_line(x, y, theta) 
    
    
if __name__ == "__main__": 
    grad_desc_simple()
    grad_desc_less_simple()
    #main()

