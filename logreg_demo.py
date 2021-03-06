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
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image

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

def grad_desc(x, y, function, dfunction):
    # theta_0 = theta_0 - alpha*(sum(H(x_i) - y_i)))
    # theta_1 = theta_0 - alpha*(sum(H(x_i) - y_i))*x_i)
    """ ripped from 
    https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1
    Just steps through a graph to minima
    """
    samples = np.array([(x, f(x, z)) for x in two_range(-10,10)])
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

def grad_decent_real():
    """!
    This finds some info from real data
    """
    #reads in data
    test_data = []
    with open("Car_details_v3.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_data.append(row)
    # lets take a look at the data
    print(dict_print(test_data[3]))
    print("data length:", len(test_data))
    # ovbiously max power is the most important, lets look at that and clean it
    keys_to_delete = []
    for row in test_data:
        if 'max_power' not in row: 
            print("deleting row: ")
            print(dict_print(row))
            keys_to_delete.append(row)
            continue
        if 'engine' not in row:
            print("deleting row: ")
            print(dict_print(row))
            keys_to_delete.append(row)
            continue
        if row['max_power'] == '':
            print("deleting row: ")
            print(dict_print(row))
            keys_to_delete.append(row)
            continue
        if row['engine'] == '':
            print("deleting row: ")
            print(dict_print(row))
            keys_to_delete.append(row)
            continue
        #special cases of garbage
        if row['max_power'] == ' bhp':
            print("deleting row: ")
            print(dict_print(row))
            keys_to_delete.append(row)
            continue

    # getting rid of all the bad data in probably a dumb way
    print(len(test_data))
    for row in keys_to_delete:
        test_data.remove(row)
     
    print(len(test_data))
    print(f"deleted {len(keys_to_delete)}")
    #so the above finnaly worked to delete the junk
    """
    it would be nice to make this something we can work with, lets make a graph 
    and get rid if the units
    """
    power = [float(row['max_power'].strip(" bhp")) for row in test_data]
    engine_s = [float(row['engine'].strip(" CC")) for row in test_data]
    price = [float(row['selling_price']) for row in test_data]

    # lets plot some
    #plot_regression_line(power, engine_s)
    plot_regression_line(power[:100], engine_s[:100], price[:100])

    # lets see what our line of best fit finds
    shaped_x = np.array(power)
    print(shaped_x)
    # we need to add 1 to X, this represent the non varible term
    shaped_x = np.c_[ np.ones(len(power)), shaped_x ]
    print(shaped_x)
    shaped_y = np.array(engine_s)
    print(shaped_y)
    theta = normal_equation(shaped_x, shaped_y)
    print(theta)
    # now we have theta, lets slap that line in and see how she looks (prob bad)
    plot_regression_line(np.array(power), np.array(engine_s), b=theta)


def dict_print(d):
    """!
    you still need to do the print yourself
    """
    return json.dumps((dict(d)), indent=4, sort_keys=True)

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
    if b is not None:
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
    print("X transpose is: ")
    print(Xt)
    # Xt dot X is the dot product (matrix mutiplication) of the two. remember that for this to work, they must be (m,n) * (n,m) dimentions and therefor Xt dot X != X dot Xt
    Xt_dot_X = np.dot(Xt, X)
    print("X transpose dot X is:")
    print(Xt_dot_X)
    # now for the inversion of Xt_dot_X, this is the most computationaly heavy step, im told its O(n^3), so over about n = 10,000 gradient decent, 
    # might be a better choice. n is the number of features
    try:
        inv_Xt_dot_X = np.linalg.inv(Xt_dot_X)
    except np.linalg.LinAlgError as e:
        print("a sigular matrix is not invertable, it will be div zero error")
        print("but numpy.linalg.pinv gives us a way to fake it out by replacing zero with really small floats")
        inv_Xt_dot_X = np.linalg.pinv(Xt_dot_X)
    print("now lets do inv_Xt_dot_X dot Xt")
    inv_Xt_dot_X_dot_Xt = np.dot(inv_Xt_dot_X, Xt)
    print("X transpose dot X inverted dot X transpose is:")
    print(inv_Xt_dot_X_dot_Xt)
    print("last step, now we do inv_Xt_dot_X_dot_Xt_dot_y.  this will be theta matrix")
    theta = np.dot(inv_Xt_dot_X_dot_Xt, y)
    print("X transpose dot X inverted dot X transpose dot y (theta) is:")
    print(theta)
    print("theta calcutated to be: ".format(theta))
    return theta

##### logreg  #####
## help from https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc 
## for color picking https://www.rapidtables.com/web/color/RGB_Color.html
###################
## Constants ##
# in BGR!!
mask_of_green = ((150,255,0), (0, 255,225))
## End Constants ##

## math bits ##
def weightInitialization(n_features):
    w = np.zeros((1, n_features))
    b = 0
    return w, b

def sigmoid_activation(result):
    return 1/(1+np.exp(-result))

def model_optimize(w, b, X, Y):
    m = X.shape[0]

    #prediction
    final_result = sigmoid_activation(np.dot(w, X.T) + b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1 - Y_T) * (np.log(1 - final_result)))))
    # gradient calc
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))
    grads = {"dw": dw, "db": db}

    return grads, cost

def model_predict(w, b, X, Y, learning_rate, num_interations):
    costs = []
    for i in range(num_interations):
        grads, cost = model_optimize(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        # weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)

        if (i % 100 == 0):
            costs.append(cost)
            print(f"cost after {i} interation is {cost}")

    #final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}

    return coeff, gradient, costs

def predict(final_pred, m):
    '''
    final_pred is sigmoid activation funtion of final function 
    eg: sigmoid(np.dot(w, X) + b) where w and b have been found by the learning algo
    '''
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.32:
            y_pred[0][i] = 1
    return y_pred
## end math bits ##

## helper bits ##
def give_random_max_saturation_color():
    '''
    In BGR (blue, green, red) since thats what cv2 likes
    Only using colors with max saturation to make it easy to define what green is
    '''
    pick_255 = random.randint(0,2)
    if pick_255 == 0:
       return (255, random.randint(0,255), random.randint(0,255))
    if pick_255 == 1:
       return (random.randint(0,255), 255, random.randint(0,255))
    if pick_255 == 2:
       return (random.randint(0,255), random.randint(0,255), 255)

def show_color(bgr_tuple, size=(100,100)):
    # given bgr tuple of ints, display color
    rgb_tuple = bgr_tuple[::-1]
    just_color = Image.new("RGB", size, rgb_tuple)
    just_color.show(5)
## end helper bits ##

## data bits ##
def generate_labeled_data(size=5000):
    '''
    Since for our green definition the red is going up as we prgress
    through the green spectrum, we just measure that

    returns (bgr_color_tuple, bool_is_green)
    '''
    for i in range(size):
        color = give_random_max_saturation_color()
        #color = (0, 255, 68)
        #print("green red bottom: ", mask_of_green[0][2])
        #print("green red top: ", mask_of_green[1][2])
        #print("color red part value: ", color[2])
        #print(mask_of_green[0][2] < color[2] < mask_of_green[1][2])
        red_check = mask_of_green[0][2] < color[2] < mask_of_green[1][2]
        blue_check = mask_of_green[0][0] > color[0] > mask_of_green[1][0]
        if color[1] == 255 and red_check and blue_check:
            is_green = True
        else:
            is_green = False
        #if is_green:
        #    print(color)
        #    show_color(color)
        yield (normalize_color_values(color), is_green)

def generate_unlabled_data(size=5000):
    for i in range(size):
        yield give_random_max_saturation_color()

def normalize_color_values(color):
    return (color[0]/255, color[1]/255, color[2]/255)

## end data bits ##

## reporting bits ##
def confusion_matrix_data(num_samples, numiter, learnrate, y_prediction, y_real):
    """
    prints totals pos and neg and related data results
    referance https://www.unite.ai/what-is-a-confusion-matrix/
    """
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(num_samples):
        if y_real[i] == y_prediction[0][i]:
            if y_real[i] == 0:
               true_neg += 1
            if y_real[i] == 1:
               true_pos += 1
        if y_real[i] != y_prediction[0][i]:
            if y_real[i] == 1:
               false_neg += 1
            if y_real[i] == 0:
               false_pos += 1

    accuracy = (true_pos + true_neg) / num_samples # ratio correctly predicted
    try:
        precision = true_pos / (true_pos + false_pos)  # ratio of correct positive to all positives predicted
    except ZeroDivisionError as e:
        print("predicted no positives at all!")
        precision = -1
    try:
        recall = true_pos / (true_pos + false_neg)  # ratio of correct positive to all positives actually present
    except ZeroDivisionError as e:
        print("There where no positives in the data!")
        recall = -1
    print(f"with current params (NUMITER={numiter}, LEARNRATE={learnrate}, SAMPLES={num_samples}), model got this on the training set: \n" \
            f"true_pos: {true_pos} \n" \
            f"true_neg: {true_neg} \n" \
            f"false_pos: {false_pos} \n" \
            f"false_neg: {false_neg} \n" \
            f"Accuracy: {accuracy} \n" \
            f"Precision: {precision} \n" \
            f"Recall: {recall}")
    

## end reporting bits ##
if __name__ == "__main__": 
    #grad_desc_simple()
    #grad_desc_less_simple()
    #grad_decent_real()
    #main()

    #show_color(give_random_max_saturation_color())

    #print("green limits:")
    #show_color(mask_of_green[0])
    #show_color(mask_of_green[1])
    
    NUMITER = 50000
    LEARNRATE = 0.001
    SAMPLES = 10000

    print(f"getting {SAMPLES} labeled samples:")
    labled_data = list(generate_labeled_data(SAMPLES))
    greens = [d[0] for d in labled_data if d[1]]
    print("percent_green: ", (len(greens) / len(labled_data)*100))
    
    # logistic regression
    X = np.array([[d[0][0], d[0][1], d[0][2]] for d in labled_data])
    Y = np.array([int(d[1]) for d in labled_data])
    w, b = weightInitialization(3) # 3 is for Blue, green, red
    coeff, gradient, costs = model_predict(w, b, X, Y, learning_rate=LEARNRATE, num_interations=NUMITER)
    print("final values for coeffcient weights and intercept: ", coeff)
    final_w = coeff["w"]
    final_b = coeff["b"]
    final_prediction = sigmoid_activation(np.dot(final_w, X.reshape(3, X.shape[0])) + final_b)
    print("final_prediction: ", final_prediction)
    y_pred = predict(final_prediction, X.shape[0])
    confusion_matrix_data(SAMPLES, NUMITER, LEARNRATE, y_pred, Y) 
    # try a fresh one
    fresh_test_set = list(generate_labeled_data(500))
    X_fresh = np.array([[d[0][0], d[0][1], d[0][2]] for d in fresh_test_set])
    Y_fresh = np.array([int(d[1]) for d in labled_data])
    fresh_pred = sigmoid_activation(np.dot(final_w, X_fresh.reshape(3, X_fresh.shape[0]))+ final_b)
    y_fresh_pred = predict(fresh_pred, X_fresh.shape[0])
    confusion_matrix_data(500, NUMITER, LEARNRATE, y_fresh_pred, Y_fresh)
    
    
