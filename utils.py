import numpy as np
import matplotlib.pyplot as plt

# function used to load data from a file.
def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y

# function used to plot data.
def plot_data(X, y, positive_class_label="y=1", negative_class_label="y=0"):
    pos = y == 1
    neg = y == 0
    
    # plot examples
    plt.plot(X[pos, 0], X[pos, 1], 'kx', label=positive_class_label)
    plt.plot(X[neg, 0], X[neg, 1], 'rx', label=negative_class_label)

# feature mapping function for polynomial features.
def map_feature(X1, X2):
  
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

# defining the sigmoid function for plotting the decision boundary.
def sigmoid(z):
 
    return 1/(1+np.exp(-z))

# function used to plot the decision boundary.
def plot_decision_boundary(X, y, w, b):

    plot_data(X[:, 0:2], y)
    
    if X.shape[1] <= 2:

        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        plt.plot(plot_x, plot_y, c="b")
        
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))

        # evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = sigmoid(np.dot(map_feature(u[i], v[j]), w) + b)
        
        # important to transpose z before calling contour       
        z = z.T
        
        # plot z = 0.5
        plt.contour(u,v,z, levels = [0.5], colors="g")