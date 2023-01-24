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
