import numpy as np
import matplotlib.pyplot as plt

# function used to load data from a file.
def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y