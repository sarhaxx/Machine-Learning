from sklearn.metrics import ConfusionMatrixDisplay
from keras.datasets import mnist
import projet_etu as l
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti, load_usps, get_usps, show_usps
import matplotlib.pyplot as plt
from Loss import MSELoss, CELoss
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


(train_X, train_y), (test_X, test_y) = mnist.load_data()



plt.hist(train_y, bins=10)
plt.show()