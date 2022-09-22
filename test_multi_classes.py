from sklearn.metrics import ConfusionMatrixDisplay
from keras.datasets import mnist
import projet_etu as nn
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti, load_usps, get_usps, show_usps
import matplotlib.pyplot as plt
from Loss import MSELoss, CELoss
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

# Accuracy 0.8759342301943199
# avec les données usps conrairement a mnist, on obtenu convergenace qu'apres 5000 iterations
# avce un pas de gradient egale a 1e-5 avec biais
# De même que les algorithmes présentés ci-dessus, les réseaux de neurones (neural networks) sont 
# biaisés en faveur de la classe majoritaire (i.e. la très grande majorité de l’espace sera prédite
# comme ayant une probabilité nulle de survenance d’un individu minoritaire), et dans les cas extrêmes 
# ignorent complètement les individus minoritaires.


uspsdatatrain = "./data/USPS_train.txt"
uspsdatatest = "./data/USPS_test.txt"
alltrainx,alltrainy = load_usps(uspsdatatrain)
alltestx,alltesty = load_usps(uspsdatatest)
x_train, y_train = get_usps([0,1,2,3,4,5,6,7,8,9],alltrainx,alltrainy)
x_test, y_test = get_usps([0,1,2,3,4,5,6,7,8,9],alltestx,alltesty)


X1 = x_train/ 255.0
X2 = x_test / 255.0

y1 = y_train
y2 = y_test 

x_train, y_train = shuffle(X1, y1, random_state=0)
x_test, y_test = shuffle(X2, y2, random_state=0)

y_train_sparse = np.zeros((y_train.size,10))
y_train_sparse[np.arange(y_train.size),y_train] = 1

y_test_sparse= np.zeros((y_test.size,10))
y_test_sparse[np.arange(y_test.size),y_test] = 1

linear1 = nn.Linear((x_train.shape[1]),128)
linear2 = nn.Linear(128,64)
linear3 = nn.Linear(64,10)
linear4 = nn.Linear(64,10)
tan = nn.TanH()
sig = nn.Sigmoide()
softmax = nn.softmax()
loss = CELoss()

p_loss = []



L = [linear1,tan,linear2,tan,linear3]

seq = nn.Sequentiel(L)

for i in range(0):
    opt = nn.Optim(seq, loss, 1e-5)
    delta = opt.step(x_train, y_train_sparse)
    p_loss.append(delta)

def predict_multi(input):
    yhat = L[0].forward(input)
    for i in range(1,len(L)):
        yhat = L[i].forward(yhat)

    yhat = softmax.forward(yhat)
    yhaat = np.zeros(len(input))


    for i in range(len(yhat)):
        yhaat[i] = np.argmax(yhat[i])
    return yhaat

#y_pred = predict_multi(x_test)

#nb = np.where(y_pred != y_test, 0, 1).mean()
#print(nb)
#ConfusionMatrixDisplay.from_predictions( y_test, y_pred, labels = np.arange(10) )
#plt.show()



####################################################################################
print("mnist")


#Accuracy  : 0.8762
# On a un score de 0.8227 avec les données de MNIST en 200 ietrations,
# avec epsilon à 1e-5 3 lineaire et deux tangeante  (lin,tan,lin ,tan,lin dans cet ordre)
# on a la matrice de confusion 
# et la le graphe de la loss 

#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(train_X.shape[0], train_X.shape[1]**2) / 255.0
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1]**2) / 255.0


y_train_sparse = np.zeros((train_y.size,10))
y_train_sparse[np.arange(train_y.size),train_y] = 1

y_test_sparse = np.zeros((test_y.size,10))
y_test_sparse[np.arange(test_y.size),test_y] = 1


linear1 = nn.Linear((train_X.shape[1]),128)
linear2 = nn.Linear(128,64)
linear3 = nn.Linear(64,10)
linear4 = nn.Linear(64,10)

L = [linear1,tan,linear2,tan,linear3]
seq = nn.Sequentiel(L)


hist_loss = nn.SGD(seq,train_X, y_train_sparse, 5, 200, loss, eps = 1e-5)

def predict_multi(input):
    yhat = L[0].forward(input)
    for i in range(1,len(L)):
        yhat = L[i].forward(yhat)

    yhat = softmax.forward(yhat)
    yhaat = np.zeros(len(input))

    for i in range(len(yhat)):
        yhaat[i] = np.argmax(yhat[i])
    return yhaat

y_pred = predict_multi(test_X)

plt.figure()
plt.legend("Variation du cout par epoch")
plt.xlabel("epochs")
plt.ylabel("Cout")
plt.plot([1, 2, 3, 4, 5] , hist_loss, 'r-')
plt.show()

print("Accuracy  :" , np.where(y_pred != test_y, 0, 1).mean())
ConfusionMatrixDisplay.from_predictions( test_y, y_pred, labels = np.arange(10) )
plt.show()


############################################""

