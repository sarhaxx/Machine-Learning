from ast import Break
from cgi import test
from projet_etu import Linear,TanH, Sequentiel, Optim, Sigmoide
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
import matplotlib.pyplot as plt
from Loss import MSELoss

# (0 ,1)


trainx, trainy = gen_arti(nbex = 500, data_type = 2, epsilon = 0.1)
testx, testy = gen_arti(nbex = 500, data_type = 2, epsilon = 0.1)

trainy = np.where(trainy==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))


linear1 = Linear(2, 100)
linear2 = Linear(100, 50)
linear3 = Linear(50 ,1)

tan = TanH()
sig = Sigmoide()

mse = MSELoss()

L = [linear1, tan, linear2, tan, linear3, sig]

seq = Sequentiel(L)

max_iter2 = 500
for iter in range(max_iter2):
    #Calcul forward
    op = Optim(seq, mse, 1e-4)
    op.step(trainx, trainy)

def predict(X):
    yhat = L[0].forward(X)
    for i in range(1,len(L)):
        yhat = L[i].forward(yhat)
    return np.where(yhat >= 0.5 , 1, 0)

print("accuracy = ", np.where(testy == predict(testx) , 1 , 0).sum() / len(testx))

plot_frontiere(testx, predict ,step=200)
plot_data(testx,testy)
plt.show()
