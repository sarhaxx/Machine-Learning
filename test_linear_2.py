import projet_etu as l
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti, load_usps, get_usps, show_usps
import matplotlib.pyplot as plt
from Loss import MSELoss

trainx, trainy = gen_arti(nbex = 500, data_type = 0, epsilon = 0.1)
testx, testy = gen_arti(nbex = 500, data_type = 0, epsilon = 0.1)

linear1 = l.Linear((trainx.shape[1]), 20)
tan = l.TanH()
linear2 = l.Linear(20, (trainy.shape[1]))

mse = MSELoss()

max_iter1 = 100
for iter in range(max_iter1):

    #Forward
    z1 = linear1.forward(trainx)
    yhat = linear2.forward(z1)
    
    #Loss
    #loss = mse.forward(trainy,yhat)
    #print("Linear Iteration ",iter," : => LOSS : ", loss.sum())
    #Backwards
    bmse = mse.backward(trainy,yhat)
    bl2 = linear2.backward_delta(z1,bmse)
    bl1 = linear1.backward_delta(trainx,bl2)

    #Update paramaters and grad 0 

    linear2.backward_update_gradient(z1, bmse)
    linear1.backward_update_gradient(trainx, bl2)

    linear2.update_parameters(1e-3)
    linear1.update_parameters(1e-3)

    linear2.zero_grad()
    linear1.zero_grad()

def predict(X):
    z1 = linear1.forward(X)
    yhat = linear2.forward(z1)
    return np.sign(yhat)

print("accuracy = ", np.where(testy == predict(testx) , 1 , 0).sum()/len(testx))

plot_frontiere(testx, predict ,step=200)
plot_data(testx,testy)
plt.show()