import projet_etu as l
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti, load_usps, get_usps, show_usps
import matplotlib.pyplot as plt
from Loss import MSELoss

## (0, 1)

trainx, trainy = gen_arti(nbex = 500, data_type = 0, epsilon = 0.1)
testx, testy = gen_arti(nbex = 500, data_type = 0, epsilon = 0.1)


linear1 = l.Linear((trainx.shape[1]), (trainy.shape[1]))

mse = MSELoss()

max_iter1 = 100
for iter in range(max_iter1):
    #Forward
    yhat = linear1.forward(trainx)
    #Loss
    loss = mse.forward(trainy,yhat)
    #Backwards
    bmse = mse.backward(trainy,yhat)
    bl1 = linear1.backward_delta(trainx,bmse)
    
    #Update paramaters and grad 0 
    linear1.backward_update_gradient(trainx, bmse)
    linear1.update_parameters(1e-3)
    linear1.zero_grad()
    
def predict(X):
    yhat = linear1.forward(X)
    return np.sign(yhat)

print("Linear Iteration ",iter," : => LOSS : ", loss.sum())

print("accuracy = ", np.where( testy == predict(testx) , 1 , 0).sum()/len(testx))

plot_frontiere(testx, predict ,step=200)
plot_data(testx,testy)
plt.show()