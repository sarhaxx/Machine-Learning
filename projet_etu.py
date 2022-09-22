from json import load
from turtle import forward
import numpy as np
from Loss import MSELoss, CELoss


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None
    def zero_grad(self):
        ## Annule gradient
        pass
    def forward(self, X):
        ## Calcule la passe forward
        pass
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass
    
class Linear(Module):
    def __init__(self,input,output):
        self._parameters = np.random.uniform(-1, 1, (input, output)) # input * output
        self._bias = np.random.uniform(-1, 1, (1, output)) 
        self._gradient = np.zeros((input, output))   # input * output
        self._b_grad = np.zeros((1, output))         # input * output
        self.input = input      
        self.output = output    

    def zero_grad(self):
        ## Annule gradient
        self._gradient = np.zeros((self.input, self.output))
        self._b_grad = np.zeros((1,self.output))

    def forward(self, input):
        ## Calcule la passe forward
        assert input.shape[1] == self.input
        return (input @ self._parameters) + self._bias # batch * output

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step * self._gradient # intput * output
        self._bias -= gradient_step * self._b_grad

    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self.input
        assert delta.shape[1] == self.output
        assert input.shape[0] == delta.shape[0]

        self._gradient += input.T @ delta 
        self._b_grad += np.sum(delta, axis = 0)
        
    def backward_delta(self, input, delta):
        assert input.shape[1] == self.input
        assert delta.shape[1] == self.output
        return delta @ self._parameters.T

class TanH(Module):

    def forward(self, input):
        return np.tanh(input) # input * input

    def backward_delta(self, input, delta):
        return (1 - np.tanh(input) ** 2) * delta 

    def update_parameters(self, gradient_step=1e-3):
        pass
    def backward_update_gradient(self, input, delta):
        pass 
    def zero_grad(self):
        pass
   
class Sigmoide(Module):


    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward_delta(self, input, delta):
        sig  = 1/(1 + np.exp(-input))
        return delta * (sig * (1 - sig))

    def update_parameters(self, gradient_step=1e-3):
        pass
    def backward_update_gradient(self, input, delta):
        pass
    def zero_grad(self):
        pass


class softmax(Module):

    def forward(self, input):
        exp_input = np.exp(input)
        return exp_input / exp_input.sum(axis = 1).reshape((-1,1))
    
    def backward_delta(self, input, delta):
        exp_input = np.exp(input)
        res = exp_input / np.sum(exp_input, axis=1).reshape((-1, 1))
        return delta * (res * (1 - res))

    def update_parameters(self, gradient_step=1e-3):
        pass
     
class Sequentiel():
    def __init__(self, L):
        self.layers = L
        self.input = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        self.input = [input]
        for i in range(len(self.layers)):
            self.input.append(self.layers[i].forward(self.input[-1]))
        return self.input[-1]

    def backward(self, delta):
        for i in range(len(self.layers)-1,-1,-1):
            self.layers[i].backward_update_gradient(self.input[i] , delta)
            delta = self.layers[i].backward_delta(self.input[i], delta)

    def update_parameters(self, eps):
        for layer in self.layers:
            layer.update_parameters(eps)

    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

            
class Optim():

    def __init__(self, net, loss = MSELoss(), eps= 1e-3):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        #FORWARD
        yhat = self.net.forward(batch_x)
        #LOSS
        loss = self.loss.forward(batch_y, yhat)
        delta = self.loss.backward(batch_y, yhat)
        #BACKWARD
        self.net.backward(delta)
        self.net.update_parameters(self.eps)
        self.net.zero_grad()
        print(loss.mean())
        return loss

def SGD(net, datax, datay, batch, max_iter, loss, eps):
    batch_x = np.array_split(datax, batch)
    batch_y = np.array_split(datay, batch)
    hist_loss = []

    for i in range(batch):
        op = Optim(net,loss,eps)
        for iter in range(max_iter):
            l = (op.step(batch_x[i],batch_y[i]))
            print("epoch : ", i, "iteration :", iter, " loss :", l.mean())

        hist_loss.append(l.mean())
    return hist_loss