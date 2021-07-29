from typing import Callable
import numpy as np 
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

from utils import Tools
from lossfunc import *

class minibatch_iteration(object):
    
    def iterate_minibatches(input, target, batch_size, shuffle=False):
        assert input.shape[0] == target.shape[0]
        if shuffle:
            indices = np.arange(input.shape[0])
            np.random.shuffle(indices)  
        for start_idx in range(0, input.shape[0]-batch_size+1, batch_size):
            if shuffle:
                output = indices[start_idx:start_idx:batch_size]
            else:
                output = slice(start_idx, start_idx+batch_size)
            yield input[output], target[output]


class Session(object):
    """对不同种类的神经层实现训练过程"""
    
    def __init__(self, layers, lossfunccls:Callable):
        self.layers = layers
        self.input = []
        self.losscls = lossfunccls
        
    def inference(self, train_data, y_, val=False):
        curr_batch_size = len(y_)
        self.input = train_data
        datalayer = train_data
        
        if val == False:
            # forward for training
            for layer in self.layers:
                datalayer = layer.forward_propagation(datalayer)
                # print(f"shape: {datalayer.shape}")
        else:
            # forward for prediction
            for layer in self.layers:
                datalayer = layer.inference(datalayer)
        
        y = datalayer
        data_loss, delta, acc = self.losscls.loss(y, y_, curr_batch_size)
        return y, data_loss, delta, acc

    def bp(self, delta, lr):
        deltaLayer = delta
        for i in reversed(range(1,len(self.layers))):
            deltaLayer = self.layers[i].back_propagation(self.layers[i-1].out, deltaLayer, lr)
            
        self.layers[0].back_propagation(self.input, deltaLayer, lr)
    
    def train_steps(self, train_data, y_, lr):
        _,loss,delta,acc = self.inference(train_data, y_, val=False)
        self.bp(delta, lr)
        return acc,loss
    
    def validation(self, data_v, y_v):
        y,loss,_,acc = self.inference(data_v, y_v, val=True)
        return y, loss, acc
        

class Optimizer(object):
    def __init__(self, optParams=None, dataType=np.float):
        if optParams is not None:
            self.gamma, self.eps = optParams
        self.dataType = dataType
        self.isInited = False
        self.v = []
        
    def initV(self, w):
        if (self.isInited == False):
            for i in range(len(w)):
                self.v.append(np.zeros(w[i].shape, dtype=self.dataType))
            self.isInited = True
    
    def getUpweights(self, w, dw, lr):
        self.initV(w)
        wNew = []
        for i in range(len(w)):
            wi, self.v[i] = self.optimize(w[i], dw[i], self.v[i], lr)
            wNew.append(wi)
        
        return tuple(wNew)
    
    def optimize(self):
        pass



class SGDOptimizer(Optimizer):
    def __init__(self, optParams, dataType):
        super().__init__(optParams, dataType)
        assert (self.dataType is not None)
        
    def initV(self, w):
        return super().initV(w)
        
    def getUpweights(self, w, dw, lr):
        wNew = []
        for i in range(len(w)):
            wi = self.OptimzSGD(w[i], dw[i], lr)
            wNew.append(wi)
        return tuple(wNew)
    
    def optimize(self, x, dw, lr):
        x -= lr*dx
        return x
    
    
class MomentumOptimizer(Optimizer):
    def __init__(self, optParams, dataType):
        super().__init__(optParams, dataType)
        assert(self.optParams is not None)
    
    def initV(self, w):
        return super().initV(w)
            
    # def getUpweights(self, w, dw, lr):
    #     self.initV(w)
        
    #     wNew = []
    #     for i in range(len(w)):
    #         wi, self.v[i] = self.OptimzMomentum(w[i], dw[i], self.v[i], lr)
    #         wNew.append(wi)
        
    #     return tuple(wNew)
    def getUpweights(self, w, dw, lr):
        return super().getUpweights(w, dw, lr)
    
    def optimize(self, x, dx, v, lr):
        v = self.gamma * v + lr*dx
        x -= v
        return x, v
    
class NAGOptimizer(Optimizer):
    def __init__(self, optParams, dataType):
        super().__init__(optParams, dataType)
        assert(self.optParams is not None)
        # assert(grad_func is not None)
        # if not callable(grad_func):
        #     raise Exception("The gradient function is not callable")
    
    def initV(self, w):
        return super().initV(w)
    
    # def getUpweights(self, w, dw, lr):
    #     self.initV(w)
    #     wNew = []
        
    #     for i in range(len(w)):
    #         wi, self.v[i] = self.optimize(w[i], dw[i], self.v[i], lr)
    #         wNew.append(wi)
    #     return tuple(wNew)
    def getUpweights(self, w, dw, lr):
        return super().getUpweights(w, dw, lr)
    
    def optimize(self, x, dx, v, lr):
        vt = self.gamma*v + lr*dx
        x += self.gamma*v - (1.+self.gamma)*vt
        
        return x, vt


class AdagradOptimizer(Optimizer):
    
    def __init__(self, optParams=None, dataType=np.float):
        self.eps = optParams
        self.dataType = dataType
        self.isInited = False
        self.g = []
        
    def initV(self, w):
        return super().initV(w)
    # def getUpweights(self, w, dw, lr):
    #     self.initV(w)
    #     wNew = []
    #     for i in range(len(w)):
    #         wi, self.v[i] = self.optimize(w[i], dw[i], self.v[i], lr)
    #         wNew.append(wi)
    #     return tuple(wNew)
    def getUpweights(self, w, dw, lr):
        return super().getUpweights(w, dw, lr)
    
    def optimize(self, x, dx, v, lr):
        g = v + dx ** 2
        x -= lr * dx / (np.sqrt(g), self.eps)
        return x, g
    

class RMSOptimizer(Optimizer):
    
    def __init__(self, optParams=None, dataType=np.float):
        super().__init__(optParams=optParams, dataType=dataType)
        
    def initV(self, w):
        return super().initV(w)
    # def getUpweights(self, w, dw, lr):
    #     self.initV(w)
    #     wNew = []
    #     for i in range(len(w)):
    #         w
    def getUpweights(self, w, dw, lr):
        return super().getUpweights(w, dw, lr)
    
    def optimize(self, x, dx, v, lr):
        v_new = self.gamma*v + (1-self.gamma)*(dx**2)
        x -= lr * dx / (np.sqrt(v_new)+self.eps)
        return x, v
    

class AdamOptimizer(object):
    
    def __init__(self, optParams, dataType):
        self.beta1, self.beta2, self.eps = optParams
        self.dataType = dataType
        self.isInited = False
        self.m = []
        self.v = []
        self.Iter = 0
        
    def initMV(self, w):
        if (self.isInited == False):
            for i in range(len(w)):
                self.m.append(np.zeros(w[i].shape, dtype=self.dataType))
                self.v.append(np.zeros(w[i].shape, dtype=self.dataType))
            self.isInited = True
            
    def getUpweights(self, w, dw, lr):
        self.initMV(w)
        
        t = self.Iter + 1
        wNew = []
        for i in range(len(w)):
            wi, self.m[i], self.v[i] = self.optimize(w[i], dw[i], self.m[i], self.v[i], lr, t)
            wNew.append(wi)
        return tuple(wNew)
    
    def optimize(self, x, dx, m, v, lr, t):
        m = self.beta1*m + (1-self.beta1)*dx
        mt = m / (1 - self.beta1**t)
        v = self.beta2 * v + (1 - self.beta2) * (dx**2)
        vt = v / (1 - self.beta2**t)
        x -= lr * mt / (np.sqrt(vt) + self.eps)
        return x, m, v
    
    

    
        
        