import numpy as np 

class ReLU(object):
    @staticmethod
    def activate(x):
        return np.maximum(0,x)
    
    @staticmethod
    def backward(delta, x):
        delta[x<=0]  = 0
        return delta
    
class Sigmoid(object):
    def __init__(self):
        self.o = []
        
    def activate(self, x):
        self.o = 1. / (1+np.exp(-x))
        return self.o
    
    def backward(self, _1, _2):
        return self.o * (1 - self.o)
    
class Tanh(object):
    def __init__(self):
        self.o = []
        
    def activate(self, x):
        self.o = np.tanh(x)
        return self.o
    
    def backward(self, _1, _2):
        return 1 - self.o ** 2
    
class Pass(object):
    @staticmethod
    def activate(x):
        return x
    
    def backward(delta, x):
        return delta