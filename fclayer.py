import numpy as np 

import sys
import os 

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

from utils import Tools

class FCLayer(object):
    
    def __init__(self, name, miniBatchesSize, i_size, o_size, activatorCls, optimizerCls, optParams, 
                 needReshape, dataType, init_w):
        self.name = name
        self.miniBatchesSize = miniBatchesSize
        self.i_size = i_size
        self.o_size = o_size
        self.activator = activatorCls
        self.optimizerObj = optimizerCls(optParams, dataType)
        self.needReshape = needReshape
        self.dataType = dataType
        self.w = init_w * np.random.randn(i_size, o_size).astype(dataType)
        self.b = np.zeros(o_size, dataType)
        self.out = []
        self.deltaPrev = []
        self.deltaOri = []
        self.shapeOfOrinIn = ()
        self.inputReshaped = []
        
    def forward_propagation(self, input):
        self.shapeOfOrinIn = input.shape
        self.inputReshaped = input if self.needReshape is False else input.reshape(input.shape[0], -1)
        self.out = self.activator.activate(Tools.matmul(self.inputReshaped, self.w)+self.b)
        return self.out
    
    def inference(self, input):
        self.shapeOfOrinIn = input.shape
        self.out = self.forward_propagation(input)
        return self.out
    
    def bpDelta(self):
        deltaPrevReshaped = Tools.matmul(self.deltaOri, self.w.T)
        self.deltaPrev = deltaPrevReshaped if self.needReshape is False else deltaPrevReshaped.reshape(self.shapeOfOrinIn)
        return self.deltaPrev
    
    def bpWeights(self, input, lr):
        dw = Tools.matmul(self.inputReshaped.T, self.deltaOri)
        db = np.sum(self.deltaOri, axis=0, keepdims=True).reshape(self.b.shape)
        weight = (self.w, self.b)
        dweight = (dw, db)
        self.w, self.b = self.optimizerObj.getUpweights(weight, dweight, lr)
        
    def back_propagation(self, input, delta, lr):
        
        self.deltaOri = self.activator.backward(delta, self.out)
        self.bpDelta()
        self.bpWeights(input, lr)
        return self.deltaPrev
        