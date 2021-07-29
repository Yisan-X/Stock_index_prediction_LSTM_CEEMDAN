import numpy as np
import operator as op
import numba

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from utils import Tools
from activator import *

class RnnLayer(object):
    def __init__(self, Name, miniBatchesSize, nodesNum, LayersNum, optimizerCls, optmParams, dropoutRate, dataType, init_reg):
        self.name = Name
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = LayersNum
        self.dataType = dataType
        self.init_reg = init_reg
        self.isInit = False
        self.dropoutRate = dropoutRate
        self.dropoutMask = []
        
        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(LayersNum)]
        self.rnnParams = []
        self.deltaPrev = [] # The error of the last layer
        
    def _initWeight(self, D, H, layersNum, dataType):
        rnnParams = []
        for layer in range(layersNum):
            Wh = np.random.uniform(-1*self.init_reg, self.init_reg, (H,H)).astype(dataType)
            if (layer == 0):
                Wx = np.random.uniform(-1*self.init_reg, self.init_reg, (D,H)).astype(dataType)
            else:
                Wx = np.random.uniform(-1*self.init_reg, self.init_reg, (H,H)).astype(dataType)
            b = np.zeros(H, dataType)
            rnnParams.append({"Wx": Wx, "Wh": Wh, "b": b})
        self.rnnParams = rnnParams
        
    def _initWeightOrthogonal(self, D, H, layersNum, dataType):
        
        rnnParams = []
        for layer in range(layersNum):
            Wh = Tools._initWeightOrthogonal( (H,H), self.init_reg, dataType)
            DH = D if layer==0 else H
            Wx = Tools.initOrthogonal( (DH, H), self.init_reg, dataType)
            b = np.zeros(H, dataType)
            rnnParams.append({"Wx": Wx, "Wh": Wh, "b": b})
        self.rnnParams = rnnParams
        
    
    def rnn_step_forward(self, x, prev_h, Wx, Wh, b):
        next_h, cache = None
        Z = Tools.matmul(Wx, x) + Tools.matmul(Wh, prev_h) + b
        next_h = np.tanh(Z)
        dtanh = Tools.grad_tanh(next_h)
        cache = (x, prev_h, Wx, Wh, dtanh)
        return next_h, cache
    
    def rnn_forward(self, x):
        """
        N: the number of samples
        T: the number of times
        D: The dimension of each nodes
        H: the number of nodes
        """
        N,T,D = x.shape
        L = self.layersNum
        H = self.rnnParams[0]["b"].shape
        xh = x
        for layer in range(L):
            h = np.zeros((N,T,H))
            h0 = np.zeros((N,H))
            cache = []
            for t in range(T):
                h[:,t,:], temp_cache = self.rnn_step_forward(xh[:,t,:], 
                                                             h[:,t-1,:] if t > 0 else h0,
                                                             self.rnnParams[layer]["Wx"], 
                                                             self.rnnParams[layer]["Wh"],
                                                             self.rnnParams[layer]["b"])
                cache.append(temp_cache)
            xh = h # the input data of the next layer
            self.rnnParams[layer]["h"] = H
            self.rnnParams[layer]["cache"] = cache
            
        return h
            
    def inference(self, x):
        """
        N: the number of samples
        T: the number of times
        D: the dimension of each input
        H: the dimension of the layer
        L: the number of layers
        """
        N,T,D = x.shape 
        H = self.nodesNum
        L = self.layersNum
        if (self.isInit == False):
            self._initWeightOrthogonal(D,H,L,self.dataType)
            self.isInit = True
        
        h = self.rnn_forward(x)
        self.out = h 
        return self.out
    
    def forward_propagation(self, input):
        out_temp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropoutRNN(out_temp, self.dropoutRate)
        return self.out
    
    def rnn_step_backward(self, dnext_h, cache):
        """
        cache: the temporary memory from the forward propogation
        cache = (x, prev_h, Wx, Wh, dtanh)
        dnext_h.shape = (N, H)
        """
        dx, dprev_h, dWx, dWh, db = None, None, None, None, None # The gradients we need to calculate
        x, prev_h, Wx, Wh, dtanh = cache
        dz = dnext_h * dtanh 
        # \prod_{t=k+1}^T \grad{h^{t}}{h^{t-1}} * \grad{h^k}{X}
        dx = Tools.matmul(dz, Wx.T)
        # \prod_{t=k+1}^T \grad{h^{t}}{h^{t-1}} * \grad{h^k}{h}
        dprev_h = Tools.matmul(dz, Wh.T)
        # \prod_{t=k+1}^T \grad{h^{t}}{h^{t-1}} * \grad{h^k}{Wx}
        dWx = Tools.matmul(x.T, dz)
        # \prod_{t=k+1}^T \grad{h^{t}}{h^{t-1}} * \grad{h^k}{Wh}
        dWh = Tools.matmul(prev_h.T, dz)
        db = np.sum(dz, axis=0)
        return dx, dprev_h, dWx, dWh, db
    
    def rnn_backward(self, dh):
        """
        dh.shape = (N,T,H)
        """
        N,T,H = dh.shape
        x,_,_,_,_ = self.rnnParams[0]["cache"][0]
        D = x.shape[1]
        
        dh_prevL = dh
        dweights = []
        for layer in range(self.layersNum-1,-1,-1):
            cache = self.rnnParams[layer]["cache"]
            DH = D if layer==0 else H
            dx = np.zeros((N,T,DH))
            dWx = np.zeros((DH,H))
            db = np.zeros(H)
            dprev_h_t = np.zeros((N,H))
            
            for t in range(T-1,-1,-1):
                dx[:,t,:], dprev_h_t, dWx_t, dWh_t, db_t = self.rnn_step_backward(dh_prevL[:,t:,]+dprev_h_t,cache[t])
                dWx += dWx_t
                dWh += dWh_t
                db += db_t
                
            dh_prevL = dx # Same as the neural network
            dweight = (dWx, dWh, db)
            dweights.append(dweight)
            
        return dx, dweights
    
    def bpWeights(self, dw, lr):
        """
        Updating the weight using optimizer functions
        """
        L = self.layersNum
        for l in range(L):
            w = (self.rnnParams[l]["Wx"], self.rnnParams[l]["Wh"], self.rnnParams[l]["b"])
            w = self.optimizerObjs[l].getUpweights(w, dw[L-1-l], lr)

    def back_propagation(self, input, delta_ori, lr):
        
        if self.dropoutRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask
            
        N,T,D = input.shape 
        H = delta.shape[1]
        dh = np.zeros((N,T,H), self.dataType)
        dh = delta
        dx, dweight = self.rnn_backward(dh)
        
        self.bpWeights(dweight, lr)
        
        return dx
            
        
        
        
        
        
        
                
                
                
                
                
            
            
            
            
            