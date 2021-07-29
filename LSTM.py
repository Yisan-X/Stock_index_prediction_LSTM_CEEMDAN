import numpy as np 
import numba 

import sys
import os 
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

from utils import Tools
from activator import *

class LSTMLayer(object):
    
    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRate, dataType, init_reg):
        
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_reg = init_reg
        self.isInited = False
        self.dropoutRate = dropoutRate
        self.dropoutMask = []
        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        self.lstmParams = []
        self.deltaPrevs = [] # The error of the next layer
        
    def _initWeight(self, D, H, layersNum, dataType):
        lstmParams = []
        for layer in range(layersNum):
            # generate Whi, Whf, Whg, Who
            Wh = np.random.uniform(-1*self.init_reg, self.init_reg, (H, 4*H)).astype(dataType)
            # Wc = np.random.uniform(-1*self.init_reg, self.init_reg, (H, 4*H)).astype(dataType) # 带窥视孔的lstm
            if (layer == 0):
                # generate Wxi, Wxf, Wxg, Wxo
                Wx = np.random.uniform(-1*self.init_reg, self.init_reg, (D, 4*D)).astype(dataType)
            else:
                Wx = np.random.uniform(-1*self.init_reg, self.init_reg, (H, 4*H)).astype(dataType)
            b = np.zeros(4*H, dataType)
            
            lstmParams.append({"Wx": Wx, "Wh": Wh, "b": b})
        
        self.lstmParams = lstmParams
    
    def _initWeightOrthogonal(self, D, H, layersNum, dataType):
        
        lstmParams = []
        for layer in range(layersNum):
            Wh = Tools.initOrthogonal((H, 4*H), self.init_reg, dataType)
            DH = D if layer == 0 else H
            Wx = Tools.initOrthogonal((DH, 4*H), self.init_reg, dataType)
            b = np.zeros(4*H, dataType)
            lstmParams.append({"Wx":Wx, "Wh": Wh, "b": b})
            
        self.lstmParams = lstmParams
        
    def lstm_step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        H = prev_h.shape[1]
        z = Tools.matmul(x, Wx) + Tools.matmul(prev_h, Wh) + b # The new memory cell
        i = Tools.sigmoid(z[:,:H])
        f = Tools.sigmoid(z[:,H:2*H])
        o = Tools.sigmoid(z[:,2*H:3*H])
        g = np.tanh(z[:,3*H:])
        next_c = f * prev_c + i * g
        next_h = o * np.tanh(next_c)
        cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c)
        
        return next_h, next_c, cache
    
    def lstm_forward(self, x):
        """
        N: the number of samples
        T: the number of times
        D: The dimension of each nodes
        H: the number of input data
        
        The input data has dimension D, the hidden state has dimension H,
        the minibatch size is N
        """
        N,T,D = x.shape
        L = self.layersNum
        H = int(self.lstmParams[0]["b"].shape[0] / 4)
        xh = x # 首层输入
        for layer in range(L):
            h = np.zeros((N,T,H))
            h0 = np.zeros((N,H))
            c = np.zeros((N,T,H))
            c0 = np.zeros((N,H))
            cache = []
            for t in range(T):
                h[:,t,:], c[:,t,:], temp_cache = self.lstm_step_forward(xh[:,t,:], h[:,t-1,:] if t > 0 else h0,
                                                                        c[:,t-1,:,], self.lstmParams[layer]["Wx"], 
                                                                        self.lstmParams[layer]["Wh"], self.lstmParams[layer]["b"])
                cache.append(temp_cache)
            xh = h # 输出作为下一层的输入
            self.lstmParams[layer]["h"] = h 
            self.lstmParams[layer]["c"] = c
            self.lstmParams[layer]["cache"] = cache
            
        return h
        
    def inference(self, x):
        """forward for prediction"""
        N,T,D = x.shape
        H = self.nodesNum
        L = self.layersNum
        if (self.isInited == False):
            self._initWeightOrthogonal(D, H, L, self.dataType)
            self.isInited = True
        
        h = self.lstm_forward(x)
        self.out = h
        return self.out
    
    def forward_propagation(self, input):
        """forward after dropout"""
        out_temp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropoutRNN(out_temp, self.dropoutRate)
        return self.out     
     
    def lstm_step_backward(self, dnext_h, dnext_c, cache):
        x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c = cache
        dnext_c = dnext_c + dnext_h * o * (1 - np.tanh(next_c)**2) # \grad{J}{C_t} = \grad{J}{m_t} * \diff{m_t}{C_t} + \grad{J}{C_t} * f_{t+1}
        di = dnext_c * g
        df = dnext_c * prev_c
        do = dnext_h * np.tanh(next_c)
        dg = dnext_c * i
        dprev_c = dnext_c * f
        dz = np.hstack((i*(1-i)*di, f*(1-f)*df, o*(1-o)*do, (1-g**2)*dg))
        dx = Tools.matmul(dz, Wx.T)
        dprev_h = Tools.matmul(dz, Wh.T)
        dWx = Tools.matmul(x.T, dz)
        dWh = Tools.matmul(prev_h.T, dz)
        db = np.sum(dz, axis=0)
        
        return dx, dprev_h, dprev_c, dWx, dWh, db
    
    def lstm_backward(self, dh):
        N,T,H = dh.shape
        x,_,_,_,_,_,_,_,_,_ = self.lstmParams[0]["cache"][0] # 第一层第一个结点的cache
        D = x.shape[1]
        dh_prevl = dh
        dweights = []
        
        for layer in range(self.layersNum-1, -1, -1):
            cache = self.lstmParams[layer]["cache"]
            DH = D if layer == 0 else H 
            dx = np.zeros((N,T,DH))
            dWx = np.zeros((DH, 4*H))
            dWh = np.zeros((H, 4*H))
            db = np.zeros((4*H))
            dprev_h = np.zeros((N,H))
            dprev_c = np.zeros((N,H))
            for t in range(T-1, -1, -1):
                dx[:,t,:], dprev_h, dprev_c, dWx_t, dWh_t, db_t = self.lstm_step_backward(dh_prevl[:,t,:]+dprev_h, dprev_c, cache[t])
                dWx += dWx_t
                dWh += dWh_t
                db += db_t
                
            dh_prel = dx
            dweight = (dWx, dWh, db)
            dweights.append(dweight)
            
        return dx, dweights
    
    def bpWeights(self, dw, lr):
        L = self.layersNum
        for l in range(L):
            w = (self.lstmParams[l]["Wx"], self.lstmParams[l]["Wh"], self.lstmParams[l]["b"])
            self.lstmParams[l]["Wx"], self.lstmParams[l]["Wh"], self.lstmParams[l]["b"] = self.optimizerObjs[l].getUpweights(w, dw[L-1-l], lr)
    
    def back_propagation(self, input, delta_ori, lr):
        if self.dropoutRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask
        
        N,T,D = input.shape
        H = delta.shape[1]
        dh = np.zeros((N,T,H), self.dataType)
        dh = delta
        dx, dweight = self.lstm_backward(dh)
        self.bpWeights(dweight, lr)
        return dx
    
        
                        
            
            
        
        
        
        
        