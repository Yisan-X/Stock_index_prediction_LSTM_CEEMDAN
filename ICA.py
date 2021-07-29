import numpy as np 
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)


class ICA(object):
    
    def __init__(self, X):
        self.data = X
        self.N = X.shape[1]
        self.D = X.shape[0]
        self.isCentered = False
        self.isWhitened = False
        self.W = None
        self.Y = None
        
    def center(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        centered = X - mean
        return centered, mean
    
    def covariance(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        n = np.shape(X)[1] - 1
        m = X - mean
        
        return np.dot(m, m.T) / n
    
    def whiten(self, X):
        if self.isCentered == False:
            Xc = self.center(X)
            self.isCentered = True
        else:
            Xc = X
            
        coVarM = self.covariance(Xc)
        U,S,V = np.linalg.svd(coVarM)
        d = np.diag(1./np.sqrt(S))
        whiteM = np.dot(U, np.dot(d, U.T))
        Xw = np.dot(whiteM, X)
        return Xw, whiteM
    
    def factICA(self, signals, n_components, alpha=1, threshold=1e-8, iterations=5000):
        m,n = signals.shape
        if n_components is None:
            n_components = m
        self.n_components = n_components
        W = np.random.rand(n_components,m)
        for c in range(n_components):
            w = W[c,:].copy().reshape(m,1)
            w = w / np.sqrt((w**2).sum()) # diagonalization
            
            i = 0
            lim = 100
            while ((lim > threshold) & (i < iterations)):
                ws = np.dot(w.T, signals)
                wg = np.tanh(ws*alpha).T 
                wg_ = (1 - np.square(np.tanh(ws))) * alpha # gradient of function g
                wNew = (signals*wg.T).mean(axis=1) - wg_.mean()*w.squeeze()
                wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / np.sqrt((wNew**2).sum())
                lim = np.abs(np.abs((wNew * w).sum())-1)
                w = wNew
                i += 1
            
            W[c,:] = w.T 
        
        self.W = W
        
        return W
    
    def fit(self, n_components=None, alpha=1, threshold=1e-8, iterations=5000):
        if not self.isCentered:
            Xc, meanX = self.center(self.data)
            self.isCentered = True
        else:
            Xc = self.data
            
        if not self.isWhitened:
            Xw, whiteM = self.whiten(self.data)
            self.isWhitened = True
        else:
            Xw = self.data
        
        
        W = self.factICA(Xw, n_components, alpha, threshold, iterations)
        unMixed = Xw.T.dot(W.T)
        # unMixed = (unMixed.T - meanX).T 
        self.Y = unMixed
        return unMixed, W
    
    def IC_ordering(self, W=None, Y=None):
        if W is None:
            A = self.W
        if Y is None:
            X_L = self.Y
        if A is None:
            raise Exception("The un-mixed matrix is None")
        if X_L is None:
            raise Exception("The un-mixed signals is None")
        
        assert self.n_components == A.shape[1]
        
        N,D = X_L.shape
        
        name = ["IC%d"%(i+1) for i in range(D)]
        mixMatrix = np.linalg.pinv(A.T)
        num_IC = list(range(D))
        J = list()
        
        for i in range(D):
            Q = 0
            selected = num_IC.copy()
            selected.remove(i)
            org_selected = self.data[selected,:]
            Y_selected = X_L[:,selected]
            A_selected = mixMatrix[selected,:]
            re_signals = Y_selected.dot(A_selected)
            n = Y_selected.shape[1]
            for i in range(n):
                Q += self._RHD(org_selected[i,:], Y_selected[:,i])
            J.append(Q)
        
        result = dict(zip(name, J))
        return result
        
    @staticmethod
    def _RHD(x1, x2):
        assert x1.shape == x2.shape
        N = x1.shape[0]
        x1_ = np.diff(x1)
        x2_ = np.diff(x2)
        sgn = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        R1 = np.array(list(map(sgn, x1_)))
        R2 = np.array(list(map(sgn, x2_)))
        Q_i = np.sum(np.square(R1 - R2)) / (N-1)
        return Q_i
    
        
    
     
        
        