import numpy as np 
import numba

class Tools:
    
    @numba.jit
    def matmul(a,b):
        return np.matmul(a,b)
    
    @staticmethod
    def softmax(y):
        max_y = np.max(y, axis=1)
        max_y.shape = (-1, 1)
        y1 = y - max_y
        exp_y = np.exp(y1)
        sigma_y = np.sum(exp_y, axis=1)
        sigma_y.shape = (-1, 1)
        softmax_y = exp_y / sigma_y
        return softmax_y
    
    @staticmethod
    def crossEntropy(y, y_):
        n = len(y)
        return np.sum(-np.log(np.clip(y[range(n), y_], 1e-10, None, None))) / n
    
    @staticmethod
    def crossEntropylogit(y, y_):
        return -np.log(np.clip(y[range(n), y_], 1e-10, None, None)) / n
    
    @staticmethod
    def mse(y, y_):
        return np.mean((y-y_)**2, axis=-1) / 2
    
    @staticmethod
    def sigmoid(x):
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1+z)
        
    @staticmethod
    def grad_sigmoid(y):
        return y * (1-y)
    
    @staticmethod
    def grad_tanh(y):
        return 1 - y**2
    
    def dropoutRNN(x, p):
        if p <= 0. or p > 1.:
            raise ValueError("Dropout reserve p must be in interval (0, 1]")
        
        if p == 1:
            return x,None
        
        mask = np.random.binomial(1, p, x[0].shape)
        hat_x = (x * mask) / p
        
        return hat_x, mask
    
    @staticmethod
    def initOrthogonal(shape, initRng, dType):
        reShape = (shape[0], np.prod(shape[1:]))
        x = np.random.uniform(-1 * initRng, initRng, reShape).astype(dType)
        u,_,vt = np.linalg.svd(x, full_matrices=False)
        w = u if u.shape == reShape else vt 
        w = w.reshape(shape)
        return w
    
    
            
    
    