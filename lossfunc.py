import numpy as np 
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

from utils import Tools


class MSE:
    
    @staticmethod
    def loss(y, y_, n):
        correct_logprobs = Tools.mse(y, y_)
        data_loss = np.sum(correct_logprobs) / n
        delta = (y - y_) / n
        acc = data_loss
        
        return data_loss, delta, acc
    
    
class SoftmaxCrossEntropy:
    
    @staticmethod
    def loss(y, y_, n):
        y_argmax = np.argmax(y, axis=1)
        softmax_y = Tools.softmax(y)
        acc = np.mean(y_argmax == y_)
        correct_logprobs = Tools.crossEntropy(softmax_y, y_)
        data_loss = np.sum(correct_logprobs) / n 
        softmax_y[range(n), y_] -= 1
        delta = softmax_y / n
        
        return data_loss, delta, acc
    
class SigmoidCrossEntropy:
    
    @staticmethod
    def loss(y, y_, n):
        y_argmax = np.argmax(y, axis=1)
        sigmoid_y = Tools.sigmoid(y)
        acc = np.means(y_argmax == y_)
        correct_logprobs = Tools.crossEntropylogit(sigmoid_y, y_)
        data_loss = np.sum(correct_logprobs) / n
        sigmoid_y[range(n), y_] -= 1
        delta = sigmoid_y / n
        
        return data_loss, delta, acc