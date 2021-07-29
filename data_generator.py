import numpy as np 
import logging.config
import random, time
import matplotlib.pyplot as plt 

import sys
import os 

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)

from SequentialNeuralNet.utils import *
from SequentialNeuralNet.rnn import *
from SequentialNeuralNet.optimizers import *
from SequentialNeuralNet.activator import *
from SequentialNeuralNet.Structure import *

# exec_abs = os.getcwd()
# log_conf = exec_abs + "/config/logging.conf"
# logging.config.fileConfig(log_conf)
# logger = logging.getLogger("main")



class SeqData(object):
    
    def __init__(self, dataType):
        self.dataType = dataType
        self.x, self.y, self.x_v, self.y_v = self.initData()
        self.sample_range = [i for i in range(len(self.y))] # The index for training data
        self.sample_range_v = [i for i in range(len(self.y_v))] # The index for testing data
        
    def initData(self):
        test_start = (Params.TRAINING_EXAMPLES + Params.TIMESTEPS) * Params.SAMPLE_GAP
        test_end = test_start + (Params.TESTING_EXAMPLES + Params.TIMESTEPS) * Params.SAMPLE_GAP

        train_X, train_y = self.generate_data(self.curve(np.linspace(
            0, test_start, Params.TRAINING_EXAMPLES+Params.TIMESTEPS, dtype=self.dataType
        )))
        test_X, test_y = self.generate_data(self.curve(np.linspace(
            test_start, test_end, Params.TESTING_EXAMPLES+Params.TIMESTEPS, dtype=self.dataType
        )))
        
        return train_X, train_y, test_X, test_y
    
        
    def generate_data(self, seq):
        X = []
        y = []
        for i in range(len(seq) - Params.TIMESTEPS - Params.PRED_STEPS):
            X.append([seq[i:i+Params.TIMESTEPS]])
            y.append([seq[i+Params.TIMESTEPS:i+Params.TIMESTEPS+Params.PRED_STEPS]])
        return np.swapaxes(np.array(X, dtype=self.dataType), 1, 2), np.swapaxes(np.array(y, dtype=self.dataType), 1, 2)
    
    def curve(self, x):
        return np.sin(np.pi*x / 3.) + np.cos(np.pi*x/3.) + np.sin(np.pi*x / 1.5) + np.random.uniform(-0.05, 0.05, len(x))
    
    def getTrainRanges(self, miniBatchesSize):
        rangeAll = self.sample_range
        random.shuffle(rangeAll)
        rngs = [rangeAll[i:i+miniBatchesSize] for i in range(0, len(rangeAll), miniBatchesSize)]
        return rngs
    
    def getTrainDataByRng(self, rng):
        xs = np.array([self.x[sample] for sample in rng], self.dataType)
        values = np.array([self.y[sample] for sample in rng])
        return xs, rng
    
    def getValData(self, valCapacity):
        samples_v = [i for i in range(valCapacity)]
        x_v = np.array([self.x_v[sample_v] for sample_v in samples_v], dtype=self.dataType)
        y_v = np.array([self.y_v[sample_v] for sample_v in samples_v], dtype=self.dataType)
        
        return x_v, y_v
    
                
            
    