'''
    author: Jiyao WANG
    e-mail: jiyaowang130@gmail.com
    release date:
'''

import os,sys,shutil
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Logging import Logging

def start(conf,data,model,evaluate):
    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # define log name
    log_path = os.path.join(os.getcwd(), 'log/%s_%s.log' % (conf.data_name, conf.model_name))

    #prepare data for training and evaluating
    data.initializeRankingHandle()