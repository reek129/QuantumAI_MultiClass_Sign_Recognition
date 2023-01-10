# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 22:10:15 2023

@author: reekm
"""


import torch.nn as nn
import torch.nn.functional as F

class ClassicalTransferLearningModel(nn.Module):
    def __init__(self,num_of_features,n_qubits,n_classes):
        super().__init__()
        # print("num of features",num_of_features)
        
        self.fc1 = nn.Linear(num_of_features,n_qubits)
        self.tanh = nn.Tanh()
        # self.relu1 = nn.ReLU6()
        self.fc2 = nn.Linear(n_qubits,n_classes)
        
    def forward(self,x):
        # x = self.tanh(x)
        
        x = self.fc1(x)
        x = self.tanh(x)
        # x = self.relu1(x)
        x = self.fc2(x)
#        print(x)
#        print("softmax")
#        print(F.softmax(x,dim=1))
        x = F.softmax(x,dim=1)
#        return
        
        return x
