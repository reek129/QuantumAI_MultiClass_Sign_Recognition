# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 22:11:35 2023

@author: reekm
"""
import torchvision
from ClassicalTransferLearningModel import ClassicalTransferLearningModel

from TL_Model import TLModel
from Constants import RESNET_MODEL,VGG16_MODEL, ALEXNET_MODEL, INCEPTION_V3_MODEL 

class classicalModel(TLModel):
    
    
    def __init__(self, model_name = RESNET_MODEL,n_qubits= 4,n_classes = 2):
#        check super line if it throughs error
        super().__init__(model_name)
        self.model = super().get_model()
        self.num_ftrs = super().get_model_num_ftrs()
        self.n_qubits = n_qubits
        self.model_name = model_name
        self.n_classes = n_classes
        
        self.update_TL_model()
        
    def update_TL_model(self):
        
        plugin_model = ClassicalTransferLearningModel(self.num_ftrs,self.n_qubits,self.n_classes)
        
        if self.model_name == RESNET_MODEL:
            self.model.fc = plugin_model
        elif self.model_name == ALEXNET_MODEL:
            self.model.classifier = plugin_model
        elif self.model_name == VGG16_MODEL:
            self.model.classifier = plugin_model
        elif self.model_name == INCEPTION_V3_MODEL:
            self.model.fc = plugin_model
        
#    overridding inherited method    
    def get_model(self):
        return self.model

    
    
    
    