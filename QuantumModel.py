# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:38:23 2023

@author: reekm
"""


from TL_Model import TLModel
from Constants import RESNET_MODEL,VGG16_MODEL, ALEXNET_MODEL, INCEPTION_V3_MODEL 
from QuantumModule import DressedQuantumNet2

class QuantumModel(TLModel):
    def __init__(self,qc_circuit_key,model_name= RESNET_MODEL,n_qubits = 4,n_classes=2,approach=1):
#        self.trial=trial
        super().__init__(model_name)
        self.model = super().get_model()
        self.num_ftrs = super().get_model_num_ftrs()
        
        self.model_name = model_name
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.approach = approach
        
        self.qc_circuit_key = qc_circuit_key
        
#        self.quantum_model()
        self.update_TL_model()
        
    def update_TL_model(self):
        
        plugin_model = DressedQuantumNet2(self.qc_circuit_key,self.num_ftrs,self.n_qubits,self.n_classes,self.approach)
        
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