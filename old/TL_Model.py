# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 10:39:40 2023

@author: reekm
"""


import torchvision
from Model import Model

class TLModel(Model):
    
    model =None
    num_ftrs = 0
    def __init__(self, model_name = "resnet18"):
        
        self.get_torch_model(model_name)
            
        self.freeze_initial_layers()
            
    def get_torch_model(self,model_name):
        
        if model_name == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
            self.num_ftrs = self.model.fc.in_features
        elif model_name == "alexnet":
            self.model = torchvision.models.alexnet(pretrained=True)
            self.num_ftrs =  self.model.classifier[1].in_features
        elif model_name == "vgg16":
            self.model = torchvision.models.vgg16(pretrained=True)
            self.num_ftrs = self.model.classifier[0].in_features
        elif model_name == "inception_v3":
            self.model = torchvision.models.inception_v3(pretrained=True)
            self.num_ftrs = self.model.fc.in_features
        
        
    def freeze_initial_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
    def get_model_num_ftrs(self):
        return self.num_ftrs
    
#    overiding abstract method
    def get_model(self):
        return self.model