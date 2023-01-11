# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 14:50:46 2023

@author: reekm
"""


import optuna.pruners as optuna_pruners

class OptunaHyperBandPrunner:
#    def __init__(self,min_resource=1, max_resource="auto", reduction_factor=3):
#        self.min_resource = min_resource
#        self.max_resource = max_resource
#        self.reduction_factor = reduction_factor
        
    def set_reduction_factor(self,reduction_factor):
        self.reduction_factor = reduction_factor
        
    def set_max_resource(self,max_resource):
        self.max_resource = max_resource
        
    def set_min_resource(self,min_resource):
        self.min_resource = min_resource
        
    def get_hyperband_pruner_obj(self):
        pruner=optuna_pruners.HyperbandPruner(
        min_resource=self.min_resource, max_resource=self.max_resource, reduction_factor=self.reduction_factor
    )
        return pruner
        
        
        