# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 12:50:23 2023

@author: reekm
"""


#import optuna.pruners as optuna_pruners
from OptunaHyperBandPruner import OptunaHyperBandPrunner
from OptunaMedianPruner import OptunaMedianPrunner
from Constants import HYPER_BAND_PRUNER,MEDIAN_PRUNNER

class optunaPruners:
    pruner=None
    def __init__(self,pruner_params,pruner_name = HYPER_BAND_PRUNER):
        self.pruner_name=pruner_name
        self.pruner_params = pruner_params  # need to be a dictionary
        self.load_pruner()
        
    def load_HyperBand_Pruner(self):
#        min_resource=1, max_resource="auto", reduction_factor=3  # default Values
        op_hyband_prunner = OptunaHyperBandPrunner()
        op_hyband_prunner.set_min_resource(self.pruner_params['min_resource'])
        op_hyband_prunner.set_max_resource(self.pruner_params['max_resource'])
        op_hyband_prunner.set_reduction_factor(self.pruner_params['reduction_factor'])
        
        self.pruner = op_hyband_prunner.get_hyperband_pruner_obj()
        
#        return pruner
    
    def load_Median_pruner(self):
#         n_startup_trials=3, n_warmup_steps=5, interval_steps=3
         
         op_median_prunner = OptunaMedianPrunner()
         op_median_prunner.set_n_startup_trials(self.pruner_params['n_startup_trials'])
         op_median_prunner.set_n_warmup_steps(self.pruner_params['n_warmup_steps'])
         op_median_prunner.set_interval_steps(self.pruner_params['interval_steps'])
         
         self.pruner = op_median_prunner.get_median_pruner_obj()
#         return pruner
     
    def load_pruner(self):
        if self.pruner_name == HYPER_BAND_PRUNER:
            self.load_HyperBand_Pruner()
        elif self.pruner_name == MEDIAN_PRUNNER:
            self.load_Median_pruner()
            
    def get_pruner(self):
        return self.pruner
            
    
         
         