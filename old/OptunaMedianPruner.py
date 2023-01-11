# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 14:57:11 2023

@author: reekm
"""

import optuna.pruners as optuna_pruners

class OptunaMedianPrunner:
#    def __init__(self, n_startup_trials=3, n_warmup_steps=5, interval_steps=3):
#        self.n_startup_trials = n_startup_trials
#        self.n_warmup_steps = n_warmup_steps
#        self.interval_steps = interval_steps
        
    def set_n_startup_trials(self,n_startup_trials):
        self.n_startup_trials = n_startup_trials
        
    def set_n_warmup_steps(self,n_warmup_steps):
        self.n_warmup_steps = n_warmup_steps
        
    def set_interval_steps(self,interval_steps):
        self.interval_steps = interval_steps
        
    def get_median_pruner_obj(self):
        pruner=optuna_pruners.MedianPruner(
        n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_warmup_steps, interval_steps=self.interval_steps
    )
        return pruner
        
      