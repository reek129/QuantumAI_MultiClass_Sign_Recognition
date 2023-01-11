# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 12:29:00 2023

@author: reekm
"""

import optuna.samplers as optuna_samplers
from Constants import TPE_SAMPLER

class optunaSampler:
    sampler=None
    def __init__(self,sampler_name = TPE_SAMPLER):
        self.match_samplers(sampler_name)
        
    def load_TPESampler(self):
        self.sampler = optuna_samplers.TPESampler()   
        
    def match_samplers(self,sampler_name):
        if sampler_name == TPE_SAMPLER:
            self.sampler = self.load_TPESampler()
            
    def get_sampler(self):
        return self.sampler
    