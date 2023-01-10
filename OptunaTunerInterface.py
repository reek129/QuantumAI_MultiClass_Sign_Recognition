# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 12:12:10 2023

@author: reekm
"""

from abc import ABC, abstractmethod

class optunaTunerInterface:
    @abstractmethod
    def objective_single(self,trial):
        pass
    @abstractmethod
    def sampler(self,sampler_name):
        pass
    @abstractmethod
    def pruner(self,pruner_name):
        pass
    @abstractmethod
    def create_study(self):
        pass
    @abstractmethod
    def create_vizualization(self):
        pass
    @abstractmethod
    def optimize(self):
        pass