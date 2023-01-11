# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 22:01:51 2023

@author: reekm
"""


from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def get_model(self):
        pass
    