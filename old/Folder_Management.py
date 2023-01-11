# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 12:01:02 2023

@author: reekm
"""


import time
import os

class FolderManagement:
    def __init__(self,parent):
        self.parent = parent
        self.make_directory_based_ontimestamp()
        
    def get_timestamp(self):
        # ts stores the time in seconds
#        ts = time.time()
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()).replace('-','_').replace(':','_').replace(' ','_time_')
        print(ts)
        return ts
    
    def make_directory_based_ontimestamp(self,folder_name=None):
        directory = self.get_timestamp()
        if folder_name is not None:
            directory=folder_name
        
        self.path = os.path.join(self.parent, directory) 
    #    print(os.path.exists(path))
    #    print(type(os.path.exists(path)))
        if (not os.path.exists(self.path)):
            os.makedirs(self.path)
            
    def get_model_path(self):
        return self.path