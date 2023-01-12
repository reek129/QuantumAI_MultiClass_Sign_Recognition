# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 13:02:59 2023

@author: reekm
"""
from Constants import PENNY_MIDDLE_LAYER

#d = {"min_resource":1, "max_resource":"auto", "reduction_factor":3}
#d=str("C:\Users\reekm\Documents\september2\Template6_MC_factoryDP\QAI_LAB_CEL_approach_1_hybrid_alexnet_BC_2022\2023_01_10_time_06_34_12").replace("\","/")
#print(d)
#print(min(6/2,3))
##a = input("ENTER NUMBER OF SINGLE QUBIT GATE YOU NEED IN VQC (SHOULD BE GREATER THAN 0 AND LESS THAN 4)")
##print(a)
#
#a = -1
#while a<1 or a>3:
#    try:
#        a = int (input("ENTER NUMBER OF SINGLE QUBIT GATE YOU NEED IN VQC (SHOULD BE GREATER THAN 0 AND LESS THAN 4)"))
#    except ValueError:
#        a = 1
#    print(a)
#    
#    
#for b in range(1,a+1):
#    print(f"b is {PENNY_MIDDLE_LAYER + str(b)}")
#    
#    

import pandas as pd
import collections
#
#file_name = pd.read_csv("Best_models_for_attack.csv")
#print(file_name)
#
#for index,best in file_name.iterrows():
##    print(type(best))
#    best_dict = best.to_dict()
##    print(sorted(best_dict.items()))
#    k = [key for key in best_dict if key.startswith('params_'+PENNY_MIDDLE_LAYER)]
#    print(k)
#    print(type(best_dict))
#    v = [best_dict[key] for key in best_dict.keys() and key in k]
#    print(v)
#    
#    
#    
#a = ["5","1","0"]
#print("".join(a))
#
#test_dict = {
#        'params_middle_layer_1':4,
#        'params_middle_layer_3':2,
#        'params_middle_layer_2':0,
#        'z':7,
#        'x':2
#        
#        }
#
#test = ['params_middle_layer_1','params_middle_layer_3','params_middle_layer_2']
#
#k = [key for key in test_dict if key.startswith('params_'+PENNY_MIDDLE_LAYER)]
#print(k)
#
#print(sorted(k))
#
#v = [str(test_dict[kk]) for kk in sorted(k) ]
#print(v)
#val = "".join(v)
#print(val)
#print(type(val))
#



#    print(best.to_dict())
import torch
use_cuda = torch.cuda.is_available()
print(use_cuda)

device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)


import pickle
middle_layer_count = 4
f = open('store.pckl', 'wb')
pickle.dump(middle_layer_count, f)
f.close()

f = open('store.pckl', 'rb')
obj = pickle.load(f)
print(obj)
f.close()
