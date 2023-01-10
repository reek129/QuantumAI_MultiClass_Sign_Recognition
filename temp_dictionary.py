# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 13:02:59 2023

@author: reekm
"""
from Constants import PENNY_MIDDLE_LAYER

d = {"min_resource":1, "max_resource":"auto", "reduction_factor":3}
print(d)

#a = input("ENTER NUMBER OF SINGLE QUBIT GATE YOU NEED IN VQC (SHOULD BE GREATER THAN 0 AND LESS THAN 4)")
#print(a)

a = -1
while a<1 or a>3:
    try:
        a = int (input("ENTER NUMBER OF SINGLE QUBIT GATE YOU NEED IN VQC (SHOULD BE GREATER THAN 0 AND LESS THAN 4)"))
    except ValueError:
        a = 1
    print(a)
    
    
for b in range(1,a+1):
    print(f"b is {PENNY_MIDDLE_LAYER + str(b)}")
    
    