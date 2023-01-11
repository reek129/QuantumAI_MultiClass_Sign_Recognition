# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 18:25:55 2023

@author: reekm
"""

import pennylane as qml

class Layers():
    def __init__(self,n_qubits):
        self.n_qubits = n_qubits
        
    def H_layer(self,nqubits):
        for idx in range(nqubits):
            qml.Hadamard(wires=idx)
    def RY_layer(self,w):
        for idx, element in enumerate(w):
            qml.RY(element, wires=idx)
        
    def RX_layer(self,w):
        for idx, element in enumerate(w):
            qml.RX(element, wires=idx)
            
    def RZ_layer(self,w):
        for idx, element in enumerate(w):
            qml.RZ(element, wires=idx)
    
    def U1_layer(self,w):
        for idx, element in enumerate(w):
            qml.U1(element, wires=idx)
    
    
    def U2_layer(self,w):
        for idx, element in enumerate(w):
            qml.U2(element,element, wires=idx)     
            
    
    
    def U3_layer(self,w):
        for idx, element in enumerate(w):
            qml.U3(element,element,element, wires=idx)
    
    
    def PhaseShift_layer(self,w):
        for idx, element in enumerate(w):
            qml.PhaseShift(element, wires=idx)    
            
    def entangling_layer_CNOT(self,n_qubits,flag=True):
        for i in range(0, n_qubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, n_qubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            qml.CNOT(wires=[i, i + 1])
    
        if flag == True:
            qml.CNOT(wires=[n_qubits-1,0])
        else:
            qml.CNOT(wires=[0,n_qubits-1])
            
    def entangling_layer_CZ(self,n_qubits,flag=True):
        for i in range(0, n_qubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            qml.CZ(wires=[i, i + 1])
        for i in range(1, n_qubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            qml.CZ(wires=[i, i + 1])
    
        if flag == True:
            qml.CZ(wires=[n_qubits-1,0])
        else:
            qml.CZ(wires=[0,n_qubits-1])
    
    
    def entangling_layer_CRX(self,n_qubits,flag=True):
       
        # print("entangling")
        for i in range(0, n_qubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            qml.CRX(0.1,wires=[i, i + 1])
        for i in range(1, n_qubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            qml.CRX(0.1,wires=[i, i + 1])
    
        if flag == True:
            qml.CRX(0.1,wires=[n_qubits-1,0])
        else:
            qml.CRX(0.1,wires=[0,n_qubits-1])
