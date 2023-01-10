# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 18:11:53 2023

@author: reekm
"""



import pennylane as qml
class FeatureMaps():
    
    def __init__(self,n_qubits):
        self.n_qubits = n_qubits
        
        
    def feature_map1(self,x,feature_depth):
        for _ in range(feature_depth):
            for i in range(self.n_qubits):
                qml.RY(x[i],wires= i)
                qml.RZ(x[i], wires=i)
            for i in range(self.n_qubits - 1, 0, -1):
                qml.CNOT(wires=[i, i-1])
            qml.RY(x[1], wires=1) 
            qml.RZ(x[1], wires=1)
            
    def feature_map2(self,x,feature_depth):
        for _ in range(feature_depth):
            for i in range(self.n_qubits):
                qml.RX(x[i],wires= i)
                qml.RZ(x[i], wires=i)
            
            for control in range(self.n_qubits-1, 0, -1):
                target = control - 1
                qml.RX(x[target],wires= target)
                qml.CNOT(wires=[control, target])
                qml.RX(x[target],wires= target)
                
            for i in range(self.n_qubits):
                qml.RX(x[i],wires= i)
                qml.RZ(x[i],wires= i)
                
    
    def feature_map3(self,x,feature_depth): 
        
        for _ in range(feature_depth):
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])
                    qml.U1(x[i] * x[j],wires= j)
                    qml.CNOT(wires=[i, j])
            
    
    def feature_map4(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
            for i in range(self.num_qubits - 1):
                qml.CZ(wires=[i, i+1])
            for i in range(self.num_qubits):
                qml.RX(x[i],wires= i)
            for i in range(self.num_qubits-1, 0, -1):
                qml.CZ(wires=[i, i-1])
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
    
    
    def feature_map5(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RX(x[i],wires= i)
                qml.RZ(x[i],wires= i)
            for i in range(self.num_qubits-1, 0, -1):
                qml.CNOT(wires=[i, i-1])
    
    def feature_map6(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RX(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            for i in range(self.num_qubits-1, 0, -1):
                qml.CZ(wires=[i, i-1])
    
    def feature_map7(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RX(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            for i in range(self.num_qubits-1, 0, -1):
                qml.RZ(x[i-1], wires=i-1)
                qml.CNOT(wires=[i, i-1])
                qml.RZ(x[i-1],wires= i-1)
    
    def feature_map8(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RX(x[i],wires= i)
                qml.RZ(x[i],wires= i)
            for i in range(self.num_qubits-1, 0, -1):
                qml.RX(x[i-1],wires= i-1)
                qml.CNOT(wires=[i, i-1])
                qml.RX(x[i-1],wires= i-1)
    
    def feature_map9(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RX(x[i],wires= i)
                qml.RZ(x[i],wires= i)
            for control in range(self.num_qubits-1, -1, -1):
                for target in range(self.num_qubits-1, -1, -1):
                    if control != target:
                        qml.RZ(x[target],wires= target)
                        qml.CNOT(wires=[control, target])
                        qml.RZ(x[target], wires=target)
        
    
    def feature_map10(self,x,reps): 
        
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RX(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            for control in range(self.num_qubits-1, -1, -1):
                for target in range(self.num_qubits-1, -1, -1):
                    if control != target:
                        qml.RX(x[target],wires= target)
                        qml.CNOT(wires=[control, target])
                        qml.RX(x[target], wires=target)
    
    def feature_map11(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RX(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            for control in range(self.num_qubits-1, -1, -1):
                for target in range(self.num_qubits-1, -1, -1):
                    if control != target:
                        qml.RX(x[target], wires=target)
                        qml.CNOT(wires=[control, target])
                        qml.RX(x[target], wires=target)
    
    def feature_map12(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
            for i in range(self.num_qubits - 1):
                qml.CZ(wires=[i, i+1])
            for i in range(self.num_qubits):
                qml.RX(x[i], wires=i)
    
    def feature_map13(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RY(x[i],wires= i)
            for i in range(self.num_qubits - 1, 0, -1):
                qml.CZ(wires=[i, i-1])
            qml.CZ(wires=[self.num_qubits-1, 0])
            for i in range(self.num_qubits):
                qml.RY(x[i], wires=i)
    
    def feature_map14(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RY(x[i],wires= i)
                qml.RZ(x[i], wires=i)
            for i in range(self.num_qubits - 1, 0, -1):
                qml.CNOT(wires=[i, i-1])
               
            
    def feature_map15(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RY(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            for i in range(self.num_qubits - 1, 0, -1):
                qml.CZ(wires=[i, i-1])
            qml.RY(x[1],wires= 1)
            qml.RZ(x[1], wires=1)
    
    def feature_map16(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.num_qubits):
                qml.RX(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            qml.RX(x[0], wires=0)
            qml.CNOT(wires=[self.num_qubits - 1, 0])
            qml.RX(x[0], wires=0)
            for i in range(self.num_qubits-2, -1, -1):
                qml.RX(x[i+1],wires= i+1)
                qml.CNOT(wires=[i, i+1])
                qml.RX(x[i+1],wires= i+1)
    
    
    def feature_map17(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            qml.RZ(x[0], wires=0)
            qml.CNOT(wires=[self.n_qubits - 1, 0])
            qml.RZ(x[0], wires=0)
            for i in range(self.n_qubits-2, -1, -1):
                qml.RZ(x[i+1],wires= i+1)
                qml.CNOT(wires=[i, i+1])
                qml.RZ(x[i+1],wires= i+1)
    
    
    
    def feature_map18(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            for control in range(self.n_qubits - 1, 0, -1):
                target = control - 1
                qml.RX(x[target], wires=target)
                qml.CNOT(wires=[control, target])
                qml.RX(x[target],wires= target)
    
    
    
    def feature_map19(self,x,reps): 
        
        for _ in range(reps):
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            qml.CNOT(wires=[self.n_qubits-1, 0])
            for i in range(self.n_qubits-1):
                qml.CNOT(wires=[i, i+1])
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            qml.CNOT(wires=[self.n_qubits - 1, self.n_qubits - 2])
            qml.CNOT(wires=[0, self.n_qubits - 1])
            for i in range(1, self.n_qubits - 1):
                qml.CNOT(wires=[i, i-1])
                
    def feature_map20(self,x,reps): 
        
        for _ in range(reps):               # Exp 20
            for i in range(self.n_qubits):
                qml.RZ(x[i],wires= i)
                qml.RX(x[i], wires=i)
            for control in range(self.n_qubits-1, 0, -1):
                target = control - 1
                qml.RX(x[target],wires= target)
                qml.CNOT(wires=[control, target])
                qml.RX(x[target], wires=target)
            for i in range(self.n_qubits):
                qml.RZ(x[i],wires= i)
                qml.RX(x[i],wires= i)