# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:42:13 2023

@author: reekm
"""
import torch

import torch.nn as nn
import torch.nn.functional as F

from QuantumCircuits import Quantum_circuit

from Constants import PENNY_VARIATIONAL_DEPTH, PENNY_IP_LAYER_FLAG,PENNY_FRONT_LAYER
from Constants import PENNY_OP_LAYER_FLAG,PENNY_LAST_LAYER, PENNY_MEASUREMENT_LAYER
from Constants import PENNY_ENTANGLEMENT_LAYER,PENNY_MIDDLE_LAYER,PENNY_COUNT_MID_LAY
from Constants import PENNY_FMAP_DEPTH,PENNY_FMAP_ID

import pennylane as qml
from pennylane import numpy as np
from Constants import PENNY_VARIATIONAL_DEPTH,PENNY_MIDDLE_LAYER

class DressedQuantumNet2(nn.Module):
    
    def __init__(self,qc_circuit_key,num_ftrs,n_qubits,n_classes,approach=1,device_id=0):
        
        super().__init__()
        self.qc_circuit_key = qc_circuit_key
        self.n_qubits = n_qubits
        self.approach = 1
        
        self.pre_net = nn.Linear(num_ftrs, self.n_qubits)
        
        weight_shapes={'weights':(self.qc_circuit_key[PENNY_VARIATIONAL_DEPTH] * len(self.qc_circuit_key[PENNY_MIDDLE_LAYER]) ,self.n_qubits)}
        
#        get quantum Device
        dev = self.get_quantum_device(device_id)
        
        
        qnode = qml.QNode(self.quantum_net,dev,interface="torch")
        self.qlayer = qml.qnn.TorchLayer(qnode,weight_shapes)
        self.post_net = nn.Linear(self.n_qubits, n_classes)
#        self.key = key
        
    def forward(self, input_features):

        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        q_out = torch.Tensor(0, self.n_qubits)
#        q_out = q_out.to(device)

        for elem in q_in:
            q_out_elem = self.qlayer(elem).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        output = self.post_net(q_out)

        return F.softmax(output,dim=1)
        
    def get_quantum_device(self,device_id):
        if device_id == 0:
            return qml.device('default.qubit', wires=self.n_qubits)
        
    def quantum_net(self,inputs, weights):
        qc = Quantum_circuit(self.qc_circuit_key,self.n_qubits )
        
        if self.approach == 1:
            if self.qc_circuit_key[PENNY_IP_LAYER_FLAG] ==1:
                qc.front_layers(inputs)
       
            for k in range(int(self.qc_circuit_key[PENNY_VARIATIONAL_DEPTH])):
                qc.get_var_layer2(k,weights)
                
            qc.get_entanglement_layer(self.qc_circuit_key[PENNY_ENTANGLEMENT_LAYER],True)
            
            if self.qc_circuit_key[PENNY_OP_LAYER_FLAG]  == 1:
                qc.last_layers(inputs)
                
        elif self.approach == 2:
            qc.get_feature_map(inputs,self.n_qubits)
        elif self.approach == 3:
            qc.get_layer('7',inputs)
            qc.get_feature_map(inputs,self.n_qubits)
                
        exp_vals = qc.get_expectation_value(int(self.qc_circuit_key[PENNY_MEASUREMENT_LAYER]))
        return tuple(exp_vals)
        