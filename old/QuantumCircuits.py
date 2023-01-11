# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 18:23:29 2023

@author: reekm
"""

import pennylane as qml

from FeatureMaps import FeatureMaps
from Layers import Layers

from Constants import PENNY_VARIATIONAL_DEPTH,PENNY_FRONT_LAYER
from Constants import PENNY_LAST_LAYER
from Constants import PENNY_ENTANGLEMENT_LAYER,PENNY_MIDDLE_LAYER,PENNY_COUNT_MID_LAY
from Constants import PENNY_FMAP_DEPTH,PENNY_FMAP_ID


class Quantum_circuit(Layers):
    
#    ip = 1 
#    front_layer = '70'
#    op = 0
#    last_layer = '3'
#    entanglement_layer = 0
#    middle_layer= '02'
#    measurement = 2
#    fmap_depth = 3
#    var_depth = 3
#    model_id = 'default'
#    featureMap_id = 100
#    n_qubits =4
    
    def __init__(self,qc_circuit_key,n_qubits):
        super().__init__(n_qubits)
        self.qc_circuit_key = qc_circuit_key
        self.n_qubits = n_qubits
        self.feature_map = FeatureMaps(n_qubits)
        
    def get_layer(self,layerId,weights):
        if layerId == '0':
            super().RY_layer(weights)
        elif layerId =='1':
            super().U1_layer(weights)
        elif layerId =='2':
            super().RZ_layer(weights)
        elif layerId =='3':
            super().RX_layer(weights)
        elif layerId =='4':
            super().U2_layer(weights)
        elif layerId =='5':
            super().U3_layer(weights)
        elif layerId =='6':
            super().PhaseShift_layer(weights)
        elif layerId =='7':
            super().H_layer(self.n_qubits)
            
    def get_entanglement_layer(self,ent_id,flag=False):
        if ent_id == 0:
            super().entangling_layer_CNOT(self.n_qubits,flag)
        if ent_id ==1:
            super().entangling_layer_CZ(self.n_qubits,flag)
        if ent_id == 2:
            super().entangling_layer_CRX(self.n_qubits,flag)
            
    def front_layers(self,w):
        for x in self.qc_circuit_key[PENNY_FRONT_LAYER]:
            self.get_layer(x,w)
            
    def last_layers(self,w):
        for x in self.qc_circuit_key[PENNY_LAST_LAYER]:
            self.get_layer(x,w)
            
    def get_var_layer(self,weights):
        self.get_entanglement_layer(self.qc_circuit_key[PENNY_ENTANGLEMENT_LAYER])
        for x in self.qc_circuit_key[PENNY_MIDDLE_LAYER]:
            self.get_layer(x,weights)
            
            
    def get_var_layer2(self,k,weights):
        self.get_entanglement_layer(self.qc_circuit_key[PENNY_ENTANGLEMENT_LAYER])
        for i,j in enumerate(self.qc_circuit_key[PENNY_MIDDLE_LAYER]):
            self.get_layer(j,weights[(self.qc_circuit_key[PENNY_VARIATIONAL_DEPTH]*i)+k])
            
    def get_expectation_value(self,measurementId):
        gate_set = [qml.PauliX, qml.PauliY, qml.PauliZ]
        exp_val = [qml.expval(gate_set[measurementId](position)) for position in range(self.n_qubits)]
        return exp_val
    
    
    def get_feature_map(self,x,qubits):
#        featureMap_id = self.featureMap_id
        featureMap_id = self.qc_circuit_key[PENNY_FMAP_ID]
        reps = self.qc_circuit_key[PENNY_FMAP_DEPTH]
        
        if featureMap_id == 1:
            self.feature_map.feature_map1(x,qubits,reps)
        elif featureMap_id == 2:
            self.feature_map.feature_map2(x,qubits,reps)
        elif featureMap_id == 3:
            self.feature_map.feature_map3(x,qubits,reps)
        elif featureMap_id == 4:
            self.feature_map.feature_map4(x,qubits,reps)
        elif featureMap_id == 5:
            self.feature_map.feature_map5(x,qubits,reps)
        elif featureMap_id == 6:
            self.feature_map.feature_map6(x,qubits,reps)
        elif featureMap_id == 7:
            self.feature_map.feature_map7(x,qubits,reps)
        elif featureMap_id == 8:
            self.feature_map.feature_map8(x,qubits,reps)
        elif featureMap_id == 9:
            self.feature_map.feature_map9(x,qubits,reps)
        elif featureMap_id == 10:
            self.feature_map.feature_map10(x,qubits,reps)
        elif featureMap_id == 11:
            self.feature_map.feature_map11(x,qubits,reps)
        elif featureMap_id == 12:
            self.feature_map.feature_map12(x,qubits,reps)
        elif featureMap_id == 13:
            self.feature_map.feature_map13(x,qubits,reps)
        elif featureMap_id == 14:
            self.feature_map.feature_map14(x,qubits,reps)
        elif featureMap_id == 15:
            self.feature_map.feature_map15(x,qubits,reps)
        elif featureMap_id == 16:
            self.feature_map.feature_map16(x,qubits,reps)
        elif featureMap_id == 17:
            self.feature_map.feature_map17(x,qubits,reps)
        elif featureMap_id == 18:
            self.feature_map.feature_map18(x,qubits,reps)
        elif featureMap_id == 19:
            self.feature_map.feature_map19(x,qubits,reps)
        elif featureMap_id == 20:
            self.feature_map.feature_map20(x,qubits,reps)

    
    
            
            
            
            
        