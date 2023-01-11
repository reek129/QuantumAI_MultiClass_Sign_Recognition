# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 01:14:07 2023

@author: reekm
"""

import os
import pandas as pd
import torch

from classicalModel import classicalModel

#from Constants import CONDI_COMPLETE,PENNY_MIDDLE_LAYER
#from Constants import CLASSICAL_SCENARIO, HYBRID_SCENARIO 

from Constants import MODEL_NAME,HYP_LR,HYP_OPTIMIZER_NAME,HYP_GAMMA_SCHEDULAR
from Constants import HYP_STEP,HYP_NEURON,HYP_BATCH,CLASSICAL_SCENARIO, HYBRID_SCENARIO 
from Constants import PENNY_VARIATIONAL_DEPTH, PENNY_IP_LAYER_FLAG,PENNY_FRONT_LAYER
from Constants import PENNY_OP_LAYER_FLAG,PENNY_LAST_LAYER, PENNY_MEASUREMENT_LAYER
from Constants import PENNY_ENTANGLEMENT_LAYER,PENNY_MIDDLE_LAYER,PENNY_COUNT_MID_LAY
from Constants import PENNY_FMAP_DEPTH,PENNY_FMAP_ID, CONDI_COMPLETE


from PytorchOptunaHelper import pytorch_helper
from QuantumModel import QuantumModel

from Attack import Attacks
#
#from Constants import TPE_SAMPLER,HYPER_BAND_PRUNER
#from Constants import MODEL_NAME,HYP_LR,HYP_OPTIMIZER_NAME,HYP_GAMMA_SCHEDULAR
#from Constants import HYP_STEP,HYP_NEURON,HYP_BATCH


class AttackMain:
    def __init__(self,path,data_dir,approach,system_type,n_classes,eps):
        self.path = path
        self.system_type = system_type
        self.n_classes = n_classes
        self.approach = approach
        self.eps = eps
        self.data_dir = data_dir
        
        columns = ['ModelId','Train Accuracy','Test Accuracy',
           'Gradient Attack','Gradient Sign Attack',
#           'Sparse L1 Descent Attack','L2 PGD Attack','SPSA Attack',
           'Projected Gradient Descent(PGD) Attack',
#           'FAB Attack',
#           'Description'
           ]
        self.attack_results = pd.DataFrame(columns=columns)
        
        self.file = system_type + ".csv"
        self.top = 25
        self.extract_top_models()
        
        
    def extract_top_models(self):
        csvFile = os.path.join(self.path, self.file)
        csv_df = pd.read_csv(csvFile)
        complete_df = csv_df[csv_df.state == CONDI_COMPLETE]
        
#        sort data based on best value and duration {this case is for best_accuracy}
        result_df = complete_df.sort_values(by=['value','duration'],ascending=[False,True])
        self.bests= result_df.head(self.top)
        self.bests.to_csv(self.path+"Best_models_for_attack.csv",index=False)
#        print(self.bests)
        
    def load_model(self,model,pth_filepath):
        
        model.load_state_dict(torch.load(pth_filepath))  
        model.eval()
#        print(model)
        return model
            
    def generate_attack_results(self):
        
        for index,best in self.bests.iterrows():
            
            trial_number= best.number
#            lr = best.params_lr
#            optimizer = best.params_optimizer_name
#            step_size = best.params_step_size
            model_name = best.params_model_name
            n_qubits = best.params_n_qubits
            batch_size = 2 ** (best.params_batch_size+1)
            
            fileName='model_'+self.system_type+'_trial_'+str(trial_number)+'.pth'
            
            print(fileName)
            print(batch_size)
            
            if self.system_type == CLASSICAL_SCENARIO:
                self.model = classicalModel(model_name,n_qubits,self.n_classes).get_model()
            elif self.system_type == HYBRID_SCENARIO:
                print(f"Best: {best}")
                best_dict = best.to_dict()
                val_key = [key for key in best_dict if key.startswith('params_'+PENNY_MIDDLE_LAYER)]
                mid_lay_val = "".join([str(best_dict[key]) for key in sorted(val_key) ])
                
                start_with = "params_"
                
                qc_circuit_key={
                    PENNY_IP_LAYER_FLAG : best_dict[start_with + PENNY_IP_LAYER_FLAG],
                    PENNY_FRONT_LAYER : str(best_dict[start_with + PENNY_FRONT_LAYER]),
                    PENNY_OP_LAYER_FLAG : best_dict[start_with + PENNY_OP_LAYER_FLAG],
                    PENNY_LAST_LAYER : str(best_dict[start_with + PENNY_LAST_LAYER]),
                    PENNY_ENTANGLEMENT_LAYER: best_dict[start_with + PENNY_ENTANGLEMENT_LAYER],
                    PENNY_MEASUREMENT_LAYER : best_dict[start_with + PENNY_MEASUREMENT_LAYER],
                    PENNY_VARIATIONAL_DEPTH : best_dict[start_with + PENNY_VARIATIONAL_DEPTH],
                    PENNY_MIDDLE_LAYER : mid_lay_val,
                    PENNY_FMAP_ID : best_dict[start_with + PENNY_FMAP_ID] ,
                    PENNY_FMAP_DEPTH : best_dict[start_with + PENNY_FMAP_DEPTH] ,
                    
                        }
                
#        
#        ip = best.params_ip
#        front_layer = str(best.params_front_layer)
#        op = best.params_ip
#        last_layer =  str(best.params_last_layer)
#        entanglement_layer = best.params_entanglement_layer
##        middle_layer ='02'
#        middle_layer = str(best.params_middle_layer)
#        measurement = best.params_measurement
#        fmap_depth = 3
#        var_depth = best.params_var_depth
#        model_id ="Hybrid"
#        fmap_id = 100
                self.model = QuantumModel(qc_circuit_key,model_name,n_qubits,self.n_classes,self.approach).get_model()
            
            pth_filepath = os.path.join(self.path, fileName)
            self.model = self.load_model(self.model,pth_filepath)
            
            print(f"Model path : {pth_filepath}")
            
            p1 = pytorch_helper(self.model,int(batch_size),model_name,self.path,self.data_dir)
            test_acc = p1.test_model(self.model)
#            if test_acc <= 0.8:
#                continue
            
            train_acc = p1.test_model_with_train(self.model)
            
            attack = Attacks(self.model,self.eps)
            GA = attack.GradientAttack()
            GSA = attack.GradientSignAttack()
#            sparse = attack.sparseL1DescentAttack()
#            l2pgd = attack.L2PGDAttack()
#            spsa =attack.LinfSPSAAttack()
            pgd = attack.PGD()
            
            row ={
               'ModelId':trial_number,
                'Train Accuracy':train_acc,
                'Test Accuracy':test_acc,
                'Gradient Attack': p1.test_model_adversary(self.model,GA)[1],
                'Gradient Sign Attack': p1.test_model_adversary(self.model,GSA)[1],
#                'Sparse L1 Descent Attack': p1.test_model_adversary(model,sparse)[1],
#                'L2 PGD Attack':p1.test_model_adversary(model,l2pgd)[1],
#                'SPSA Attack': p1.test_model_adversary(model,spsa)[1],
                'Projected Gradient Descent(PGD) Attack': p1.test_model_adversary(self.model,pgd)[1],
    #            'FAB Attack': test_model_adversary(model,fab)[1],
#                'Description':description
                
                }
            print(row)
            
            self.attack_results = self.attack_results.append(row,ignore_index=True)
            self.attack_results.to_csv(self.path+'/'+self.system_type+'_temp_bestModel_results.csv',index=False)
        
    def get_final_result(self):
        self.attack_results.to_csv(self.path+'/'+self.system_type+'_final_bestModel_results.csv',index=False)
            
            
            
        
        