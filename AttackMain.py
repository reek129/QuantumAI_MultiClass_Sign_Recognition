# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 01:14:07 2023

@author: reekm
"""

import os
import pandas as pd
import torch

from classicalModel import classicalModel

from Constants import CONDI_COMPLETE
from PytorchOptunaHelper import pytorch_helper

from Attack import Attacks
#
#from Constants import TPE_SAMPLER,HYPER_BAND_PRUNER
#from Constants import MODEL_NAME,HYP_LR,HYP_OPTIMIZER_NAME,HYP_GAMMA_SCHEDULAR
#from Constants import HYP_STEP,HYP_NEURON,HYP_BATCH


class AttackMain:
    def __init__(self,path,system_type,n_classes,eps):
        self.path = path
        self.system_type = system_type
        self.n_classes = n_classes
        self.eps = eps
        
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
            
            model = classicalModel(model_name,n_qubits,self.n_classes).get_model()
            pth_filepath = os.path.join(self.path, fileName)
            model = self.load_model(model,pth_filepath)
            
            print(f"Model path : {pth_filepath}")
            
            p1 = pytorch_helper(model,int(batch_size),model_name,self.n_classes)
            test_acc = p1.test_model(model)
            if test_acc <= 0.8:
                continue
            
            train_acc = p1.test_model_with_train(model)
            
            attack = Attacks(model,self.eps)
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
                'Gradient Attack': p1.test_model_adversary(model,GA)[1],
                'Gradient Sign Attack': p1.test_model_adversary(model,GSA)[1],
#                'Sparse L1 Descent Attack': p1.test_model_adversary(model,sparse)[1],
#                'L2 PGD Attack':p1.test_model_adversary(model,l2pgd)[1],
#                'SPSA Attack': p1.test_model_adversary(model,spsa)[1],
                'Projected Gradient Descent(PGD) Attack': p1.test_model_adversary(model,pgd)[1],
    #            'FAB Attack': test_model_adversary(model,fab)[1],
#                'Description':description
                
                }
        print(row)
        
        self.attack_results = self.attack_results.append(row,ignore_index=True)
        self.attack_results.to_csv(self.path+'/'+self.system_type+'_temp_bestModel_results.csv',index=False)
        
    def get_final_result(self):
        self.attack_results.to_csv(self.path+'/'+self.system_type+'_final_bestModel_results.csv',index=False)
            
            
            
        
        