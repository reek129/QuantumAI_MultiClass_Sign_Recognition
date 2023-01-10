# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:03:38 2023

@author: reekm
"""

from Folder_Management import FolderManagement
from OptunaTuner import optunaTuner

from Constants import TPE_SAMPLER,HYPER_BAND_PRUNER
from AttackMain import AttackMain
#from Constants import MEDIAN_PRUNNER

from Constants import MODEL_NAME,HYP_LR,HYP_OPTIMIZER_NAME,HYP_GAMMA_SCHEDULAR
from Constants import HYP_STEP,HYP_NEURON,HYP_BATCH,CLASSICAL_SCENARIO, HYBRID_SCENARIO 
from Constants import PENNY_VARIATIONAL_DEPTH, PENNY_IP_LAYER_FLAG,PENNY_FRONT_LAYER
from Constants import PENNY_OP_LAYER_FLAG,PENNY_LAST_LAYER, PENNY_MEASUREMENT_LAYER
from Constants import PENNY_ENTANGLEMENT_LAYER,PENNY_MIDDLE_LAYER,PENNY_COUNT_MID_LAY
from Constants import PENNY_FMAP_DEPTH,PENNY_FMAP_ID

import pandas as pd

class Main_Classical:
    def __init__(self):
        
#        Binary dataset directory = "data/dataset1"  [Stop Sign, NOT a Stop Sign]
#        Multi dataset directory = "data/dataset2"   [stop,yield,do not enter, others ]
        self.data_dir = "data/dataset2"
        
#        type of problem Multiclass = MC, Binary = BC
        self.problem_type = "MC"
        
#        Option = [CLASSICAL_SCENARIO,HYBRID_SCENARIO]
        self.system_type = HYBRID_SCENARIO
        
#        Option [1:without_fMAP,2:with_fmap] []
#        Note: classical no impact, Hybrid Models Only
        self.approach = 1
        
#        Model name
#        Options = ["resnet18", "alexnet", "vgg16","inception_v3"]
        model_name = "alexnet"
        
#        Name of parent folder
        self.parent_folder_name = "QAI_LAB_CEL_"+str(self.approach)+"_"+self.system_type+"_"+model_name+"_"+self.problem_type+"_2022/"
        
#        number of classes in the problem
        self.n_classes = 4
        
#        create folder for saving result
        fm = FolderManagement(self.parent_folder_name)
        self.path = fm.get_model_path()
        
#        optuna parameters
        optuna_params = {}
        if self.system_type == CLASSICAL_SCENARIO:
            optuna_params[MODEL_NAME] = [model_name]
            optuna_params[HYP_LR] = [1e-4,1e-2]
            optuna_params[HYP_OPTIMIZER_NAME] = ["SGD","Adam"]
            optuna_params[HYP_GAMMA_SCHEDULAR] = [1e-4,1e-2]
            optuna_params[HYP_STEP] = [5,10]
            optuna_params[HYP_NEURON] = [2,8]
            optuna_params[HYP_BATCH] = [0,6]
        elif self.system_type == HYBRID_SCENARIO:
            
            middle_layer_count = self.get_middle_layer_count()
            max_var_depth = 6/middle_layer_count
#            Model Parameters
            optuna_params[MODEL_NAME] = [model_name]
            optuna_params[HYP_LR] = [1e-4,1e-2]
            optuna_params[HYP_OPTIMIZER_NAME] = ["SGD","Adam"]
            optuna_params[HYP_GAMMA_SCHEDULAR] = [1e-4,1e-2]
            optuna_params[HYP_STEP] = [5,10]
            optuna_params[HYP_NEURON] = [2,8]
            optuna_params[HYP_BATCH] = [0,6]
            
#            pennylane parameters
            optuna_params[PENNY_VARIATIONAL_DEPTH] = [1,max_var_depth]
            optuna_params[PENNY_IP_LAYER_FLAG] = [0,1]
            optuna_params[PENNY_FRONT_LAYER] = [0,7]
            optuna_params[PENNY_OP_LAYER_FLAG] = [0,1]
            optuna_params[PENNY_LAST_LAYER] = [0,7]
            optuna_params[PENNY_ENTANGLEMENT_LAYER] = [0,1]
            optuna_params[PENNY_MEASUREMENT_LAYER] = [0,2]
            optuna_params[PENNY_FMAP_DEPTH] = [0,3]
            optuna_params[PENNY_FMAP_ID] = [0,20]
            
            
            for counter in range(1,middle_layer_count+1):
                optuna_params[PENNY_MIDDLE_LAYER+"_"+str (counter)] = ["0","1","2","3","4","5","6"]
            
            optuna_params[PENNY_COUNT_MID_LAY] = middle_layer_count
#            "var_depth":trial.suggest_int("var_depth",1,6),
#         "ip":trial.suggest_int("ip",0,1),
#         "front_layer":trial.suggest_int("front_layer",0,7),
#         "op":trial.suggest_int("op",0,1),
#         "last_layer":trial.suggest_int("last_layer",0,7),
#        "entanglement_layer":trial.suggest_int("entanglement_layer",0,1),
#        "measurement":trial.suggest_int("measurement",0,2),
#         "middle_layer":trial.suggest_categorical("middle_layer",["0","1","2","3","4","5","6"])
        
#        Samplers 
#         option [TPE Sampler]
        sampler_name = TPE_SAMPLER
        
#        Prunner
#        Option = [HYPER_BAND_PRUNER,MEDIAN_PRUNNER]
        pruner_name = HYPER_BAND_PRUNER
        
#        Options for HYPER_BAND_PRUNER
#        min_resource=1, max_resource="auto", reduction_factor=3
        pruner_parameters = {
                "min_resource":1,
                "max_resource":"auto",
                "reduction_factor":3
                }
        
#        Options for MEDIAN_PRUNNER
#        n_startup_trials=3, n_warmup_steps=5, interval_steps=3
#        pruner_parameters = {
#                "n_startup_trials":3,
#                "n_warmup_steps":5,
#                "interval_steps":3
#                }
        
        self.n_trials = 2
        
        print("Before Entering Optuna Tuner: printing optuna_params")
        print(optuna_params)
        
        self.optuna_tuner = optunaTuner(
                data_dir=self.data_dir,
                approach = self.approach,
                system_type = self.system_type,
                n_classes = self.n_classes,
                parameters= optuna_params,
                pruner_parameters = pruner_parameters,
                path = self.path,
                direction=['maximize'],
                sampler_name=sampler_name,
                pruner_name = pruner_name,
                n_trials= self.n_trials
                )
        
        
#        Attack Code for classical
        eps = 1.0
        attack_main = AttackMain(self.path,self.system_type,self.n_classes,eps)
        attack_main.generate_attack_results()
        attack_main.get_final_result()
        
        
    def get_middle_layer_count(self):
        middle_layer_count = -1
        while middle_layer_count<1 or middle_layer_count>3:
            try:
                middle_layer_count = int (input("ENTER NUMBER OF SINGLE QUBIT GATE YOU NEED IN VQC (SHOULD BE GREATER THAN 0 AND LESS THAN 4)"))
            except ValueError:
                middle_layer_count = 1
                
        print(f"Middle layer Count: {middle_layer_count}")
        return middle_layer_count
        
        
Main_Classical()
        
        
        

        
        
        