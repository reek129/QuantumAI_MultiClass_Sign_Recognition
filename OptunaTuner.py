# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 12:20:39 2023

@author: reekm
"""

from OptunaTunerInterface import optunaTunerInterface
from OptunaSamplers import optunaSampler
from OptunaPruners import optunaPruners
import optuna

import pandas as pd

from Constants import TPE_SAMPLER,HYPER_BAND_PRUNER
from Constants import MODEL_NAME,HYP_LR,HYP_OPTIMIZER_NAME,HYP_GAMMA_SCHEDULAR
from Constants import HYP_STEP,HYP_NEURON,HYP_BATCH
from Constants import CLASSICAL_SCENARIO, HYBRID_SCENARIO 
from Constants import PENNY_VARIATIONAL_DEPTH, PENNY_IP_LAYER_FLAG,PENNY_FRONT_LAYER
from Constants import PENNY_OP_LAYER_FLAG,PENNY_LAST_LAYER, PENNY_MEASUREMENT_LAYER
from Constants import PENNY_ENTANGLEMENT_LAYER,PENNY_MIDDLE_LAYER,PENNY_COUNT_MID_LAY

from classicalModel import classicalModel
from PytorchOptunaHelper import pytorch_helper
from QuantumModel import QuantumModel

import torch
import pickle as pkl

class optunaTuner(optunaTunerInterface):
    
    def __init__(self,data_dir,approach,system_type,n_classes,parameters,pruner_parameters,path,direction=['maximize'],sampler_name=TPE_SAMPLER,pruner_name = HYPER_BAND_PRUNER,n_trials=25):
        self.param = parameters
        self.batch_size_ls = [2,4,8,16,32,64,128]
        self.results =pd.DataFrame(columns=['ModelId','Best Loss','Trial','Test Accuracy','Description'])
        self.counter = 0
        
        self.sampler_name = sampler_name
        self.pruner_name = pruner_name
        self.pruner_parameters =pruner_parameters
        self.n_trials = n_trials
        self.direction = direction
        self.path = path
        self.n_classes = n_classes
        self.system_type = system_type
        self.approach = approach
        self.data_dir = data_dir
        
#        setting sampler and Prunner
        self.sampler()
        self.pruner()
        
#        creating optuna study
        self.create_study()
#        
##        optimizing optuna
        self.optimize()
        
        
        print("Going for Final Output results")
        
        self.saving_output()
#        create_vizualization
        self.create_vizualization()
        
        
    def saving_output(self):
        print(self.study.best_params)
        
        print("Best trial: ")
        print(self.study.best_trial)
        pkl.dump(self.study, open(self.path+"/"+self.system_type+".pkl", "wb"))
        
        df = self.study.trials_dataframe()
        df.to_csv(self.path+"/"+self.system_type+".csv",index=False)
        
        self.results.to_csv(self.path+"/Analysis_results_2.csv",index=False)
                
        
    def objective_single(self,trial):
        
        if self.system_type == CLASSICAL_SCENARIO:
            params = {
    
            MODEL_NAME: trial.suggest_categorical(MODEL_NAME,self.param[MODEL_NAME]),
            
            HYP_LR: trial.suggest_loguniform(HYP_LR,
                                             self.param[HYP_LR][0],
                                             self.param[HYP_LR][1]),
            HYP_OPTIMIZER_NAME: trial.suggest_categorical(HYP_OPTIMIZER_NAME,
                                                          self.param[HYP_OPTIMIZER_NAME]),
            HYP_GAMMA_SCHEDULAR:trial.suggest_loguniform(HYP_GAMMA_SCHEDULAR, 
                                                         self.param[HYP_GAMMA_SCHEDULAR][0],
                                                         self.param[HYP_GAMMA_SCHEDULAR][1]),
            HYP_STEP:trial.suggest_int(HYP_STEP,
                                       self.param[HYP_STEP][0],
                                       self.param[HYP_STEP][1]),
            HYP_NEURON:trial.suggest_int(HYP_NEURON,
                                         self.param[HYP_NEURON][0],
                                         self.param[HYP_NEURON][1]),
            HYP_BATCH : trial.suggest_int(HYP_BATCH,
                                          self.param[HYP_BATCH][0],
                                         self.param[HYP_BATCH][1])
    
                }
            
            self.model = classicalModel(params[MODEL_NAME],params[HYP_NEURON],self.n_classes).get_model()
            
        elif self.system_type == HYBRID_SCENARIO:
            params = {
    
            MODEL_NAME: trial.suggest_categorical(MODEL_NAME,self.param[MODEL_NAME]),
            
            HYP_LR: trial.suggest_loguniform(HYP_LR,
                                             self.param[HYP_LR][0],
                                             self.param[HYP_LR][1]),
            HYP_OPTIMIZER_NAME: trial.suggest_categorical(HYP_OPTIMIZER_NAME,
                                                          self.param[HYP_OPTIMIZER_NAME]),
            HYP_GAMMA_SCHEDULAR:trial.suggest_loguniform(HYP_GAMMA_SCHEDULAR, 
                                                         self.param[HYP_GAMMA_SCHEDULAR][0],
                                                         self.param[HYP_GAMMA_SCHEDULAR][1]),
            HYP_STEP:trial.suggest_int(HYP_STEP,
                                       self.param[HYP_STEP][0],
                                       self.param[HYP_STEP][1]),
            HYP_NEURON:trial.suggest_int(HYP_NEURON,
                                         self.param[HYP_NEURON][0],
                                         self.param[HYP_NEURON][1]),
            HYP_BATCH : trial.suggest_int(HYP_BATCH,
                                          self.param[HYP_BATCH][0],
                                         self.param[HYP_BATCH][1]),
            PENNY_VARIATIONAL_DEPTH : trial.suggest_int(PENNY_VARIATIONAL_DEPTH,
                                                        self.param[PENNY_VARIATIONAL_DEPTH][0],
                                                        self.param[PENNY_VARIATIONAL_DEPTH][1]),
            PENNY_IP_LAYER_FLAG : trial.suggest_int(PENNY_IP_LAYER_FLAG,
                                                    self.param[PENNY_IP_LAYER_FLAG][0],
                                                    self.param[PENNY_IP_LAYER_FLAG][1]),
            PENNY_FRONT_LAYER : trial.suggest_int(PENNY_FRONT_LAYER,
                                                  self.param[PENNY_FRONT_LAYER][0],
                                                  self.param[PENNY_FRONT_LAYER][1]),
            PENNY_OP_LAYER_FLAG : trial.suggest_int(PENNY_OP_LAYER_FLAG,
                                                    self.param[PENNY_OP_LAYER_FLAG][0],
                                                  self.param[PENNY_OP_LAYER_FLAG][1]),
            PENNY_LAST_LAYER : trial.suggest_int(PENNY_LAST_LAYER,
                                                 self.param[PENNY_LAST_LAYER][0],
                                                  self.param[PENNY_LAST_LAYER][1]),
            PENNY_ENTANGLEMENT_LAYER : trial.suggest_int(PENNY_ENTANGLEMENT_LAYER,
                                                         self.param[PENNY_ENTANGLEMENT_LAYER][0],
                                                         self.param[PENNY_ENTANGLEMENT_LAYER][1]),
            PENNY_MEASUREMENT_LAYER : trial.suggest_int(PENNY_MEASUREMENT_LAYER,
                                                         self.param[PENNY_MEASUREMENT_LAYER][0],
                                                         self.param[PENNY_MEASUREMENT_LAYER][1]),
    
                }
            
            print(f"printing before error : {self.param[PENNY_COUNT_MID_LAY]}")
#            key =  [params["ip"],str(params["front_layer"]),params["op"],str(params["last_layer"]),params["entanglement_layer"],params["measurement"],3,params["var_depth"],'111',100]
            try:
                temp_mid_layer = ""
                for counter in range(1,self.param[PENNY_COUNT_MID_LAY]+1):
                    middle_lay_key = PENNY_MIDDLE_LAYER+"_"+str (counter)
                    params[middle_lay_key] = trial.suggest_categorical(middle_lay_key,self.param[middle_lay_key])
                    temp_mid_layer = temp_mid_layer + params[middle_lay_key] 
    #                qc_circuit_key[middle_lay_key] = params[middle_lay_key] 
    #                key.append(middle_lay_key)
            except Exception as ex:
                print(ex)
            
            qc_circuit_key = {
                    PENNY_IP_LAYER_FLAG : params[PENNY_IP_LAYER_FLAG],
                    PENNY_FRONT_LAYER : str(params[PENNY_FRONT_LAYER]),
                    PENNY_OP_LAYER_FLAG : params[PENNY_OP_LAYER_FLAG],
                    PENNY_LAST_LAYER : str(params[PENNY_LAST_LAYER]),
                    PENNY_ENTANGLEMENT_LAYER: params[PENNY_ENTANGLEMENT_LAYER],
                    PENNY_MEASUREMENT_LAYER : params[PENNY_MEASUREMENT_LAYER],
                    PENNY_VARIATIONAL_DEPTH : params[PENNY_VARIATIONAL_DEPTH]
                    }
                        
            qc_circuit_key[PENNY_MIDDLE_LAYER] = temp_mid_layer
            
            
            
            print(f"Printing all hybrid parameters : {params}")
            print(f"Printing all Circuit parameters : {qc_circuit_key}")
            
            self.model = QuantumModel(qc_circuit_key,params["model_name"],params["n_qubits"],self.n_classes,self.approach).get_model()
#            key =  [params["ip"],str(params["front_layer"]),params["op"],str(params["last_layer"]),params["entanglement_layer"],params["middle_layer"],params["measurement"],3,params["var_depth"],'111',100]
    
#            self.model = classicalModel(params[MODEL_NAME],params[HYP_NEURON],self.n_classes).get_model()

            
    #    print("model is \n",model)
        print(self.model)
        p1 = pytorch_helper(self.model,self.batch_size_ls[params[HYP_BATCH]],params[MODEL_NAME],self.path,self.data_dir)
        
        criterion = p1.get_crossEntropy_criterion()
        print(f"Optimizer parameters are : {self.model,params[HYP_OPTIMIZER_NAME],params[HYP_LR]}")
        optimizer = p1.get_optima_optimizer(self.model,params[HYP_OPTIMIZER_NAME],params[HYP_LR])
        
        scheduler = p1.get_optima_Schedular(optimizer,params[HYP_STEP],params[HYP_GAMMA_SCHEDULAR])
        
        best_model, best_acc, best_loss = p1.train_model_optuna(trial, self.model, criterion,scheduler, optimizer, num_epochs=15)
        
        torch.save(best_model.state_dict(),self.path +"/"+ f"model_{self.system_type}_trial_{trial.number}.pth")
        
        row = {
           'ModelId':str(trial.number),
           'Best Loss':str(best_loss),
           'Trial':str(trial),
           'Test Accuracy':str(best_acc * 100),
           'Description': str(params)
           }
        self.results = self.results.append(row,ignore_index=True)
        self.results.to_csv(self.path+"/temp_Analysis"+self.system_type+"_results_2.csv",index=False)
    
    #    return best_loss
        return best_acc
#    def objective_multiple(self,trial):
        
    def sampler(self):
        op_sampler = optunaSampler(self.sampler_name)
        self.sampler = op_sampler.get_sampler()
        
    def pruner(self):
        op_pruner = optunaPruners(self.pruner_parameters,self.pruner_name)
        self.pruner = op_pruner.get_pruner()
        
    def create_study(self):
        self.study = optuna.create_study(
            sampler=self.sampler,
            pruner=self.pruner,
            directions=self.direction)
        
    def optimize(self):
        self.study.optimize(func=self.objective_single, n_trials=self.n_trials)
        
    def create_vizualization(self):
        optuna.visualization.plot_parallel_coordinate(self.study).write_image(self.path+"/plot_parallel_coordinate.png")
        #optuna.visualization.plot_contour(self.study, params=['optimizer_name','var_depth']).write_image(self.path+"/plot_contour.png")
        optuna.visualization.plot_slice(self.study).write_image(self.path+"/plot_slice.png")
        optuna.visualization.plot_param_importances(self.study).write_image(self.path+"/plot_param_importances.png")
        optuna.visualization.plot_optimization_history(self.study).write_image(self.path+"/plot_optimization_history.png")
        optuna.visualization.plot_intermediate_values(self.study).write_image(self.path+"/plot_intermediate_values.png")

