# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 22:14:41 2023

@author: reekm
"""
CLASSICAL_SCENARIO = "classical"
HYBRID_SCENARIO = "hybrid"

TPE_SAMPLER = "TPESampler"

HYPER_BAND_PRUNER = "HyperBand"
MEDIAN_PRUNNER = "MedianBand"

#optunaParameterName
MODEL_NAME = "model_name"
HYP_LR = "lr"
HYP_OPTIMIZER_NAME = "optimizer_name"
HYP_GAMMA_SCHEDULAR = "gamma_lr_scheduler"
HYP_STEP = "step_size"
HYP_NEURON="n_qubits"
HYP_BATCH="batch_size" 

RESNET_MODEL = "resnet18"
VGG16_MODEL = "vgg16"
ALEXNET_MODEL = "alexnet"
INCEPTION_V3_MODEL = "inception_v3"

CONDI_COMPLETE = 'COMPLETE'

PENNY_VARIATIONAL_DEPTH = "var_depth"
PENNY_IP_LAYER_FLAG = "ip"
PENNY_FRONT_LAYER = "front_layer"
PENNY_OP_LAYER_FLAG = "op"
PENNY_LAST_LAYER = "last_layer"
PENNY_ENTANGLEMENT_LAYER = "entanglement_layer"
PENNY_MEASUREMENT_LAYER = "measurement"
PENNY_MIDDLE_LAYER = "middle_layer"
PENNY_COUNT_MID_LAY = "count_middle_layer"
PENNY_FMAP_DEPTH = "fmap_depth"
PENNY_FMAP_ID = "fmap_id"