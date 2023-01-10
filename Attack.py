# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 01:13:25 2023

@author: reekm
"""


import torch.nn as nn

from advertorch.attacks import PGDAttack
from advertorch.attacks import L2PGDAttack
from advertorch.attacks import FABAttack
from advertorch.attacks import SparseL1DescentAttack
from advertorch.attacks import LinfSPSAAttack
from advertorch.attacks import GradientSignAttack
from advertorch.attacks import GradientAttack

class Attacks():
    model = ''
    loss_fn =nn.CrossEntropyLoss(reduction="sum")
    eps=0.05
    targeted =False
    def __init__(self,model,eps=0.05):
        self.model=model
#        self.loss_fn =loss_fn
        self.eps = eps
        
    def sparseL1DescentAttack(self,nb_iter=40,rand_init=False,eps_iter=0.01,l1_sparsity=0.95):
        return SparseL1DescentAttack(predict= self.model,loss_fn = self.loss_fn,
                               eps=self.eps,nb_iter=nb_iter,rand_init=rand_init,
                               eps_iter=eps_iter,l1_sparsity=l1_sparsity,
                          targeted=self.targeted)
        
    def GradientAttack(self):
        return GradientAttack(predict= self.model,loss_fn = self.loss_fn,
                               eps=self.eps,targeted=self.targeted)
        
    def GradientSignAttack(self):
        return GradientSignAttack(predict= self.model,loss_fn = self.loss_fn,
                               eps=self.eps,targeted=self.targeted)
        
    def L2PGDAttack(self,nb_iter=40,eps_iter=0.01):
        return L2PGDAttack(predict= self.model,loss_fn = self.loss_fn,
                               eps=self.eps,nb_iter=nb_iter,eps_iter=eps_iter,targeted=self.targeted)
        
    def LinfSPSAAttack(self,delta=0.01,lr=0.01,nb_iter=1,nb_sample=128,max_batch_size=64):
        return LinfSPSAAttack(predict= self.model,eps=self.eps,delta=delta,lr=lr,
                             nb_iter=nb_iter,nb_sample = nb_sample,max_batch_size=max_batch_size,
                             targeted=self.targeted,loss_fn = self.loss_fn)
        
    def PGD(self,nb_iter=40,eps_iter=0.01,clip_min=0.0,clip_max=1.0):
        return PGDAttack(predict = self.model,loss_fn = self.loss_fn,
                      eps=self.eps,nb_iter=nb_iter,eps_iter=eps_iter,clip_min=clip_min,clip_max=clip_max,
                      targeted =self.targeted)
        
    def FAB(self,n_iter=100,n_restarts=1,alpha_max=0.1, eta=1.05, beta=0.9,verbose=False,norm='Linf'):
        return FABAttack(predict= self.model,eps=self.eps,n_iter=n_iter,n_restarts=n_restarts,
                        loss_fn =  self.loss_fn,
                        alpha_max=alpha_max, eta=eta,
                        beta=beta,verbose=verbose,
                       norm=norm)
        
        
    