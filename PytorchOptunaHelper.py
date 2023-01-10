# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 00:59:00 2023

@author: reekm
"""


import torch
import os
import time
import optuna
import copy

import torch.nn as nn

from advertorch.utils import predict_from_logits
from torch.optim import lr_scheduler

from torchvision import datasets, transforms

from Constants import INCEPTION_V3_MODEL

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(42)


class pytorch_helper:
    def __init__(self,model,batch_size,model_name,path,data_dir):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.model = model
        self.model.to(self.device)
        self.batch_size = batch_size
        self.model_name = model_name
        self.path = path
        
        
    
    def get_crossEntropy_criterion(self):
        return nn.CrossEntropyLoss()
        
    def get_optima_optimizer(self,model,optimizer_name: str ="Adam",lr: int = 0.0004):
        optimizer = getattr(
            torch.optim, optimizer_name
        )(filter(lambda p:p.requires_grad, model.parameters()), lr)
        
        return optimizer
    
    def get_optima_Schedular(self,optimizer,step_size: int = 10,gamma_lr_schedular: float=0.1):
        scheduler = lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma_lr_schedular
        )
        return scheduler
        
    def dataset_transform(self):
        
        data_transforms = {
            "train": transforms.Compose(
                [
                    # transforms.RandomResizedCrop(224),     # uncomment for data augmentation
                    # transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # Normalize input channels using mean values and standard deviations of ImageNet.
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            }
        
        return data_transforms
    
    def dataset_transform_inception(self):
        
        data_transforms = {
            "train": transforms.Compose(
                [
                    # transforms.RandomResizedCrop(224),     # uncomment for data augmentation
                    # transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
                    transforms.Resize(256),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    # Normalize input channels using mean values and standard deviations of ImageNet.
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            }
        
        return data_transforms
    
    def dataset(self):
        image_datasets = None
        if self.model_name == INCEPTION_V3_MODEL:
            image_datasets = {
                x if x == "train" else "validation": datasets.ImageFolder(
                    os.path.join(self.data_dir, x), self.dataset_transform_inception()[x]
                )
                for x in ["train", "val"]
            }
        else:
            image_datasets = {
                x if x == "train" else "validation": datasets.ImageFolder(
                    os.path.join(self.data_dir, x), self.dataset_transform()[x]
                )
                for x in ["train", "val"]
            }
            
            
#        print(image_datasets)
        return image_datasets
    
    
    def dataset_sizes(self):
        dataset_size = {x: len(self.dataset()[x]) for x in ["train", "validation"]}
        return dataset_size
    
    def class_names(self):
        class_names = self.dataset()["train"].classes
        return class_names
    
    def dataset_dataloaders(self):
        dataloaders = {
            x: torch.utils.data.DataLoader(self.dataset()[x], batch_size=self.batch_size, shuffle=True)
            for x in ["train", "validation"]
        }
        
        return dataloaders
    
    
    
    
    def train_model_optuna(self, trial, model, criterion,scheduler, optimizer, num_epochs=20):
        since = time.time()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = 10000.0
        best_acc_train = 0.0
        best_loss_train=10000.0
        
        print("Training started:")
        dataloaders = self.dataset_dataloaders()
        dataset_sizes = self.dataset_sizes()
        
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
    
            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train() 
                else:
                    model.eval()  
    
                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
    #                 print(inputs.size(0))
    #                 break
#                    batch_size_ = len(inputs)
    #                 print("Batch Size : ",batch_size_)
    #                 break
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
    
                    optimizer.zero_grad()
    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = None
                        if self.model_name == INCEPTION_V3_MODEL and phase== "train":
                            outputs = model(inputs)
                            outputs = outputs[0]
                        else:
                            outputs = model(inputs)
                            
                        _, preds = torch.max(outputs, 1)
                        
                        loss = criterion(outputs, labels)
    #                     print(outputs.grad_fn)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    running_loss += loss.item() * inputs.size(0)
    #                 batch_corrects = torch.sum(preds == labels.data).item()
    #                 running_corrects += batch_corrects
                    running_corrects += torch.sum(preds == labels.data)
    
    #             print("running loss:- ",running_loss)
    #             print("running_corrects:- ",running_corrects)
    #             print(phase)
    #             print(dataset_sizes[phase])
                
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
    
                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                if phase == "validation" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                if phase == "train" and epoch_acc > best_acc_train:
                    best_acc_train = epoch_acc
                if phase == "train" and epoch_loss < best_loss_train:
                    best_loss_train = epoch_loss
    #         print(best_acc)
                if phase == "train":
                    scheduler.step()
            
            trial.report(epoch_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if epoch > 6 and best_acc < 0.8:
                raise optuna.TrialPruned()
    
        time_elapsed = time.time() - since
        model.load_state_dict(best_model_wts)
        
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best loss: {:4f}'.format(best_loss))
        print("\n\nBest test loss: {:.4f} | Best test accuracy: {:.4f}".format(best_loss, best_acc))
    
        
    #     torch.save(copy.deepcopy(model.state_dict()),MODEL_PATH +"/"+name+'.pth',_use_new_zipfile_serialization=False)
        return model, best_acc,best_loss
    
    
    
    
    def test_model_adversary(self,model,adversary):
        since = time.time()
        print("\n\nTesting and Adversarial attack performance testing started:")
        print(adversary)
        running_corrects_test=0
        running_corrects_test_untargetted=0
    
        batch_corrects_test=0
        batch_corrects_test_untargetted =0 
    
        phase = "validation"
        
        dataloaders = self.dataset_dataloaders()
        dataset_sizes = self.dataset_sizes()

    
        for inputs,labels in dataloaders[phase]:
    #        batch_size_ = len(inputs)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            adv_untargeted = adversary.perturb(inputs, labels)
            
            outputs = None
            if self.model_name == "inception_v3":
                outputs = model(inputs)
                outputs = outputs[0]
                
                att_outputs = model(adv_untargeted)
                att_outputs = att_outputs[0]
            else:
                outputs = model(inputs)
                att_outputs = model(adv_untargeted)
                
            _, pred_cln = torch.max(outputs, 1)
            _, pred_untargeted_adv =  torch.max(att_outputs, 1)
#            pred_cln = predict_from_logits(model(inputs))
#            pred_untargeted_adv = predict_from_logits(model(adv_untargeted))
          # print("__________________")
          # print("original: ",pred_cln)
          # print("untargetted Attack: ",pred_untargeted_adv)
          # print("__________________")
          # outputs = model(inputs)
          # _,preds = torch.max(outputs,1)
            
            # batch_corrects = torch.sum(preds == labels.data).item()
            batch_corrects_test = torch.sum(pred_cln == labels.data).item()
            batch_corrects_test_untargetted = torch.sum(pred_untargeted_adv == labels.data).item()
            
            
            running_corrects_test += batch_corrects_test
            running_corrects_test_untargetted += batch_corrects_test_untargetted 
            
        # final_acc = running_corrects / dataset_sizes[phase]
        final_acc_test = running_corrects_test / dataset_sizes[phase]
        final_acc_test_untargetted = running_corrects_test_untargetted / dataset_sizes[phase]
        
    #     print("Final Accuracy(original model): ",final_acc,final_acc_test)
        time_elapsed = time.time() - since
        print(
            "Testing completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
        )
    
        print("Final Accuracy(original model): ",final_acc_test)
        print("Adversarial Attack(untargetted) Accuracy: ",final_acc_test_untargetted)
        return final_acc_test,final_acc_test_untargetted
    
    
    def test_model_with_train(self,model):
        since = time.time()
        print("\n\nTraining accuracy testing started:")
        
        running_corrects =0
        batch_corrects=0
        
        phase = "train"
        dataloaders = self.dataset_dataloaders()
        dataset_sizes = self.dataset_sizes()
    
        for inputs,labels in dataloaders[phase]:
    #        batch_size_ = len(inputs)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
#            preds = predict_from_logits(model(inputs))
            outputs = None
            if self.model_name == "inception_v3":
                outputs = model(inputs)
                outputs = outputs[0]
                
#                att_outputs = model(adv_untargeted)
#                att_outputs = att_outputs[0]
            else:
                outputs = model(inputs)
#                att_outputs = model(adv_untargeted)
                
            _, preds = torch.max(outputs, 1)
    
            batch_corrects = torch.sum(preds == labels.data).item()
            running_corrects += batch_corrects
            
        final_acc = running_corrects / dataset_sizes[phase]
    
        
        time_elapsed = time.time() - since
        print(
            "Testing completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
        )
        print("Final Accuracy(for training Data model): ",final_acc)
        
        return final_acc

    
    def test_model(self,model):
        since = time.time()

        running_corrects =0
#        batch_corrects=0
#        running_loss = 0.0
        model.eval()
        phase = "validation"
        dataloaders = self.dataset_dataloaders()
        dataset_sizes = self.dataset_sizes()
        
#        print(dataloaders)
#        print(dataset_sizes)
        
        for inputs,labels in dataloaders[phase]:
    #        batch_size_ = len(inputs)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = None
                if self.model_name == "inception_v3" and phase== "train":
                    outputs = model(inputs)
                    outputs = outputs[0]
                else:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
#                print("______________________")
#                print("outputs:- \n",outputs)
#                print(_)
#                print("preds:- ",preds)
#                print("labels:- ",labels.data)
#                print("______________________")
                
#                loss = criterion(outputs, labels)
    
#            running_loss += loss.item() * inputs.size(0)
    #                 batch_corrects = torch.sum(preds == labels.data).item()
    #                 running_corrects += batch_corrects
            running_corrects += torch.sum(preds == labels.data)
            
#            preds = predict_from_logits(model(inputs))
#            outputs = None
#            if self.model_name == "inception_v3":
#                outputs = model(inputs)
#                outputs = outputs[0]
#                
##                att_outputs = model(adv_untargeted)
##                att_outputs = att_outputs[0]
#            else:
#                outputs = model(inputs)
##                att_outputs = model(adv_untargeted)
#                
#            _, preds = torch.max(outputs, 1)
#    
#            batch_corrects = torch.sum(preds == labels.data)
#            print("______________________")
#            print("outputs:- \n",outputs)
#            print(_)
#            print("preds:- ",preds)
#            print("labels:- ",labels.data)
#            print("______________________")
##            batch_corrects = torch.sum(preds == labels.data).item()
#            running_corrects += batch_corrects
            
        final_acc = running_corrects / dataset_sizes[phase]
    
        # print("Final Accuracy(original model): ",final_acc,final_acc_test)
        time_elapsed = time.time() - since
        print(
            "Testing completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
        )
        print("Final Accuracy(test model): ",final_acc)
        
        return final_acc

    
    