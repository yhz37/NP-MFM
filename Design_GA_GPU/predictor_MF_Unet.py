#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:08:33 2022

@author: y
"""

import torch
from torch.utils.data import Dataset, DataLoader

class EvalDataset(Dataset):
    def __init__(self,x):
        self.data = x
        
    def __getitem__(self, index):
        x_index = self.data[index]
        return x_index
    
    def __len__(self):
        return len(self.data)



def predictor_MF_Unet(input,MF_Unet,case,device):
    if '_MAP' in case:
        if 'Aleatoric_He' in case:
            outputs=[]
            ss=[]
            Eval_set = EvalDataset(input)
            Eval_loader = DataLoader(dataset=Eval_set, 
                                      batch_size=100, 
                                      shuffle=False, num_workers=4)
            for net in MF_Unet:
                net.eval()
                net.to(device)
                with torch.no_grad():
                    for ind,dataset_ind in enumerate(Eval_loader):
                        dataset_ind = dataset_ind.to(device=device, dtype=torch.float32)
                        output_ind,s_ind = net(dataset_ind)
                        if ind==0:
                            output=output_ind
                            s=s_ind
                        else:
                            output = torch.cat((output,output_ind),0)
                            s = torch.cat((s,s_ind),0)
                output=output.to(device='cpu')
                s=s.to(device='cpu')
                outputs.append(output)
                ss.append(s)
                net.train()
            mean = sum(outputs)/len(outputs)
            var_e = sum([((x - mean) ** 2) for x in outputs])/len(outputs)
            std_e = var_e**0.5
            var_a = torch.exp(sum(ss)/len(ss))
            std_a = var_a**0.5
            std = (var_e+var_a)**0.5
            return mean,std_a,std_e,std
            
        else:
            outputs=[]
            Eval_set = EvalDataset(input)
            Eval_loader = DataLoader(dataset=Eval_set, 
                                      batch_size=100, 
                                      shuffle=False, num_workers=4)
            for net in MF_Unet:
                net.eval()
                net.to(device)
                with torch.no_grad():
                    for ind,dataset_ind in enumerate(Eval_loader):
                        dataset_ind = dataset_ind.to(device=device, dtype=torch.float32)
                        output_ind = net(dataset_ind)
                        if ind==0:
                            output=output_ind
                        else:
                            output = torch.cat((output,output_ind),0)
                output=output.to(device='cpu')
                outputs.append(output)
                net.train()
            mean = sum(outputs)/len(outputs)
            var = sum([((x - mean) ** 2) for x in outputs])/len(outputs)
            std = var**0.5
            return mean, std
    else:
        Eval_set = EvalDataset(input)
        Eval_loader = DataLoader(dataset=Eval_set, 
                                  batch_size=100, 
                                  shuffle=False, num_workers=4)
        MF_Unet.eval()
        net.to(device)
        with torch.no_grad():
            for ind,dataset_ind in enumerate(Eval_loader):
                dataset_ind = dataset_ind.to(device=device, dtype=torch.float32)
                output_ind = MF_Unet(dataset_ind)
                if ind==0:
                    output=output_ind
                else:
                    output = torch.cat((output,output_ind),0)
        output=output.to(device='cpu')
        MF_Unet.train()
        return output