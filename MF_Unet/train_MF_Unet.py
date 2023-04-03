#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:58:43 2022

@author: y
"""
from torch.utils.data import DataLoader
from torch import optim
import torch
from tqdm import tqdm
import copy
import numpy as np
import torch.nn as nn

from evaluate import evaluate
from FluidMixerDataset import FluidMixerDataset
from log_gaussian_loss import log_gaussian_loss
from unet import UNet2_7, UNet2_9,UNet2_7_Aleatoric, UNet2_9_Aleatoric,UNet2_9_NPPI


def initial_net(case,device):
    if '_9' in case:
        if 'Aleatoric_He' in case:
            net = UNet2_9_Aleatoric(n_channels=1, n_classes=1, bilinear=False)
        elif 'NPPI' in case:
            net = UNet2_9_NPPI(n_channels=1, n_classes=1, bilinear=False)
        else:
            net = UNet2_9(n_channels=1, n_classes=1, bilinear=False)
    elif '_7' in case:
        if 'Aleatoric_He' in case:
            net = UNet2_7_Aleatoric(n_channels=1, n_classes=1, bilinear=False)
        else:
            net = UNet2_7(n_channels=1, n_classes=1, bilinear=False)
    net.to(device=device)

    return net


def train_MF_Unet(case,train_set,val_set,epochs,batch_size,lr,device,val,amp):
    if '_MAP' in case:
        if 'Aleatoric_He' in case:
            criterion = log_gaussian_loss
        else:
            criterion = nn.MSELoss()
        MF_Unet,Final_lr,scheduler = train_net_MAP(case=case,
                                                device=device,
                                                train_set=train_set,
                                                val_set=val_set,
                                                criterion=criterion,
                                                epochs=epochs,
                                                batch_size=batch_size,
                                                learning_rate=lr,
                                                val_percent=val / 100,
                                                amp=amp)
    else:

        criterion = nn.MSELoss()
        MF_Unet,Final_lr,scheduler = train_net(case=case,
                                           device=device,
                                           train_set=train_set,
                                           val_set=val_set,
                                           criterion=criterion,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           learning_rate=lr,
                                           val_percent=val / 100,
                                           amp=amp)
    return MF_Unet,Final_lr,scheduler

def train_net(case,
              device,
              train_set,
              val_set,
              criterion,
              epochs: int = 5,
              batch_size: int = 10,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              amp: bool = False):
    
    net = initial_net(case,device)
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor =0.98, min_lr=1e-6 )  #goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    n_train = len(train_set)


    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='CG') as pbar:
            for batch, (Input, Output) in enumerate(train_loader):


                assert Input.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded Input have {Input.shape[1]} channels. Please check that ' \
                    'the Input are loaded correctly.'

                Input = Input.to(device=device, dtype=torch.float32)
                Output = Output.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    Output_pred = net(Input)
                    loss = criterion(Output_pred, Output)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(Input.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (Epoch)': epoch_loss})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                #         histograms = {}
                #         for tag, value in net.named_parameters():
                #             tag = tag.replace('/', '.')
                #             histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                #             histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device,criterion,case)
                        scheduler.step(val_score)
        for param_group in optimizer.param_groups:
             print(param_group['lr'])
    for param_group in optimizer.param_groups:
        Final_lr = param_group['lr']
    return net, Final_lr,scheduler

def train_net_MAP(case,
                  device,
                  train_set,
                  val_set,
                  criterion,
                  no_nets: int = 5,
                  epochs: int = 100,
                  batch_size: int = 10,
                  learning_rate: float = 1e-5,
                  val_percent: float = 0.1,
                  amp: bool = False):
    nets = []
    for n in range(no_nets):
        # 3. Create data loaders
        net = initial_net(case,device)
        loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
        sub_idx = np.random.choice(np.arange(0, len(train_set)), size = (int(len(train_set)*0.8),), replace=True)
        Train_x,Train_y = train_set[sub_idx]
        Train_x = Train_x.numpy()
        Train_y = Train_y.numpy()
        train_set_net = FluidMixerDataset(Train_x,Train_y)
        train_loader = DataLoader(train_set_net, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=0, amsgrad=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor =0.98, min_lr=1e-6 )  #goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        global_step = 0

        # 5. Begin training
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=int(len(train_set)*0.8), desc=f'Net {n+1}/{no_nets} - Epoch {epoch + 1}/{epochs}', unit='CG') as pbar:
                for batch, (Input, Output) in enumerate(train_loader):


                    assert Input.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded Input have {Input.shape[1]} channels. Please check that ' \
                        'the Input are loaded correctly.'

                    Input = Input.to(device=device, dtype=torch.float32)
                    Output = Output.to(device=device, dtype=torch.float32)

                    with torch.cuda.amp.autocast(enabled=amp):
                        if 'Aleatoric_He' in case:
                            Output_pred,s = net(Input)
                            loss = criterion(Output_pred, Output,s)                            
                        else:                            
                            Output_pred = net(Input)
                            loss = criterion(Output_pred, Output)

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(Input.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()

                    pbar.set_postfix(**{'loss (Epoch)': epoch_loss})

                    # Evaluation round
                    division_step = (len(train_set) // (10 * batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:

                            val_score = evaluate(net, val_loader, device,criterion,case)
                            scheduler.step(val_score)
            for param_group in optimizer.param_groups:
                print(param_group['lr']) 
            
        net.to('cpu')
        nets.append(copy.deepcopy(net))
    for param_group in optimizer.param_groups:
        Final_lr = param_group['lr']
    return nets, Final_lr,scheduler

