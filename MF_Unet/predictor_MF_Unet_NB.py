#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 22:57:16 2022

@author: y
"""


import torch


def predictor_MF_Unet_NB(input, MF_Unet, case, device):
    if '_MAP' in case:
        if 'Aleatoric_He' in case:
            outputs = []
            ss = []

            for net in MF_Unet:
                net.eval()
                net.to(device)
                with torch.no_grad():
                    output, s = net(
                        input.to(device=device, dtype=torch.float32))

                # output = output.to(device='cpu')
                # s = s.to(device='cpu')
                outputs.append(output)
                ss.append(s)
                net.train()
            mean = sum(outputs)/len(outputs)
            var_e = sum([((x - mean) ** 2) for x in outputs])/len(outputs)
            std_e = var_e**0.5
            var_a = torch.exp(sum(ss)/len(ss))
            std_a = var_a**0.5
            std = (var_e+var_a)**0.5
            return mean, std_a, std_e, std

        else:
            outputs = []

            for net in MF_Unet:
                net.eval()
                net.to(device)
                with torch.no_grad():
                    output = net(input.to(device=device, dtype=torch.float32))

                # output = output.to(device='cpu')
                outputs.append(output)
                net.train()
            mean = sum(outputs)/len(outputs)
            var = sum([((x - mean) ** 2) for x in outputs])/len(outputs)
            std = var**0.5
            return mean, std
    else:

        MF_Unet.eval()
        MF_Unet.to(device)
        with torch.no_grad():
            output = MF_Unet(input.to(device=device, dtype=torch.float32))

        # output = output.to(device='cpu')
        MF_Unet.train()
        return output
