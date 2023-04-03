import argparse
import logging
import sys
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import scipy.io as sio
import pickle

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import matplotlib.pyplot as plt
torch.cuda.set_device(3)

Data = sio.loadmat('/home/y/y/Python/MFNN/Data/mCGG_7P_Training_0.7_1_500.mat')

input = Data['input']
PBCM_CG = Data['PBCM_CG']
CFD_CG = Data['CFD_CG']
PBCM_CG = np.transpose(PBCM_CG)
CFD_CG = np.transpose(CFD_CG)
numData = input.shape[0]
input = input.reshape(numData,1,-1)
PBCM_CG = PBCM_CG.reshape(numData, 1, -1)
CFD_CG = CFD_CG.reshape(numData, 1, -1)


TestData = sio.loadmat('/home/y/y/Python/MFNN/Data/mCGG_7P_Testing_0.7_1_20.mat')

Test_input = TestData['Test_input']
Test_PBCM_CG = TestData['Test_PBCM_CG']
Test_CFD_CG = TestData['Test_CFD_CG']
Test_PBCM_CG = np.transpose(Test_PBCM_CG)
Test_CFD_CG = np.transpose(Test_CFD_CG)
Test_numData = Test_input.shape[0]
Test_input = Test_input.reshape(Test_numData,1,-1)
Test_PBCM_CG = Test_PBCM_CG.reshape(Test_numData, 1, -1)
Test_CFD_CG = Test_CFD_CG.reshape(Test_numData, 1, -1)

dir_checkpoint = Path('./checkpoints/')

class FluidMixerDataset(Dataset):    

    # Initialize your data
    def __init__(self, x, y):        
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len        


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 10,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              amp: bool = False):
    

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor =0.98, min_lr=1e-6 )  #goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0

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

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)
        for param_group in optimizer.param_groups:
             print(param_group['lr'])
    for param_group in optimizer.param_groups:
        Final_lr = param_group['lr']
    return Final_lr,scheduler


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')


    net.to(device=device)
    try:
        # # 1. Create dataset
        # dataset = FluidMixerDataset(PBCM_CG, CFD_CG)

        # # 2. Split into train / validation partitions
        # n_val = int(len(dataset) * args.val / 100)
        # n_train = len(dataset) - n_val
        # train_set, val_set = random_split(dataset, [n_train, n_val],                generator=torch.Generator().manual_seed(0))
        
        train_set = FluidMixerDataset(PBCM_CG, CFD_CG)
        n_train = len(train_set)
        val_set = FluidMixerDataset(Test_PBCM_CG, Test_CFD_CG)
        n_val = len(val_set)        

        
        Final_lr,scheduler  = train_net(net=net,
                                        epochs=args.epochs,
                                        batch_size=args.batch_size,
                                        learning_rate=args.lr,
                                        device=device,
                                        val_percent=args.val / 100,
                                        amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
    
    net = net.to('cpu')
    criterion = nn.MSELoss()
    PBCM_CG_Val,CFD_CG_Val = val_set[:]
    PBCM_CG_Val = PBCM_CG_Val.to(device='cpu', dtype=torch.float32)
    CFD_CG_Val = CFD_CG_Val.to(device='cpu', dtype=torch.float32)
    CFD_CG_Val_Predicted = net(PBCM_CG_Val)
    loss_Val = criterion(CFD_CG_Val_Predicted, CFD_CG_Val)
    PBCM_CG_Val = PBCM_CG_Val.detach().numpy()
    CFD_CG_Val = CFD_CG_Val.to(device='cpu', dtype=torch.float32)
    CFD_CG_Val = CFD_CG_Val.detach().numpy()
    CFD_CG_Val_Predicted = CFD_CG_Val_Predicted.detach().numpy()
    
    PBCM_CG_Train,CFD_CG_Train = train_set[:]
    PBCM_CG_Train = PBCM_CG_Train.to(device='cpu', dtype=torch.float32)
    CFD_CG_Train_Predicted = net(PBCM_CG_Train)
    CFD_CG_Train = CFD_CG_Train.to(device='cpu', dtype=torch.float32)
    loss_Train = criterion(CFD_CG_Train_Predicted, CFD_CG_Train)
    PBCM_CG_Train = PBCM_CG_Train.detach().numpy()
    CFD_CG_Train = CFD_CG_Train.to(device='cpu', dtype=torch.float32)
    CFD_CG_Train = CFD_CG_Train.detach().numpy()
    CFD_CG_Train_Predicted = CFD_CG_Train_Predicted.detach().numpy()
    
    xConc = np.linspace(0, 1, num=100)
    xConc = xConc.reshape([100,1])
    for i in range(5):
        plt.plot(xConc,PBCM_CG_Train[i,:,:].reshape([100,1]), 'g--', xConc, CFD_CG_Train[i,:,:].reshape([100,1]), 'r-',xConc,CFD_CG_Train_Predicted[i,:,:].reshape([100,1]), 'b:')
        plt.show()
    for i in range(5):
        plt.plot(xConc,PBCM_CG_Val[i,:,:].reshape([100,1]), 'g--', xConc, CFD_CG_Val[i,:,:].reshape([100,1]), 'r-',xConc,CFD_CG_Val_Predicted[i,:,:].reshape([100,1]), 'b:')
        plt.show()
    print(loss_Train.item())
    print(loss_Val.item())
    
    with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([net, Final_lr, loss_Train, loss_Val, scheduler], f)