""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 16)
        # self.down1 = Down(16, 32)
        # factor = 2 if bilinear else 1
        # self.down2 = Down(32, 64 // factor)
        # self.up1 = Up(64, 32 // factor, bilinear)
        # self.up2 = Up(32, 16, bilinear)
        # self.outc = OutConv(16, n_classes)
        # self.inc = DoubleConv(n_channels, 8)
        # factor = 2 if bilinear else 1
        # self.down1 = Down(8, 16 // factor)
        # self.up1 = Up(16, 8 // factor, bilinear)
        # self.outc = OutConv(8, n_classes)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class UNet2_7(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2_7, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        self.fc1 = nn.Linear(3079, 3079)   
        self.fc2 = nn.Linear(3079, 3072)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        xConc,xb = torch.split(x, [100,7],2)
        x1 = self.inc(xConc)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = torch.flatten(x4, 1)
        xb = torch.flatten(xb, 1)
        x6 = self.fc1(torch.cat((x5, xb),1))
        x7 = self.fc2(x6)
        x7 = torch.reshape(x7, x4.size())
        x = self.up1(x7, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class UNet2_7_Aleatoric(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2_7_Aleatoric, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        self.fc1 = nn.Linear(3079, 3079)   
        self.fc2 = nn.Linear(3079, 3072)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        
        self.up1_s = Up(256, 128 // factor, bilinear)
        self.up2_s = Up(128, 64 // factor, bilinear)
        self.up3_s = Up(64, 32, bilinear)
        self.outc_s = OutConv(32, n_classes)

    def forward(self, x):
        xConc,xb = torch.split(x, [100,7],2)
        x1 = self.inc(xConc)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = torch.flatten(x4, 1)
        xb = torch.flatten(xb, 1)
        x6 = self.fc1(torch.cat((x5, xb),1))
        x7 = self.fc2(x6)
        x7 = torch.reshape(x7, x4.size())        
        x = self.up1(x7, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        
        
        s = self.up1_s(x7, x3)
        s = self.up2_s(s, x2)
        s = self.up3_s(s, x1)
        s = self.outc(s)
        return logits,s
    
class UNet2_9(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2_9, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        self.fc1 = nn.Linear(3081, 3081)   
        self.fc2 = nn.Linear(3081, 3072)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        xConc,xb = torch.split(x, [100,9],2)
        x1 = self.inc(xConc)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = torch.flatten(x4, 1)
        xb = torch.flatten(xb, 1)
        x6 = self.fc1(torch.cat((x5, xb),1))
        x7 = self.fc2(x6)
        x7 = torch.reshape(x7, x4.size())
        x = self.up1(x7, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
    
class UNet2_9_NPPI(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2_9_NPPI, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        self.fc1 = nn.Linear(3072, 3072)   
        self.fc2 = nn.Linear(3072, 3072)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        xConc=x
        x1 = self.inc(xConc)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = torch.flatten(x4, 1)

        x6 = self.fc1(x5)
        x7 = self.fc2(x6)
        x7 = torch.reshape(x7, x4.size())
        x = self.up1(x7, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class UNet2_9_Aleatoric(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2_9_Aleatoric, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        self.fc1 = nn.Linear(3081, 3081)   
        self.fc2 = nn.Linear(3081, 3072)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        
        self.up1_s = Up(256, 128 // factor, bilinear)
        self.up2_s = Up(128, 64 // factor, bilinear)
        self.up3_s = Up(64, 32, bilinear)
        self.outc_s = OutConv(32, n_classes)

    def forward(self, x):
        xConc,xb = torch.split(x, [100,9],2)
        x1 = self.inc(xConc)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = torch.flatten(x4, 1)
        xb = torch.flatten(xb, 1)
        x6 = self.fc1(torch.cat((x5, xb),1))
        x7 = self.fc2(x6)
        x7 = torch.reshape(x7, x4.size())
        x = self.up1(x7, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        
        s = self.up1_s(x7, x3)
        s = self.up2_s(s, x2)
        s = self.up3_s(s, x1)
        s = self.outc_s(s)
        return logits,s

class UNet2_9_MAP(nn.Module):
    def __init__(self, n_channels, n_classes,init_log_noise, bilinear=False):
        super(UNet2_9_MAP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        self.fc1 = nn.Linear(3081, 3081)   
        self.fc2 = nn.Linear(3081, 3072)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))


    def forward(self, x):
        xConc,xb = torch.split(x, [100,9],2)
        x1 = self.inc(xConc)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = torch.flatten(x4, 1)
        xb = torch.flatten(xb, 1)
        x6 = self.fc1(torch.cat((x5, xb),1))
        x7 = self.fc2(x6)
        x7 = torch.reshape(x7, x4.size())
        x = self.up1(x7, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits