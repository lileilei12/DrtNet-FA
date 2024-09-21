from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.gn = nn.GroupNorm(outchannel, outchannel, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.relu(self.gn(self.conv(x)))
        return x
    
class Down(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.gn = nn.GroupNorm(outchannel, outchannel, eps=1e-6)
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=2, stride=2, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True) 
       
    def forward(self, x):
        x = self.relu(self.gn(self.conv(x)))
        return x
    
class Up(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.gn = nn.GroupNorm(outchannel, outchannel, eps=1e-6)
        self.conv = nn.ConvTranspose2d(inchannel, outchannel, kernel_size=2, stride=2, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True) 
       
    def forward(self, x):
        x = self.relu(self.gn(self.conv(x)))
        return x

class Pre(nn.Module):
    def __init__(self):
        super().__init__()
        inchannel = 3
        outchannel = 16
        self.gn = nn.GroupNorm(outchannel, outchannel, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.relu(self.gn(self.conv(x)))
        return x
    
class Map(nn.Module):
    def __init__(self):
        super().__init__()
        inchannel = 32
        outchannel = 16
        self.gn = nn.GroupNorm(outchannel, outchannel, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.relu(self.gn(self.conv(x)))
        return x
    
class Res(nn.Module):
    def __init__(self):
        super().__init__()

        self.layere1 = nn.Sequential(
            Conv(inchannel=16, outchannel=16)
        )   
        self.down1 = nn.Sequential(
            Down(inchannel=16, outchannel=64)
        )
        self.layere2 = nn.Sequential(
            Conv(inchannel=64, outchannel=64),
            Conv(inchannel=64, outchannel=64)
        )
        self.down2 = nn.Sequential(
            Down(inchannel=64, outchannel=256)
        )
        self.layere3 = nn.Sequential(
            Conv(inchannel=256, outchannel=256),
            Conv(inchannel=256, outchannel=256),
            Conv(inchannel=256, outchannel=256)
        )
        self.down3 = nn.Sequential(
            Down(inchannel=256, outchannel=512)
        )
        self.layere4 = nn.Sequential(
            Conv(inchannel=512, outchannel=512),
            Conv(inchannel=512, outchannel=512),
            Conv(inchannel=512, outchannel=512)
        )
        self.down4 = nn.Sequential(
            Down(inchannel=512, outchannel=1024)
        )
        self.up1 = nn.Sequential(
            Up(inchannel=1024, outchannel=512)
        )
        self.layerd1 = nn.Sequential(
            Conv(inchannel=512*2, outchannel=512),
            Conv(inchannel=512, outchannel=512),
            Conv(inchannel=512, outchannel=512)
        )
        self.up2 = nn.Sequential(
            Up(inchannel=512, outchannel=256)
        )
        self.layerd2 = nn.Sequential(
            Conv(inchannel=256*2, outchannel=256),
            Conv(inchannel=256, outchannel=256),
            Conv(inchannel=256, outchannel=256)
        )
        self.up3 = nn.Sequential(
            Up(inchannel=256, outchannel=64)
        )
        self.layerd3 = nn.Sequential(
            Conv(inchannel=64*2, outchannel=64),
            Conv(inchannel=64, outchannel=64)
        )
        self.up4 = nn.Sequential(
            Up(inchannel=64, outchannel=16)
        )
        self.layerd4 = nn.Sequential(
            Conv(inchannel=16*2, outchannel=16)
        )  

    def forward(self, x):

        features1 = []
        features2 = []
        for i in range(4):
            layer = getattr(self, f"layere{i+1}")
            down = getattr(self, f"down{i+1}")
            x = layer(x) + x
            features1.append(x)
            x = down(x)
        features1 = features1[::-1]
        for k in range(4):
            layer = getattr(self, f"layerd{k+1}")
            up = getattr(self, f"up{k+1}")
            box = []
            x = up(x)
            s = x
            for j, conv in enumerate(layer):
                if j == 0:
                    x = conv(torch.cat([x, features1[k]],dim=1))
                if j != 0:
                    x = conv(x)
                box.append(x)
            features2.append(box)
            x = x + s

        return x, features2
    
class Res_ViT_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            Conv(inchannel=16, outchannel=16)
        )   
        self.down1 = nn.Sequential(
            Down(inchannel=16, outchannel=64)
        )
        self.layer2 = nn.Sequential(
            Conv(inchannel=64, outchannel=64),
            Conv(inchannel=64, outchannel=64)
        )
        self.down2 = nn.Sequential(
            Down(inchannel=64, outchannel=256)
        )
        self.layer3 = nn.Sequential(
            Conv(inchannel=256, outchannel=256),
            Conv(inchannel=256, outchannel=256),
            Conv(inchannel=256, outchannel=256)
        )
        self.down3 = nn.Sequential(
            Down(inchannel=256, outchannel=512)
        )
        self.layer4 = nn.Sequential(
            Conv(inchannel=512, outchannel=512),
            Conv(inchannel=512, outchannel=512),
            Conv(inchannel=512, outchannel=512)
        )
        self.down4 = nn.Sequential(
            Down(inchannel=512, outchannel=1024)
        )

    def forward(self, x):

        features = []
        for i in range(4):
            layer = getattr(self, f"layer{i+1}")
            down = getattr(self, f"down{i+1}")
            x = layer(x) + x
            features.append(x)
            x = down(x)

        return x, features[::-1]
    
class Res_ViT_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = nn.Sequential(
            Up(inchannel=1024, outchannel=512)
        )
        self.layer1 = nn.Sequential(
            Conv(inchannel=512*3, outchannel=512),
            Conv(inchannel=512*2, outchannel=512),
            Conv(inchannel=512*2, outchannel=512)
        )
        self.up2 = nn.Sequential(
            Up(inchannel=512, outchannel=256)
        )
        self.layer2 = nn.Sequential(
            Conv(inchannel=256*3, outchannel=256),
            Conv(inchannel=256*2, outchannel=256),
            Conv(inchannel=256*2, outchannel=256)
        )
        self.up3 = nn.Sequential(
            Up(inchannel=256, outchannel=64)
        )
        self.layer3 = nn.Sequential(
            Conv(inchannel=64*3, outchannel=64),
            Conv(inchannel=64*2, outchannel=64)
        )
        self.up4 = nn.Sequential(
            Up(inchannel=64, outchannel=16)
        )
        self.layer4 = nn.Sequential(
            Conv(inchannel=16*3, outchannel=16)
        )  

    def forward(self, x, f1, f2):
        
        for j in range(4):
            layer = getattr(self, f"layer{j+1}")
            up = getattr(self, f"up{j+1}")
            x = up(x)
            s = x
            for k, conv in enumerate(layer):
                if k == 0:
                    x = conv(torch.cat([x, f2[j], f1[j][0]], dim=1))
                if k != 0:
                    x = conv(torch.cat([x, f1[j][k]], dim=1)) 
            x = x + s

        return x