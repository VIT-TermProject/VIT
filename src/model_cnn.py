import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class My(nn.Module):
    def __init__(self):
        super(My,self).__init__()
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(3,64,3,2,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,2,1),
            nn.ReLU(),
        )
        """
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(128,256,3,2,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(256,512,3,2,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
        )
        
        self.up1 = nn.ConvTranspose2d(512,256,4,2,1)
        self.up2 = nn.ConvTranspose2d(256,128,4,2,1)
        """
        self.up3 = nn.ConvTranspose2d(128,64,4,2,1)
        self.up4 = nn.ConvTranspose2d(64,32,4,2,1)
        self.out = nn.Conv2d(32,3,3,1,1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        """
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.up1(x)
        x = self.up2(x)
        """
        x = self.up3(x)
        x = self.up4(x)
        out = self.out(x)
        
        return out
    
if __name__ == "__main__":
    x = torch.FloatTensor(np.zeros((1,3,64,64))) # 1,3,64,64 -> 1, 16, 16x16x3
    net = My()
    out = net(x)
    print(out.shape)
