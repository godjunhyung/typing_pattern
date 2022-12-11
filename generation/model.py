import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, pass_len, batch_size):
        super().__init__()
        self.nz_size = 100
        self.gen = nn.Sequential(
            nn.Linear(in_features=30000+self.nz_size, out_features=2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, pass_len*101)
        )
        self.pass_len = pass_len
        
    def forward(self, img):
        batch_size = img.shape[0]
        noise = torch.randn(batch_size, self.nz_size, device='cuda')
        img = img.reshape(batch_size, -1)
        x = torch.cat((img,noise),-1)
        x = self.gen(x).reshape(batch_size, 101, self.pass_len)
        return x

class Discriminator(nn.Module):
    def __init__(self, pass_len):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Linear(in_features=pass_len*101, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, pwd):
        b_size = pwd.shape[0]
        pwd = pwd.reshape(b_size, -1)
        x = self.dis(pwd)
        return x