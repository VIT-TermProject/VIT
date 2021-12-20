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

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.image_height = image_height
        self.patch_size = patch_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.up1 = nn.ConvTranspose2d(512,256,4,2,1)
        self.up2 = nn.ConvTranspose2d(256,128,4,2,1)
        self.up3 = nn.ConvTranspose2d(128,64,4,2,1)
        self.up4 = nn.ConvTranspose2d(64,32,4,2,1)
        self.out = nn.Conv2d(32,3,3,1,1)
        #self.out = nn.Sequential(
        #    nn.Conv2d(128,64,3,1,1),
        #    nn.ReLU(),
        #    nn.Conv2d(64,3,kernel_size=1),
        #)
        #self.out = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)  
        #self.out = nn.Conv2d(2,32,kernel_size=3,stride=1,padding=1)  
        #self.out = nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1)  
        # transformer output shape이 안바뀌네요... deconvolution 쓰자니 더 커지고.. 고민됩니다...
        #self.up1 = nn.ConvTranspose2d(2,64,kernel_size=4,stride=2,padding=1)


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        
        x = self.transformer(x)
        x = x[:,1:]
        #x = x.transpose(0,1).contiguous().view(x.size(1), -1, 512)
        x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),4,1,stride=1)  # 1안 patch를 2x16x16로 취급
        #x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),16,1,stride=1) # 2안 patch를 512x1x1로 취급
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.up4(x))
        x = self.out(x)
        return x


if __name__ == "__main__":
    x = torch.FloatTensor(np.zeros((1,3,64,64))) # 1,3,64,64 -> 1, 16, 16x16x3
    net = ViT(image_size=64,patch_size=4,num_classes=1,dim=512,mlp_dim=1024, depth=6, heads=16)
    out = net(x)
    print(out.shape)
