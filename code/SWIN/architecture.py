import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np
import torch.nn.functional as F


class SwinEmbedding(nn.Module):
    def __init__(self, 
                 emb_size : int = 96,
                 patch_size : int = 4,
                 n_channels : int = 3):
        super().__init__()
        
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        self.linear_embedding = nn.Conv2d(self.n_channels, 
                                          self.emb_size, 
                                          kernel_size=self.patch_size,
                                          stride = self.patch_size)
        self.rearrange = Rearrange('b c h w -> b (h w) c')
        
    def forward(self, 
                x : torch.Tensor) -> torch.Tensor:
        x = self.linear_embedding(x)
        x = self.rearrange(x)
        
        return x
    
class PatchMerging(nn.Module):
    def __init__(self,
                 emb_size : int):
        super().__init__()
        self.emb_size = emb_size
        self.linear = nn.Linear(4*self.emb_size, 2*self.emb_size)
        
    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H = W = int(np.sqrt(L) / 2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s1 s2 c)', s1=2, s2=2, h=H, w=W)
        x = self.linear(x)
        return x
    
class ShiftedWindowMSA(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=7, shifted=True):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        self.linear1 = nn.Linear(emb_size, 3*emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)

        self.pos_embeddings = nn.Parameter(torch.randn(window_size*2 - 1, window_size*2 - 1))
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1
        
        self.row_mask = nn.Parameter(torch.zeros((self.window_size**2, self.window_size**2)), requires_grad=False)

    def forward(self, x):
        h_dim = self.emb_size / self.num_heads
        height = width = int(np.sqrt(x.shape[1]))
        x = self.linear1(x)
        
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)
        
        if self.shifted:
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))
        
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)            
        
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        wei = (Q @ K.transpose(4,5)) / np.sqrt(h_dim)
        
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding
        
        if self.shifted:
            row_mask = self.row_mask
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask
        
        wei = F.softmax(wei, dim=-1) @ V
        
        x = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)
        x = rearrange(x, 'b h w c -> b (h w) c')
        
        return self.linear2(x)
    
    
class MLP(nn.Module):
    def __init__(self, 
                 emb_size : int):
        super().__init__()
        self.emb_size = emb_size
        
        self.ff = nn.Sequential(
            nn.Linear(self.emb_size, 2*self.emb_size),
            nn.GELU(),
            nn.Linear(2*self.emb_size, self.emb_size)
        )
    
    def forward(self, 
                x : torch.Tensor) -> torch.Tensor:
        return self.ff(x)

class SwinEncoder(nn.Module):
    def __init__(self,
                 emb_size : int,
                 num_heads : int,
                 window_size : int = 7):
        super().__init__()
        
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.WMSA = ShiftedWindowMSA(self.emb_size,
                                     self.num_heads,
                                     self.window_size,
                                     shifted=False)
        
        self.SWMSA = ShiftedWindowMSA(self.emb_size,
                                      self.num_heads,
                                      self.window_size,
                                      shifted=True)
        
        self.ln = nn.LayerNorm(self.emb_size)
        
        self.MLP = MLP(self.emb_size)
        
    def forward(self, 
                x : torch.Tensor) -> torch.Tensor:
        # Window attention
        x = x + self.WMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        
        # Shifted Window attention
        x = x + self.SWMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        
        return x
        
        
class SWINTransformer(nn.Module):
    def __init__(self,
                 emb_size : int,
                 num_class : int = 5):
        super().__init__()
        self.emb_size = emb_size
        self.num_class = num_class
        
        self.embedding = SwinEmbedding(self.emb_size, patch_size=2)
        self.patch_merging1 = PatchMerging(self.emb_size)
        self.patch_merging2 = PatchMerging(self.emb_size*2)
            
        self.stage1 = nn.ModuleList([SwinEncoder(self.emb_size, 3, window_size=4) for i in range(2)])
        self.stage3 = nn.ModuleList([SwinEncoder(self.emb_size * 2, 6, window_size=4) for i in range(6)])
        self.stage4 = nn.ModuleList([SwinEncoder(self.emb_size * 4, 12, window_size=4) for i in range(4)])
        
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size = 1)
        self.layer = nn.Linear(self.emb_size * 4, self.num_class)
        
        
    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for stage in self.stage1:
            x = stage(x)
        x = self.patch_merging1(x)
        
        for stage in self.stage3:
            x = stage(x)
        
        x = self.patch_merging2(x)
        for stage in self.stage4:
            x = stage(x)
        x = self.layer(self.avgpool1d(x.transpose(1, 2)).squeeze(2))
        return x
        
            
        