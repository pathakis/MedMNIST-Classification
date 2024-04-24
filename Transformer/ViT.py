# Written by Oscar Rosman
# Date: 2024-04-24

from einops import rearrange
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
import torch
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from einops import repeat
from medmnist import PneumoniaMNIST

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(x,x,x)
        return attn_output


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels=3,
                 patch_size=8,
                 embedding_size=224
                 ):
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embedding_size)
            )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class ViT(nn.Module):
    def __init__(self, channels=3, 
                 img_size=224, 
                 patch_size=4, 
                 embedding_dim=32, 
                 layers=6, 
                 out_dim=37, 
                 dropout=0.1, 
                 heads=2
                 ):
        super(ViT, self).__init__()

        # Attributes
        self.channels = channels # Number of channels in the input image (Grayscale = 1, RGB = 3)
        self.height = img_size # Height of the input image
        self.width = img_size # Width of the input image
        self.patch_size = patch_size # Size of the patches to be extracted from the input image (Think of mini images within image or kenel snapshots)
        self.n_layers = layers

        # Patching
        self.patch_embedding = PatchEmbedding(in_channels=channels,
                                              patch_size=patch_size,
                                              embedding_size=embedding_dim
                                              )
        
        # Learnable parameters
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Transformer encoder
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(embedding_dim, Attention(embedding_dim, heads, dropout))),
                ResidualAdd(PreNorm(embedding_dim, FeedForward(embedding_dim, embedding_dim, dropout)))
            )
            self.layers.append(transformer_block)

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, out_dim))

    def forward(self, img):
        # Get patch embedding vectors
        img = self.patch_embedding(img)
        b, n, _ = img.shape

        # Add positional embedding to the patches
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        img = torch.cat([cls_tokens, img], dim=1)
        img += self.pos_embedding[:, :(n + 1)]

        # Transformer layers
        for layer in self.layers:
            img = layer(img)

        # Classification head
        assigned_class = self.head(img[:, 0, :])
        return assigned_class
