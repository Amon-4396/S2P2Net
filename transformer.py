import torch.nn as nn
import torch.nn.functional as F
import torch
import numbers
from einops import rearrange

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, ):
        super().__init__()
        # print('使用self atten')
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t:rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        scale = self.scale.unsqueeze(0).expand(x.shape[0], -1, q.shape[2], k.shape[2])
        dots = torch.einsum('bhid, bhjd -> bhij', q, k) * scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij, bhjd -> bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

class MLPBlk(nn.Module):
    def __init__(self, dim1, hidden_dim, dim2, dropout=0.1):
        super().__init__()
        hidden_dim = max(1, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(dim1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim2),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, ):
        super(CrossAttention, self).__init__()
        # print('使用cross atten')
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=bias)
    def forward(self, x, y):
        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim=-1)
        k, v = map(lambda t:rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        scale = self.scale.unsqueeze(0).expand(x.shape[0], -1, q.shape[2], k.shape[2])
        dots = torch.einsum('bhid, bhjd -> bhij', q, k) * scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij, bhjd -> bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

class Transformer(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, mid_dim=0, dropout=0.2):
        super(Transformer, self).__init__()
        if mid_dim == 0:
            mid_dim = self._get_mid_dim(dim, num_heads)
        self.in_embedding1 = MLPBlk(dim, mid_dim//4, mid_dim, dropout=dropout)
        self.attn = SelfAttention(mid_dim, num_heads, bias)
        self.out_embedding = MLPBlk(mid_dim, dim//4, dim, dropout=dropout)

    def forward(self, x1, class_token):
        x = torch.concat([class_token, x1], dim=1)
        temp1 = self.in_embedding1(x)
        out = self.attn(temp1) + temp1 # b, c, dim
        out = self.out_embedding(out) + x
        return out

    def _get_mid_dim(self, dim, num_heads):
        return (dim // num_heads + 1) * num_heads

if __name__ == "__main__":
    model = Transformer(169, num_heads=13)
    data1 = torch.randn((64, 64, 13, 13))
    data2 = torch.randn((64, 64, 13, 13))
    token = torch.randn((1, 1, 169))
    y = model(data1, data2, token)
    print(y[0].shape, y[1].shape)