import torch
import torch.nn as nn
import math
# from apex.normalization import FusedLayerNorm as LayerNorm
from torch.nn import LayerNorm


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.norm = LayerNorm(head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, before, after):
        B, N, C = before.shape
        qkv_b = self.qkv(before).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_b = self.norm(qkv_b)
        q_b, k_b, v_b = qkv_b[0], qkv_b[1], qkv_b[2]

        qkv_a = self.qkv(after).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_a = self.norm(qkv_a)
        q_a, k_a, v_a = qkv_a[0], qkv_a[1], qkv_a[2]

        attn_b_a = (q_b @ k_a.transpose(-2, -1)) * self.scale
        attn_b_a = attn_b_a.softmax(dim=-1)
        attn_b_a = self.attn_drop(attn_b_a)

        context_a = (attn_b_a @ v_a).transpose(1, 2).reshape(B, N, C)
        context_a = self.proj(context_a)
        context_a = self.proj_drop(context_a) + q_b.reshape(B, N, C)

        attn_a_b = (q_a @ k_b.transpose(-2, -1)) * self.scale
        attn_a_b = attn_a_b.softmax(dim=-1)
        attn_a_b = self.attn_drop(attn_a_b)

        context_b = (attn_a_b @ v_b).transpose(1, 2).reshape(B, N, C)
        context_b = self.proj(context_b)
        context_b = self.proj_drop(context_b) + q_a.reshape(B, N, C)

        # x: (B, N, C)
        return context_b, context_a


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, input):
        before_norm = self.norm1(input[0])
        after_norm = self.norm1(input[1])
        context_b, context_a = self.attn(before_norm, after_norm)
        before = before_norm + self.drop_path(context_b)
        after = after_norm + self.drop_path(context_a)
        before = before + self.drop_path(self.mlp(self.norm2(before)))
        after = after + self.drop_path(self.mlp(self.norm2(after)))
        return [before, after]


class PatchEmbed(nn.Module):
    """ Feature to Patch Embedding
    """

    def __init__(self, inchannel, embed_dim, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(inchannel, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchFeature(nn.Module):
    """ Patch Embedding to Feature
    """

    def __init__(self, inchannel, embed_dim, patch_size=16, bilinear=False):
        super().__init__()
        self.patch_size = patch_size
        self.bilinear = bilinear
        if bilinear:
            self.unsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False)
            self.conv = nn.Conv2d(embed_dim, inchannel, kernel_size=1, stride=1, bias=True)
        else:
            self.unsample = nn.ConvTranspose2d(embed_dim, inchannel, kernel_size=patch_size, stride=patch_size,
                                               padding=0)

    def forward(self, x, h, w):
        b, n, c = x.shape
        l = int(math.sqrt(n))
        x = x.transpose(1, 2).view(b, c, l, l).contiguous()
        if self.bilinear:
            x = self.unsample(x)
            x = self.conv(x)
        else:
            x = self.unsample(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, in_channel, embed_dim,
                 patch_size=16,
                 num_head=1,
                 cross_layers=1,
                 num_patches=1024,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.1,
                 attn_drop=0.1,
                 drop_path=0.1,
                 act_layer=nn.GELU,
                 norm_layer=LayerNorm):
        super(Cross_Attention, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.cross_layer = cross_layers
        self.time_token_b = torch.nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.time_token_a = torch.nn.Parameter(torch.ones(1, 1, embed_dim), requires_grad=True)
        self.embed = PatchEmbed(in_channel, embed_dim, patch_size)
        self.to_feature = PatchFeature(in_channel, embed_dim, patch_size, bilinear=True)
        blocks = []
        for _ in range(cross_layers):
            blocks.append(Block(embed_dim, num_head, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                drop_path, act_layer, norm_layer))
        self.atten = nn.Sequential(*blocks)

    def forward(self, before, after):
        b, _, h, w = before.size()
        residual_b = before
        residual_a = after
        before = self.embed(before)
        after = self.embed(after)
        time_token_b = self.time_token_b.expand(b, -1, -1)
        time_token_a = self.time_token_a.expand(b, -1, -1)
        before += time_token_b
        after += time_token_a

        if self.cross_layer > 0:
            output = self.atten([before, after])
            before = output[0]
            after = output[1]

        before = self.to_feature(before, h, w)
        after = self.to_feature(after, h, w)
        return before + residual_b, after + residual_a


if __name__ == '__main__':
    before = torch.zeros([2, 128, 64, 64])
    after = torch.zeros([2, 128, 64, 64])
    ca = Cross_Attention(128, 128, patch_size=2)
    before, after = ca(before, after)
