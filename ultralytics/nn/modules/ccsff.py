import torch
import math
import einops
import torch.nn as nn
import copy
import torch.nn.functional as F
from ..modules.block import BasicBlock
from .block import Faster_Block
from collections import OrderedDict
from timm.layers import DropPath, to_2tuple, trunc_normal_

__all__ = 'CrossLayerChannelWiseFrequencyAttention', 'CrossLayerSpatialWiseFrequencyAttention'

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class CrossLayerPosEmbedding3D(nn.Module):
    def __init__(self, num_heads=4, window_size=(5, 3, 1), spatial=True):
        super(CrossLayerPosEmbedding3D, self).__init__()
        self.spatial = spatial
        self.num_heads = num_heads
        self.layer_num = len(window_size)
        if self.spatial:
            self.num_token = sum([i ** 2 for i in window_size])    # 不同尺度拼接后每个窗口中token的个数:35
            self.num_token_per_level = [i ** 2 for i in window_size]  # 不同尺度每个窗口的token个数[25,9,1]
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), num_heads))
            coords_h = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_w = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_h = [coords_h[i] * window_size[0] / window_size[i] for i in range(len(coords_h) - 1)] + [
                coords_h[-1]]
            coords_w = [coords_w[i] * window_size[0] / window_size[i] for i in range(len(coords_w) - 1)] + [
                coords_w[-1]]
            coords = [torch.stack(torch.meshgrid([coord_h, coord_w])) for coord_h, coord_w in
                      zip(coords_h, coords_w)]
            coords_flatten = torch.cat([torch.flatten(coord, 1) for coord in coords], dim=-1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[0] - 1
            relative_coords[:, :, 0] *= 2 * window_size[0] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)
        else:
            self.num_token = sum([i for i in window_size])
            self.num_token_per_level = [i for i in window_size]
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), num_heads))
            coords_c = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_c = [coords_c[i] * window_size[0] / window_size[i] for i in range(len(coords_c) - 1)] + [
                coords_c[-1]]
            coords = torch.cat(coords_c, dim=0)
            coords_flatten = torch.stack([torch.flatten(coord, 0) for coord in coords], dim=-1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.absolute_position_bias = nn.Parameter(torch.zeros(len(window_size), num_heads, 1, 1, 1))
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        pos_indicies = self.relative_position_index.view(-1)
        pos_indicies_floor = torch.floor(pos_indicies).long()
        pos_indicies_ceil = torch.ceil(pos_indicies).long()

        pos_indicies_floor = torch.clamp(pos_indicies_floor, 0, self.relative_position_bias_table.size(0) - 1)
        pos_indicies_ceil = torch.clamp(pos_indicies_ceil, 0, self.relative_position_bias_table.size(0) - 1)

        value_floor = self.relative_position_bias_table[pos_indicies_floor]
        value_ceil = self.relative_position_bias_table[pos_indicies_ceil]
        weights_ceil = pos_indicies - pos_indicies_floor.float()
        weights_floor = 1.0 - weights_ceil

        pos_embed = weights_floor.unsqueeze(-1) * value_floor + weights_ceil.unsqueeze(-1) * value_ceil
        pos_embed = pos_embed.reshape(1, 1, self.num_token, -1, self.num_heads).permute(0, 4, 1, 2, 3)

        pos_embed = pos_embed.split(self.num_token_per_level, 3)
        layer_embed = self.absolute_position_bias.split([1 for i in range(self.layer_num)], 0)
        pos_embed = torch.cat([i + j for (i, j) in zip(pos_embed, layer_embed)], dim=-2)
        return pos_embed


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=True):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        feat = self.proj(x)
        x = x + self.activation(feat)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def overlaped_window_partition(x, window_size, stride, pad):
    B, C, H, W = x.shape
    out = torch.nn.functional.unfold(x, kernel_size=(window_size, window_size), stride=stride, padding=pad)
    return out.reshape(B, C, window_size * window_size, -1).permute(0, 3, 2, 1)   # (B,400,w_s * w_s,c)


def overlaped_window_reverse(x, H, W, window_size, stride, padding):
    B, Wm, Wsm, C = x.shape
    Ws, S, P = window_size, stride, padding
    x = x.permute(0, 3, 2, 1).reshape(B, C * Wsm, Wm)  # (B, C * kH * kW, L)
    out = torch.nn.functional.fold(x, output_size=(H, W), kernel_size=(Ws, Ws), padding=P, stride=S)
    return out  # (B,C,H,W)


# x:(B,HW,C,())
def overlaped_channel_partition(x, window_size, stride, pad):
    B, HW, C, _ = x.shape
    # out:(B,HW*window_s,L)
    out = torch.nn.functional.unfold(x, kernel_size=(window_size, 1), stride=(stride, 1), padding=(pad, 0))
    out = out.reshape(B, HW, window_size, -1)
    return out  # (B, HW, window_size, 3 * hidden_dim)


def overlaped_channel_reverse(x, window_size, stride, pad, outC):
    B, C, Ws, HW = x.shape
    x = x.permute(0, 3, 2, 1).reshape(B, HW * Ws, C)
    out = torch.nn.functional.fold(x, output_size=(outC, 1), kernel_size=(window_size, 1), padding=(pad, 0),
                                   stride=(stride, 1))
    return out


class CrossLayerSpatialWiseFrequencyAttention(nn.Module):
    def __init__(self, in_dim, layer_num=3, beta=1, num_heads=4, mlp_ratio=2, reduction=4):
        super(CrossLayerSpatialWiseFrequencyAttention, self).__init__()
        assert beta % 2 != 0, "error, beta must be an odd number!"
        self.num_heads = num_heads
        self.reduction = reduction
        # window_sizes = [5,3,1]
        self.window_sizes = [(2 ** i + beta) if i != 0 else (2 ** i + beta - 1) for i in range(layer_num)][::-1]
        self.token_num_per_layer = [i ** 2 for i in self.window_sizes]
        self.token_num = sum(self.token_num_per_layer)

        self.stride_list = [2 ** i for i in range(layer_num)][::-1]  # [4,2,1]
        self.padding_list = [[0, 0] for i in self.window_sizes]
        self.shape_list = [[0, 0] for i in range(layer_num)]

        self.hidden_dim = in_dim // reduction
        self.head_dim = self.hidden_dim // num_heads

        self.cpe = nn.ModuleList(
            nn.ModuleList([ConvPosEnc(dim=in_dim, k=3),
                           ConvPosEnc(dim=in_dim, k=3)])
            for i in range(layer_num)
        )

        self.norm1 = nn.ModuleList(LayerNormProxy(in_dim) for i in range(layer_num))
        self.norm2 = nn.ModuleList(nn.LayerNorm(in_dim) for i in range(layer_num))
        self.qkv = nn.ModuleList(
            nn.Conv2d(in_dim, self.hidden_dim * 3, kernel_size=1, stride=1, padding=0)
            for i in range(layer_num)
        )

        mlp_hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.ModuleList(
            Mlp(
                in_features=in_dim,
                hidden_features=mlp_hidden_dim)
            for i in range(layer_num)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.ModuleList(
            nn.Conv2d(self.hidden_dim, in_dim, kernel_size=1, stride=1, padding=0) for i in range(layer_num)
        )

        # self.h_pos_embed = CrossLayerPosEmbedding3D(num_heads=num_heads, window_size=self.window_sizes, spatial=True)
        self.fa = Frequencyattention(in_dim, self.hidden_dim)
        
    def forward(self, x_list, extra=None):
        WmH, WmW = x_list[-1].shape[-2:]
        shortcut_list, x_patch_list = [], []

        for i, x in enumerate(x_list):  # x:(B,in_dim,h,w)
            B, C, H, W = x.shape
            ws_i, stride_i = self.window_sizes[i], self.stride_list[i]
            # 这里计算的pad是为了让不同尺度下抽出的patch个数都为20x20
            pad_i = (math.ceil((stride_i * (WmH - 1.) - H + ws_i) / 2.), math.ceil((stride_i * (WmW - 1.) - W + ws_i) / 2.))
            self.padding_list[i] = pad_i  # [(1, 1), (1, 1), (0, 0)]
            self.shape_list[i] = [H, W]
            x = self.cpe[i][0](x)
            shortcut_list.append(x)
            x_patch = overlaped_window_partition(x, ws_i, stride=stride_i, pad=pad_i)  # (B,400,w_s*w_s,in_dim)
            x_patch_list.append(x_patch)  # [(B,400,patch_size1,in_dim),(B,400,patch_size2,in_dim),(B,400,patch_size3,in_dim)]

        out = self.fa(x_patch_list)  # (B,400,35,hidden_dim)

        out_split = out.split(self.token_num_per_layer, dim=-2)
        out_list = []  # [(B, 400, 25, hidden_dim), (B, 400, 9, hidden_dim), (B, 400, 1, hidden_dim)]
        for i, out_i in enumerate(out_split):
            ws_i, stride_i, pad_i = self.window_sizes[i], self.stride_list[i], self.padding_list[i]
            H, W = self.shape_list[i]

            out_i = overlaped_window_reverse(out_i, H, W, ws_i, stride_i, pad_i)  # out_i:(B,hidden_dim,H,W)
            out_i = shortcut_list[i] + self.norm1[i](self.proj[i](out_i))  # out_i:(B,in_dim,H,W)
            out_i = self.cpe[i][1](out_i)
            out_i = out_i.permute(0, 2, 3, 1)  # out_i:(B,H,W,in_dim)
            out_i = out_i + self.mlp[i](self.norm2[i](out_i))
            out_i = out_i.permute(0, 3, 1, 2)  # (B,in_dim,H,W)
            out_list.append(out_i)
        return out_list


class CrossLayerChannelWiseFrequencyAttention(nn.Module):
    def __init__(self, in_dim, layer_num=3, alpha=1, num_heads=4, mlp_ratio=2, reduction=4):
        super(CrossLayerChannelWiseFrequencyAttention, self).__init__()
        assert alpha % 2 != 0, "error, alpha must be an odd number!"
        self.num_heads = num_heads
        self.reduction = reduction
        self.hidden_dim = in_dim // reduction
        self.in_dim = in_dim
        # window_sizes=[17,5,1]
        self.window_sizes = [(4 ** i + alpha) if i != 0 else (4 ** i + alpha - 1) for i in range(layer_num)][::-1]
        self.token_num_per_layer = [i for i in self.window_sizes]  # [17,5,1]
        self.token_num = sum(self.token_num_per_layer)  # 23

        self.stride_list = [(4 ** i) for i in range(layer_num)][::-1]  # [16,4,1]
        self.padding_list = [0 for i in self.window_sizes]  # [0 0 0]
        self.shape_list = [[0, 0] for i in range(layer_num)]
        self.unshuffle_factor = [(2 ** i) for i in range(layer_num)][::-1]  # [4, 2, 1]

        self.cpe = nn.ModuleList(
            nn.ModuleList([ConvPosEnc(dim=in_dim, k=3),
                           ConvPosEnc(dim=in_dim, k=3)])
            for i in range(layer_num)
        )
        self.norm1 = nn.ModuleList(LayerNormProxy(in_dim) for i in range(layer_num))
        self.norm2 = nn.ModuleList(nn.LayerNorm(in_dim) for i in range(layer_num))


        self.qkv = nn.ModuleList(
            nn.Conv2d(in_dim, self.hidden_dim * 3, kernel_size=1, stride=1, padding=0)
            for i in range(layer_num)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.ModuleList(nn.Conv2d(self.hidden_dim, in_dim, kernel_size=1, stride=1, padding=0) for i in range(layer_num))

        mlp_hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.ModuleList(
            Mlp(
                in_features=in_dim,
                hidden_features=mlp_hidden_dim)
            for i in range(layer_num)
        )
        self.fa = Frequencyattention(self.in_dim, self.hidden_dim, True)
        
    def forward(self, x_list, extra=None):   # [(256, 80, 80), (256, 40, 40), (256, 20, 20)]
        shortcut_list, reverse_shape = [], []
        # q_list, k_list, v_list = [], [], []
        qkv_list = []
        for i, x in enumerate(x_list):
            B, C, H, W = x.shape
            self.shape_list[i] = [H, W]
            ws_i, stride_i = self.window_sizes[i], self.stride_list[i]  # 17,16

            pad_i = math.ceil((stride_i * (self.hidden_dim - 1.) - (self.unshuffle_factor[i])**2 * self.hidden_dim + ws_i) / 2.)
            self.padding_list[i] = pad_i  # [1, 1, 0]
            x = self.cpe[i][0](x)
            shortcut_list.append(x)

            qkv = self.qkv[i](x)  # (hidden_dim * 3,h,w)

            qkv = F.pixel_unshuffle(qkv, downscale_factor=self.unshuffle_factor[i])  # (qkv_out*16,h/4,w/4)
            reverse_shape.append(qkv.size(1) // 3)

            qkv_window = einops.rearrange(qkv, "b c h w -> b (h w) c ()")  # (b,h/4 * w/4, qkv_out*16, ())

            qkv_window = overlaped_channel_partition(qkv_window, ws_i, stride=stride_i, pad=pad_i)

            qkv_window = einops.rearrange(qkv_window, "b hw wsm (n nh c) -> n b nh c wsm hw", n=3, nh=self.num_heads)
            qkv_list.append(qkv_window)

        out = self.fa(qkv_list)  # (B,num_heads,head_dim=patch_num,patch_size1+2+3,400)
        out = einops.rearrange(out, "b nh c ws hw -> b (nh c) ws hw")  # (b,hidden_dim,23,h*w)
        out_split = out.split(self.token_num_per_layer, dim=-2)
        out_list = []  # [(B, h_d, 17, h*w), (B, h_d, 5, h*w), (B, h_d, 1, h*w)]
        for i, out_i in enumerate(out_split):
            ws_i, stride_i, pad_i = self.window_sizes[i], self.stride_list[i], self.padding_list[i]
            out_i = overlaped_channel_reverse(out_i, ws_i, stride_i, pad_i, outC=reverse_shape[i])
            out_i = out_i.permute(0, 2, 1, 3).reshape(B, -1, self.shape_list[-1][0], self.shape_list[-1][1])
            out_i = F.pixel_shuffle(out_i, upscale_factor=self.unshuffle_factor[i])

            out_i = shortcut_list[i] + self.norm1[i](self.proj[i](out_i))
            out_i = self.cpe[i][1](out_i)
            out_i = out_i.permute(0, 2, 3, 1)
            out_i = out_i + self.mlp[i](self.norm2[i](out_i))
            out_i = out_i.permute(0, 3, 1, 2)
            out_list.append(out_i)

        return out_list


class Frequencyattention(nn.Module):
    def __init__(self, in_dim, dim, channel=False, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.75, layer_num=3):
        super().__init__()  
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.head_dim = int(dim/num_heads)
        self.dim = dim
        self.in_dim = in_dim
        self.channel = channel


        self.l_heads = int(num_heads * alpha)

        self.l_dim = self.l_heads * self.head_dim


        self.h_heads = num_heads - self.l_heads

        self.h_dim = self.h_heads * self.head_dim

        # local window size. The `s` in our paper.
        self.ws = window_size

        if self.channel:
            self.h_pos_embed = CrossLayerPosEmbedding3D(self.h_heads, window_size=[17, 5, 1], spatial=False)
            self.l_pos_embed = CrossLayerPosEmbedding3D(self.l_heads, window_size=[17, 5, 1], spatial=False)
        else:
            self.h_pos_embed = CrossLayerPosEmbedding3D(num_heads=self.h_heads, spatial=True)
            self.l_pos_embed = CrossLayerPosEmbedding3D(num_heads=self.l_heads, spatial=True)

        if self.ws == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or self.head_dim ** -0.5

        # Low frequence attention
        if self.l_heads > 0:
            if self.ws != 1:
                if self.channel:
                    self.sr = nn.AvgPool1d(kernel_size=2, stride=2, padding=1, count_include_pad=False)
                    self.l_proj = nn.Linear(400, 400)
                else:
                    self.sr = nn.AvgPool2d(kernel_size=2, stride=2, padding=1, count_include_pad=False)
                    self.l_q = nn.ModuleList(
                        nn.Linear(self.in_dim, self.l_dim, bias=qkv_bias) for i in range(layer_num))
                    self.l_kv = nn.Linear(self.in_dim, self.l_dim * 2, bias=qkv_bias)
                    self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # High frequence attention
        if self.h_heads > 0:
            if self.channel:
                self.h_proj = nn.Linear(400, 400)
            else:
                self.h_qkv = nn.ModuleList(
                    nn.Linear(self.in_dim, self.h_dim * 3, bias=qkv_bias)
                    for i in range(layer_num)
                )
                self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    # channel:False x_list:[(B,patch_num,patch_size1/2/3,in_dim)]
    # channel:True x_list:[(3,B,h_heads,head_dim=patch_num,patch_token1/2/3,400)]
    def hifi(self, x_list):  # (B,H,W,C)
        if self.channel:
            q_list, k_list, v_list = [], [], []
            for x in x_list:
                q_list.append(x[0])
                k_list.append(x[1])
                v_list.append(x[2])
            q_stack = torch.cat(q_list,dim=-2)   # (B,h_heads,head_dim=patch_num,patch_size1+2+3,400)
            k_stack = torch.cat(k_list, dim=-2)
            v_stack = torch.cat(v_list, dim=-2)
            # (B,h_heads,head_dim=patch_num,patch_size1+2+3,patch_size1+2+3)
            attn = F.normalize(q_stack, dim=-1) @ F.normalize(k_stack, dim=-1).transpose(-1, -2)
            attn = attn + self.h_pos_embed()
            attn = attn.softmax(dim=-1)
            out = attn.to(v_stack.dtype) @ v_stack 
            out = self.h_proj(out)  # (B,h_heads,head_dim=patch_num,patch_size1+2+3,400)
            return out        
        else:
            token_num = sum([x.shape[2] for x in x_list])
            B = x_list[0].shape[0]
            q_list, k_list, v_list = [], [], []
            for i, x in enumerate(x_list):
                # qkv:(B,400,patch_size3,h_dim*3)
                qkv = self.h_qkv[i](x).reshape(B, 400, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 4, 1, 2, 5)
                q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h_heads, 400, patch_size, head_dim)
                q_list.append(q)
                k_list.append(k)
                v_list.append(v)
            q_stack = torch.cat(q_list, dim=-2)   # (B,h_heads,400,patch_size1+2+3,head_dim)
            k_stack = torch.cat(k_list, dim=-2)
            v_stack = torch.cat(v_list, dim=-2)


            attn = F.normalize(q_stack, dim=-1) @ F.normalize(k_stack, dim=-1).transpose(-1, -2)
            attn = attn + self.h_pos_embed()
            attn = attn.softmax(dim=-1)
            out = attn.to(v_stack.dtype) @ v_stack   # (B,h_head,400,35,head_dim)
            # 将多头concat  out:(B, 400, 35, h_dim)
            out = out.permute(0, 2, 3, 1, 4).reshape(B, 400, token_num, self.h_dim)

            x = self.h_proj(out)
            return x  # x:(B,400,35,h_dim)


    # channel:False x_list:[(B,400,patch_token1/2/3,in_dim)]
    # channel:True x_list:[(3,B,l_heads,head_dim=patch_num,patch_token1/2/3,400)]
    def lofi(self, x_list):
        if self.channel:
            B = x_list[0].shape[1]
            q_list, kv_list, kv_pool = [], [], []
            for x in x_list:
                q_list.append(x[0])
                kv_list.append(x[1:])
            for i, kv in enumerate(kv_list):
                if i != 2:
                    kv = kv.permute(0, 1, 2, 3, 5, 4).reshape(2*B*self.l_heads*self.head_dim, 400, -1)
                    kv = self.sr(kv).permute(0, 2, 1).reshape(2, B, self.l_heads, self.head_dim, -1, 400)
                kv_pool.append(kv)
            q_stack = torch.cat(q_list, dim=-2)
            kv_pool_stack = torch.cat(kv_pool, dim=-2)  # (2,B,l_heads,head_dim=patch_num,patch_token1+2+3,400)
            k_stack, v_stack = kv_pool_stack[0], kv_pool_stack[1]  # (B,l_heads,head_dim=patch_num,pool_patch_token1+2+3,400)
            # (B,l_heads,head_dim=patch_num,patch_token1+2+3,pool_patch_token1+2+3)
            attn = F.normalize(q_stack, dim=-1) @ F.normalize(k_stack, dim=-1).transpose(-1, -2)
            l_pose_embed = self.l_pos_embed()
            layer1_pose = l_pose_embed[:, :, :, :, :17].reshape(self.l_heads, 23, 17)
            layer2_pose = l_pose_embed[:, :, :, :, 17:22].reshape(self.l_heads, 23, 5)
            layer3_pose = l_pose_embed[:, :, :, :, -1, None]
            layer1_pose_pool = self.sr(layer1_pose).reshape(1, self.l_heads, 1, 23, -1)
            layer2_pose_pool = self.sr(layer2_pose).reshape(1, self.l_heads, 1, 23, -1)
            l_pose_embed = torch.cat([layer1_pose_pool, layer2_pose_pool, layer3_pose], dim=-1)
            attn = attn + l_pose_embed
            attn = attn.softmax(dim=-1)
            out = attn.to(v_stack.dtype) @ v_stack  
            out = self.l_proj(out) # (B,l_heads,head_dim=patch_num,patch_size1+2+3,400)
            return out
        else:
            B = x_list[0].shape[0]
            token_num = sum([x.shape[2] for x in x_list])
            patch_h_w = [5, 3, 1]
            in_dim = x_list[0].shape[-1]
            q_list, x_pool = [], []
            for i, x in enumerate(x_list):
                # (B,l_heads,400,patch_token,head_dim)
                q = self.l_q[i](x).reshape(B, 400, -1, self.l_heads, self.l_dim // self.l_heads).permute(0, 3, 1, 2, 4)
                q_list.append(q)
            q_stack = torch.cat(q_list, dim=-2)  # (B,l_heads,400,patch_token1+2+3,head_dim)

            for i, x in enumerate(x_list):
                if i != 2:
                    # (B,400,in_dim,5,5) (B,400,in_dim,3,3)
                    x = x.permute(0, 1, 3, 2).reshape(B*400, in_dim, patch_h_w[i], patch_h_w[i])
                    x = self.sr(x).reshape(B, 400, in_dim, -1).permute(0, 1, 3, 2)  # (B,400,pool_token,in_dim)
                x_pool.append(x)
            x_pool_stack = torch.cat(x_pool, dim=-2)  # (B,400,pool_token1+2+3,in_dim)

            l_pose_embed = self.l_pos_embed()
            layer1_pose = l_pose_embed[:, :, :, :, : 25].reshape(self.l_heads, 35, 5, 5)
            layer2_pose = l_pose_embed[:, :, :, :, 25: 34].reshape(self.l_heads, 35, 3, 3)
            layer3_pose = l_pose_embed[:, :, :, :, -1, None]
            layer1_pose_pool = self.sr(layer1_pose).reshape(1, self.l_heads, 1, 35, -1)
            layer2_pose_pool = self.sr(layer2_pose).reshape(1, self.l_heads, 1, 35, -1)
            l_pose_embed = torch.cat([layer1_pose_pool, layer2_pose_pool, layer3_pose], dim=-1)

            kv = self.l_kv(x_pool_stack).reshape(B, 400, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(3, 0, 4, 1, 2, 5)
            # (2,B,l_head,400,pool_token1+2+3,head_dim)
            k, v = kv[0], kv[1]  # (B,l_head,400,pool_token1+2+3,head_dim)

            attn = F.normalize(q_stack, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2)
            # attn = (q_stack @ k.transpose(-2, -1)) * self.scale
            attn = attn + l_pose_embed 
            attn = attn.softmax(dim=-1)  # (B,l_heads,400,patch_token1+2+3,pool_token1+2+3)

            # (B,400,patch_token1+2+3,l_heads*head_dim)
            x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B, 400, token_num, -1)
            x = self.l_proj(x)
            return x  # (B,400,35,l_dim)

    # channel:False x_list:[(B,400,patch_size1,in_dim),(B,400,patch_size2,in_dim),(B,400,patch_size3,in_dim)]
    # channel:True x_list:[(3,B,num_head,head_dim=patch_num,patch_token1/2/3,400)]
    def forward(self, x_list):  
        if self.channel:
            x_list_hi, x_list_lo = [], []
            for x in x_list:
                x_list_lo.append(x[:, :, :self.l_heads, :, :, :])
                x_list_hi.append(x[:, :, self.l_heads:, :, :, :])
            hifi_out = self.hifi(x_list_hi)
            lofi_out = self.lofi(x_list_lo)
            x = torch.cat((lofi_out, hifi_out), dim=1) # (B,num_heads,head_dim=patch_num,patch_size1+2+3,400)
            return x
        else:
            hifi_out = self.hifi(x_list)  # (B,400,35,h_dim)
            lofi_out = self.lofi(x_list)  # (B,400,35,l_dim)
            x = torch.cat((hifi_out, lofi_out), dim=-1)  # (B,400,35,in_dim)
            return x
