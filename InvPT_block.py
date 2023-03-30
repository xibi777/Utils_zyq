import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
from einops import rearrange as o_rearrange
import collections.abc as container_abcs
from itertools import repeat
import torch.nn.functional as F

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

class UpEmbed(nn.Module):

    def __init__(self,
                 patch_size,
                 embed_dim,
                 in_chans=3,
                 patch_stride=4,
                 patch_padding=2,
                 scale_factor=2,
                 **kwargs,
                 ):
        super().__init__()
        to_2tuple = self._ntuple(2)
        patch_size = to_2tuple(patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.patch_size = patch_size
        self.proj_1 = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                    nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, padding=patch_padding, stride=patch_stride,
                                    bias=False, dilation=patch_padding))

        self.proj_2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, padding=patch_padding,
                                stride=patch_stride, bias=False, dilation=patch_padding)


    def _ntuple(self, n):
        def parse(x):
            if isinstance(x, container_abcs.Iterable):
                return x
            return tuple(repeat(x, n))

        return parse

    def forward(self, x):
        # h, w = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_1(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        h, w = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.relu(x)
        x = self.proj_2(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        h, w = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.relu(x)
        return x

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
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


class SelfAttention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_method='dw_bn',
                 kv_method='avg',
                 kernel_size_q=3,
                 kernel_size_kv=2,
                 stride_kv=2,
                 stride_q=2,
                 padding_kv=0,
                 padding_q=1,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5

        self.conv_proj_q = self._build_projection(
            dim_in, kernel_size_q, padding_q,
            stride_q, q_method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, kv_method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, kv_method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        self.fuse_attn = nn.Conv2d(num_heads * 2, num_heads, 1)
        self.norm = nn.LayerNorm(dim_in)

    def _build_projection(self,
                          dim_in,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                ),
                Rearrange('b c h w -> b (h w) c'))

        elif method == 'avg':
            proj = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                ),
                Rearrange('b c h w -> b (h w) c'))

        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    # def split_x(self, x, h, w):
    #     res = h*w
    #     x_list = []
    #     for i in range(self.fea_no):
    #         _x = rearrange(x[:, res*i:res*(i+1), :], 'b (h w) c -> b c h w', h=h, w=w)
    #         x_list.append(_x)
    #     return x_list

    def forward_conv(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
            q = self.norm(q)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        return q, k, v

    def forward(self, x, h, w, messages):
        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale

        # attention message passing with upsampling
        if messages['attn'] != None:
            res_attn_score = messages['attn']

            sh = h // 4  # source h
            sw = w // 4  # source w
            bs, heads, _, _ = res_attn_score.shape

            # separate task
            res_attn_score_list = []
            res = sh * sw
            for i in range(1):
                _x = res_attn_score[:, :, res * i:res * (i + 1), :]
                _x = rearrange(_x, 'b h (m n) a -> (b h) a m n', m=sh, n=sw)
                _x = F.interpolate(_x, scale_factor=2, mode='bilinear', align_corners=False)
                _x = rearrange(_x, '(b h) a m n -> b h (m n) a', b=bs, h=heads)
                res_attn_score_list.append(_x)
            res_attn_score = torch.cat(res_attn_score_list, dim=2)

            # fuse both the attention map from last stage and current stage
            multiScaleAttention = torch.cat([attn_score, res_attn_score], dim=1)
            # attn_score += res_attn_score
            attn_score = self.fuse_attn(multiScaleAttention)
        messages['attn'] = attn_score

        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class InvPTBlock(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio,
                 qkv_bias,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.stride_q = 2
        self.embed_dim = kwargs['embed_dim']

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(self.embed_dim)
        self.norm2 = norm_layer(self.embed_dim)
        dim_mlp_hidden = int(self.embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )
        self.attn = SelfAttention(
            self.embed_dim, self.embed_dim, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.patch_embed = UpEmbed(
            **kwargs
        )

    def split_x(self, x, h, w):
        res = h * w
        x_list = []
        for i in range(self.task_no):
            _x = x[:, res * i:res * (i + 1), :]
            x_list.append(_x)
        return x_list

    def forward(self, x, messages):
        if messages['attn'] != None:
            x = self.patch_embed(x)
        h, w = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))
        res = x
        x = self.norm1(x)
        attn = self.attn(x, h, w, messages)

        # interpolate output of attention to previous resolution
        sh, sw = h // self.stride_q, w // self.stride_q
        attn = rearrange(attn, 'b (h w) c -> b c h w', h=sh, w=sw)
        attn = F.interpolate(attn, size=(h, w), mode='bilinear', align_corners=False)
        attn = rearrange(attn, 'b c h w -> b (h w) c')


        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, messages


class InStage(nn.Module):
    def __init__(
            self,
            img_size,
            in_channels,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=3,
            use_proj=True,
            CE_loss=False,
            crop_train=False,
            **kwargs,
    ):
        super(InStage, self).__init__(
            in_channels=in_channels, **kwargs)
        blocks = []
        depth = 3
        spec = {
            'ori_embed_dim': embed_dims,
            'NUM_STAGES': 3,
            'PATCH_SIZE': [0, 3, 3],
            'PATCH_STRIDE': [0, 1, 1],
            'PATCH_PADDING': [0, 2, 2],
            'DIM_EMBED': [embed_dims, embed_dims // 2, embed_dims // 4],
            'NUM_HEADS': [2, 2, 2],
            'MLP_RATIO': [4., 4., 4.],
            'DROP_PATH_RATE': [0.15, 0.15, 0.15],
            'QKV_BIAS': [True, True, True],
            'KV_PROJ_METHOD': ['avg', 'avg', 'avg'],
            'KERNEL_KV': [2, 4, 8],
            'PADDING_KV': [0, 0, 0],
            'STRIDE_KV': [2, 4, 8],
            'Q_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
            'KERNEL_Q': [3, 3, 3],
            'PADDING_Q': [1, 1, 1],
            'STRIDE_Q': [2, 2, 2],
        }
        for i in range(depth):
            kwargs.update({
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'in_chans': spec['DIM_EMBED'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': 0,
                'attn_drop_rate': 0,
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'q_method': spec['Q_PROJ_METHOD'][i],
                'kv_method': spec['KV_PROJ_METHOD'][i],
                'kernel_size_q': spec['KERNEL_Q'][i],
                'kernel_size_kv': spec['KERNEL_KV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            })
            blocks.append(
                InvPTBlock(
                    dim_in=384,
                    dim_out=384,
                    drop=0,
                    attn_drop=0,
                    drop_path=0.0,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    **kwargs
                )
            )
        self.InvPT_blocks = nn.ModuleList(blocks)
        self.linear_depth = nn.Conv2d(96, 1, kernel_size=(1, 1), stride=(1, 1))