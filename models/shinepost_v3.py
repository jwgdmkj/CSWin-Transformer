import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import einsum

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

import numpy as np
import math
import time

'''
Same, but changed F.unfold to torch.strides because F.unfold needs so much time.
Also, changed several einops function to torch tensor reshape function.
'''

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


# -------------------------------------------------------------------------------- #
## Third Module : Attn and PE
class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0.,
                 proj_drop=0., qk_scale=None, pos=2):
        '''
        x.shape = [896, 32, 56, 1]
        H = 56
        W = 56
        '''
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim

        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        # # print('num heads : ', self.num_heads)
        self.pos = pos
        head_dim = dim // num_heads

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # if -1 : last stage, no need to split
        # if 0 : x_axis(W), 1: y_axis(H), need to split. modify under code.
        self.idx = idx
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def unfold(self, k, v):
        # stt = time.time()
        # idx == 0 : x_axis / 1 : y_axis
        B, C, H, W = k.shape

        # kernel size
        if self.idx == 0:
            kernel_x, kernel_y = self.split_size * (self.pos * 2 + 1), self.split_size

        else:
            kernel_x, kernel_y = self.split_size, self.split_size * (self.pos * 2 + 1)

        # stride
        strd = self.split_size

        # calculate the number of patches in height and width
        num_patches_height = (H - kernel_y) // strd + 1
        num_patches_width = (W - kernel_x) // strd + 1

        # Shape of the output tensor
        out_shape = (B, C, num_patches_height, num_patches_width, kernel_y, kernel_x)

        # Strides for the output tensor
        strides = (k.stride(0),
                   k.stride(1),
                   k.stride(2) * strd,
                   k.stride(3) * strd,
                   k.stride(2),
                   k.stride(3))

        k_, v_ = k.as_strided(out_shape, strides), v.as_strided(out_shape, strides)

        # reshape, but v : [bhw win c]
        each_c = C // self.num_heads
        k_ = k_.reshape(B, self.num_heads, each_c, num_patches_height, num_patches_width, kernel_y, kernel_x) \
            .permute(0, 3, 4, 1, 5, 6, 2).reshape(B * num_patches_height * num_patches_width * self.num_heads,
                                                  kernel_x * kernel_y, each_c)

        v_ = v_.permute(0, 2, 3, 4, 5, 1).reshape(B * num_patches_height * num_patches_width,
                                                  kernel_y * kernel_x, C)
        k_, v_ = k_.reshape(-1, kernel_y * kernel_x, each_c), v_.reshape(-1, kernel_y * kernel_x, C)

        # etm = time.time()
        # print(f"my fold time : {etm - stt:.5f} sec")
        return k_, v_

    def get_lepe(self, x, func):
        '''
        input x(v) = [16, 3136, 32]
        1) channel is go to 2nd dim by transpose/view -> [16, 32, 56, 56]
        2) 3136(N) divide into H W, and H W divide into split_size
        3) Number of window combined with Batch, so if [56, 1] then [16 * 1 * 56, 32, 56, 1] = [896, 32, 56, 1]
           =[Bhw, C, Hsp, Wsp] (H=Hsp*h)
        4) [896, 32, 56, 1] pass through func(get_lepe) --> still [896, 32, 56, 1]
        5) reshape x and lepe into [Bhw, head_num, C//head_num, Hsp * Wsp] and permute
          --> [Bhw, head_num, Hsp*Wsp, C//head_num]
        output x, lepe : [896, 1, 56, 2]
        '''

        B_, N_, C_ = x.shape

        if self.idx == 0:  # x_axis
            x = x.transpose(-2, -1).contiguous().view(B_, C_,
                                                      self.split_size, self.split_size * (2 * self.pos + 1))
            lepe = x[:, :, :, self.split_size * self.pos: (self.pos + 1) * self.split_size]
        else:
            x = x.transpose(-2, -1).contiguous().view(B_, C_,
                                                      self.split_size * (2 * self.pos + 1), self.split_size)
            lepe = x[:, :, self.split_size * self.pos: (self.pos + 1) * self.split_size, :]
        lepe = func(lepe)

        # y : height, x : width
        bb, cc, yy, xx = x.shape
        x = x.reshape(bb, self.num_heads, cc // self.num_heads, yy, xx).\
            permute(0, 1, 3, 4, 2).\
            reshape(bb * self.num_heads, yy * xx, -1)
        bb, cc, yy, xx = lepe.shape
        lepe = lepe.reshape(bb, self.num_heads, cc // self.num_heads, yy, xx). \
            permute(0, 1, 3, 4, 2). \
            reshape(bb * self.num_heads, yy * xx, -1)

        return x, lepe

    def forward(self, qkv):
        """
        input qkv : [3, 16, 3136, 48] = [3, B, HW, C/2(=c)]
        1) after rearrange, q:[bhw p1p2 C] / k, v : [b h w C]
        2) after pad/unfold, k, v : [b*h*w, window_size, C]
        3) modify q and k : [b*h*w*num_head, window_size, c(=C//num_head)]
        """
        # print('-------------------------------- 3) LePE Attn -------------------------------------- ')

        B_, N_, C_ = qkv.shape[1], qkv.shape[2], qkv.shape[3]
        H = W = int(math.sqrt(N_))

        # Implement qkv, here, divide into self.dim//2 because original input x is divided into x1/x2
        qkv = qkv.reshape(3, B_, H, W, C_)
        q_, k_, v_ = qkv[0], qkv[1], qkv[2]

        # change shape of k and v -> need to padding and unfold
        q_ = q_.reshape(B_, H // self.split_size, self.split_size, W // self.split_size, self.split_size, C_). \
            permute(0, 1, 3, 2, 4, 5). \
            reshape(-1, self.split_size * self.split_size, C_)
        k_ = k_.permute(0, 3, 1, 2)
        v_ = v_.permute(0, 3, 1, 2)


        # idx == 0 -> x_axis / 1 -> y_axis
        if self.idx == 0:
            # # print('idx is ', self.idx, ' x axis')
            k_ = F.pad(k_, (self.split_size * self.pos, self.split_size * self.pos, 0, 0))
            v_ = F.pad(v_, (self.split_size * self.pos, self.split_size * self.pos, 0, 0))
            k_2, v_2 = self.unfold(k_, v_)

        else:
            # # print('idx is ', self.idx, ' y axis')
            k_ = F.pad(k_, (0, 0, self.split_size * self.pos, self.split_size * self.pos))
            v_ = F.pad(v_, (0, 0, self.split_size * self.pos, self.split_size * self.pos))
            k_2, v_2 = self.unfold(k_, v_)

        # Divide Embedding into Head
        bb, nn, dd = q_.shape
        q = q_.reshape(bb, nn, self.num_heads, dd//self.num_heads).permute(0, 2, 1, 3).\
            reshape(-1, nn, dd//self.num_heads)

        # v에 get_lepe 또는 q에 rel_pos_emb 더하기
        v, lepe = self.get_lepe(v_2, self.get_v)

        attn_stt = time.time()
        q *= self.scale

        # Attn 구하기
        # sim = einsum('b i d, b j d -> b i j', q, k)
        sim = (q @ k_2.transpose(-2, -1))

        attn = sim.softmax(dim=-1)
        # out = einsum('b i j, b j d -> b i d', attn, v)
        out = (attn @ v) + lepe

        # merge and combine heads
        bb, nn, dd = out.shape
        out = out.reshape(bb // self.num_heads, self.num_heads, nn, dd).\
            permute(0, 2, 1, 3).\
            reshape(-1, nn, self.num_heads * dd)

        # merge blocks back to original feature map
        out = out.reshape(B_, H // self.split_size, W // self.split_size, self.split_size, self.split_size, -1). \
            permute(0, 1, 3, 2, 4, 5). \
            reshape(B_, H // self.split_size * self.split_size * W // self.split_size * self.split_size, C_)

        return out


# -------------------------------------------------------------------------------- #
## 2nd module : Block
'''
변경점 : split_size와 pos를 넣어야 함 + num_layers, window_size 추가 필요
'''


class myAttnBlock(nn.Module):
    '''
    if not last stage, should divide into half channels.
    if last stage, not have to divide, bc see all region(7x7)

    original :
    dim, reso, num_heads, pos=1, split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
    attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, last_stage=False, idx = 1
    '''

    def __init__(self, dim, num_heads, pos, reso, mlp_ratio,
                 qkv_bias, qk_scale, split_size, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False, idx=1):
        # # print('MyAttnBlock : ', c1, c2, num_layers, pos, window_size)
        super().__init__()

        # variance
        self.dim = dim
        self.num_heads = num_heads

        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        self.pos = pos
        self.patches_resolution = reso
        self.split_size = split_size

        # branch
        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        # PE
        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, pos=pos)
                for i in range(self.branch_num)])
        else:  # if stage == 1 or 2 or 3
            # if stage == 1 : dim = 64 / patches_resolution = 56 / branch_num = 2 / split_size = 1/  num_heads = 2
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, pos=pos)
                for i in range(self.branch_num)])

        # function
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        '''
        input : [B, N^2, embed] = [B, HW, C]
        1) permute and view : [B, HW, C] (=[B, -1, C] 이후 norm)
        2) qkv : [3, B, HW, C]
        2-1) Attn input : [3, B, HW, C//2]
        3) Attend_x : cat(x1, x2) : [B, HW, C]

        0.0025초 ~ 0.018(0.27)초
        '''
        # # print('---------------------- 2) SP blaock ------------------------ ')
        # # print('SPB input : ', x.shape)

        # Padding ----------------------------------------
        B, N, C = x.shape
        H_ = W_ = int(math.sqrt(N))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H_, w=W_)
        assert x.shape[2] >= self.split_size, 'window should be less than feature map size'
        assert x.shape[3] >= self.split_size, 'window should be less than feature map size'

        # H, W,가 window_size의 배수가 아닐 경우, 패딩
        Padding = False
        if min(H_, W_) < self.split_size or H_ % self.split_size != 0 or W_ % self.split_size != 0:
            # # print('padding condition : ', H_, W_, self.split_size)
            Padding = True
            # # print(f'img_size {min(H_, W_)} is less than (or not divided by) split_size {self.split_size}, Padding.')
            pad_r = (self.split_size - W_ % self.split_size) % self.split_size
            pad_b = (self.split_size - H_ % self.split_size) % self.split_size
            x = F.pad(x, (0, pad_r, 0, pad_b))
        # Padding End ------------------------------------

        _, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # 우선 qkv를 만든 다음, channel을 1/2로 나눠, 하나는 가로방향, 하나는 세로방향으로 진행해야 됨.
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, H * W, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            # # print('branch num is 2!')
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)

        # proj, drop_path, norm2+mlp+drop_path
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # ------------------------------------------------------------------------- #

        # change shape into 4 size, [batch, embed, height, width]
        x = x.permute(0, 2, 1).contiguous()

        x = x.view(-1, C, H, W)  # b c h w

        # reverse padding
        if Padding:
            x = x[:, :, :H_, :W_]
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, H_ * W_, C)
        return x


# -------------------------------------------------------------------------------- #
## Block Merging
class Merge_Block(nn.Module):
    '''
    chan : 64 128 256 512로 2배화
    h, w :  56, 28, 14, 7로 1/2화(3136/784/196을 sqrt 해서 56, 28, 14로 만든 후 conv를 통해 1/2로) + norm
    '''

    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)

        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x


# -------------------------------------------------------------------------------- #
## 1st Module
class MyTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dim=96, depth=[2, 2, 6, 2],
                 split_size=[2, 2, 4, 7], num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 use_chk=False, pos=[2, 2, 2, 0]):
        '''
        CSWin block을 통과하기 전, embedding 진행 : Conv2d(채널 늘리기) + Rearrange(
        stage1: 3 to 64, kernel/stride/padding = 7/4/2 -> 56x56 생성
        stage2: 64 to 128, 28x28
        stage3: 128 to 256, 14x14
        stage4: 256 to 512, 7x7

        depth : [1, 2, 21, 1]
        split_size : [2, 2, 7, 7](original cswin : [1, 2, 7, 7])
        pos : [2, 2, 2, 0]
        img_size : 224
        embed_dim : 64
        num_heads : [2, 4, 8, 16]

        last stage : full region을 볼 것이기에, pos=0
        '''
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads

        # # print('myT heads : ', heads)
        # ------------------------ Stage 1 init ------------------------ #
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),  # channel 3 to 64, kernel/stride/padding = 7/4/2
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dim)
        )
        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            myAttnBlock(
                dim=curr_dim, num_heads=heads[0], pos=pos[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, idx=1)
            for i in range(depth[0])])

        # ------------------------ Stage 2 init ------------------------ #
        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList([
            myAttnBlock(
                dim=curr_dim, num_heads=heads[1], pos=pos[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer, idx=2)
            for i in range(depth[1])])
        # # print('stage 2 : ', curr_dim, heads[1], img_size // 8, split_size[1])

        # ------------------------ Stage 3 init ------------------------ #
        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        temp_stage3 = []
        temp_stage3.extend([
            myAttnBlock(
                dim=curr_dim, num_heads=heads[2], pos=pos[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer, idx=3)
            for i in range(depth[2])])
        # # print('stage 3 : ', curr_dim, heads[2], img_size // 16, split_size[-1])
        self.stage3 = nn.ModuleList(temp_stage3)

        # ------------------------ Stage 4 init ------------------------ #
        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage4 = nn.ModuleList([
            myAttnBlock(
                dim=curr_dim, num_heads=heads[3], pos=pos[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True, idx=4)
            for i in range(depth[-1])])
        # # print('stage 4 : ', curr_dim, heads[3], img_size // 32, split_size[-1])

        self.norm = norm_layer(curr_dim)
        # Classifier head
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.head.weight, std=0.02)

    def forward_features(self, x):
        '''
        input : [Batch(16), 3, 224, 224]
        after embed : [16, 3136(56^2), 64]
        '''
        B = x.shape[0]

        # # print('SP input  Shape: ', x.shape)
        # stage1 을 먼저 실행
        x = self.stage1_conv_embed(x)
        # # print('SP after embed : ', x.shape)
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # # print('ShinePost stage1 output: ', x.shape)

        # 총 4개의 Stage 존재, 각각 depth개 존재
        # (merge1, stage2), ... , (merge3, stage4) 실행
        for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
                               [self.stage2, self.stage3, self.stage4]):
            # # print('ShinePost zip step1 : ', x.shape)
            x = pre(x)
            # # print('ShinePost zip step2 : ', x.shape)
            for blk in blocks:
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            # # print('ShinePost zip step3 : ', x.shape)
            # # print('-------------------------------------------')
        x = self.norm(x)
        # # print('ShinePost final shape : ', x.shape)
        x = torch.mean(x, dim=1)
        # # print('ShinePost final mean : ', x.shape)

        return x

    def forward(self, x):
        '''
        배치 16 기준 0.1초 / CSwin 기준 0.6초
        '''
        stt = time.time()

        # # print('-------------------------- 1) ShinePost Start -----------------------------------')
        x = self.forward_features(x)
        # # print('ShinePost tf output : ', x.shape)
        x = self.head(x)
        # # print('ShinePost final output : ', x.shape)
        # # print('-------------------------1) ShinePost over --------------------------------------')
        # # print()

        edt = time.time()
        # # print(f"{edt - stt:.5f} sec")
        return x

#######################################
# -------------------------------------------------------------------------------- #
## Third Module : Attn and PE

# class myAttention(nn.Module):
#     def __init__(self, dim, num_heads, window_size, pos, norm_layer=nn.LayerNorm, bias=False,
#                  drop_path=0., mlp_ratio=4., act_layer=nn.GELU, drop=0.):
#         super().__init__()
#
#         # init parameter
#         self.dim = dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.pos = pos # see how much more windows?
#
#         # kernel size and scale
#         self.kernelX = (window_size, window_size * (self.pos * 2 + 1))
#         self.kernelY = (window_size * (self.pos * 2 + 1), window_size)
#
#         # head variance, head_dim split by 2 because x divided into x1 and x2
#         head_dim = (dim / 2) // num_heads
#         self.scale = head_dim ** -0.5
#
#         # linear to make QKV
#         self.norm1 = norm_layer(dim)
#         self.qkv = nn.Linear(dim, dim * 3, bias)
#         self.get_v = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2)
#
#         # Position Embedding
#         # self.rel_pos_emb = RelPosEmb(
#         #     block_size=window_size,
#         #     rel_size=window_size * 3,
#         #     dim_head=head_dim
#         # )
#
#         # Function after calculate QKV
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.proj = nn.Linear(dim, dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
#                        drop=drop)
#         self.norm2 = norm_layer(dim)
#
#     def get_lepe(self, x, func, flag):
#         '''
#         input : [Bhw, H/hw * W/ws, c]
#         input must be [B, C, H, W] at func
#         1) after transpose & contiguous : [Bhw c H/hw W/ws]
#         2) so, after func, x : [Bhw c H/hw W/ws]
#         3) lepe must be same size with qktv : [Bhw, H/ws * W/ws,
#
#         이를 chan을 head로 나누고 그 head를 배치로 넣어야 함. 또한 HW 한꺼번에 만들어야 함.
#         '''
#         # # print('5) get_lepe input : ', x.shape)
#
#         B_, N_, C_ = x.shape
#
#         if flag == 'x_axis':
#             x = x.transpose(-2, -1).contiguous().view(B_, C_,
#                                                       self.window_size, self.window_size * (2*self.pos+1))
#             lepe = x[:, :, :, self.window_size * self.pos: (self.pos + 1) * self.window_size]
#         else:
#             x = x.transpose(-2, -1).contiguous().view(B_, C_,
#                                                       self.window_size * (2*self.pos+1), self.window_size)
#             lepe = x[:, :, self.window_size * self.pos: (self.pos + 1) * self.window_size, :]
#         lepe = func(lepe)
#         # # print('6) lepe and x after func : ', x.shape, lepe.shape)
#
#         # y : height, x : width
#         x, lepe = map(lambda t: rearrange(t, 'b (h c) y x -> (b h) (y x) c', h=self.num_heads), (x, lepe))
#         # # print('7) get_lepe output x lepe ', x.shape, lepe.shape)
#         # lepe를 reshape 해야 함.
#         return x, lepe
#
#     def Attn(self, x, flag, H, W):
#         '''
#         Input x : [3, B, HW, C/2(=c)]
#         1) q_, k_, v_ = [B, HW, c]
#         2-1) q_ : [Bhw, H/ws * W/ws, c] such as [16, 4, 2]
#         2-2) k_/v_ : [Bhw, H/ws * W/ws, c] (단, padding값도 H/ws, W/ws에 포함) such as [Bx4x4, 12(2x6), 2(chan)]
#         3) q, k, v after map(lambda t ... ) : [Bhw * head, H/ws * W/ws, c]
#         '''
#         # # print('3) Attn input - ', x.shape)
#         B_, N_, C_ = x.shape[1], x.shape[2], x.shape[3]
#
#         # Implement qkv, here, divide into self.dim//2 because original input x is divided into x1/x2
#         q_, k_, v_ = x[0], x[1], x[2]
#         q_ = rearrange(q_, 'b (h w) c -> b h w c', h=H, w=W)
#         q_ = rearrange(q_, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)
#
#         k_ = rearrange(k_, 'b (h w) c -> b h w c', h=H, w=W).contiguous().permute(0, 3, 1, 2)
#         v_ = rearrange(v_, 'b (h w) c -> b h w c', h=H, w=W).contiguous().permute(0, 3, 1, 2)
#
#         if flag == 'x_axis':
#             # # print('-----------x axis-------------')
#             k_ = F.pad(k_, (self.window_size * self.pos, self.window_size * self.pos, 0, 0))
#             k_ = F.unfold(k_, kernel_size=(self.window_size, self.window_size * (self.pos * 2 + 1)),
#                           stride=self.window_size)
#             v_ = F.pad(v_, (self.window_size * self.pos, self.window_size * self.pos, 0, 0))
#             v_ = F.unfold(v_, kernel_size=(self.window_size, self.window_size * (self.pos * 2 + 1)),
#                           stride=self.window_size)
#         else:
#             # # print('----------y axis--------------')
#             k_ = F.pad(k_, (0, 0, self.window_size * self.pos, self.window_size * self.pos))
#             k_ = F.unfold(k_, kernel_size=(self.window_size * (self.pos * 2 + 1), self.window_size),
#                           stride=self.window_size)
#             v_ = F.pad(v_, (0, 0, self.window_size * self.pos, self.window_size * self.pos))
#             v_ = F.unfold(v_, kernel_size=(self.window_size * (self.pos * 2 + 1), self.window_size),
#                           stride=self.window_size)
#
#         k_ = rearrange(k_, 'b (c j) i -> (b i) j c', c=self.dim // 2)
#         v_ = rearrange(v_, 'b (c j) i -> (b i) j c', c=self.dim // 2)
#         # # print('q_ k_ v_ : ', q_.shape, k_.shape, v_.shape, self.num_heads)
#
#         # Divide Embedding into Head
#         q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.num_heads), (q_, k_))
#         q *= self.scale
#         # # print('4) q k v_ - ', q.shape, k.shape, v_.shape)
#
#         # v에 get_lepe 또는 q에 rel_pos_emb 더하기
#         v, lepe = self.get_lepe(v_, self.get_v, flag)
#
#         # Attn 구하기
#         sim = einsum('b i d, b j d -> b i j', q, k)
#
#         # ------------ Halo Attn Mask & Position Embedding Start -------- #
#         # sim += self.rel_pos_emb(q)
#         #
#         # # mask out padding (in the paper, they claim to not need masks, but what about padding?)
#         # device = x.device
#         # mask = torch.ones(1, 1, H, W, device=device)
#         # mask = F.unfold(mask, kernel_size=self.window_size * 3, stride=self.window_size, padding=self.window_size)
#         # # # print('mask unfold and block halo ', mask.shape, block, halo)
#         # mask = repeat(mask, '() j i -> (b i h) () j', b=B_, h=self.num_heads)
#         # mask = mask.bool()
#         #
#         # max_neg_value = -torch.finfo(sim.dtype).max
#         #
#         # # https://thought-process-ing.tistory.com/79
#         # # sim의 바꾸고자 하는 값(mask)를 max_neg_value로 변경
#         # sim.masked_fill_(mask, max_neg_value)
#         # ------------ Halo Attn Mask & Position Embedding End----------- #
#
#         attn = sim.softmax(dim=-1)
#         out = einsum('b i j, b j d -> b i d', attn, v)
#         # # print('8) qktv - ', out.shape)
#         out = out + lepe
#
#         # merge and combine heads
#         out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)
#
#         # merge blocks back to original feature map
#         out = rearrange(out, '(b h w) (p1 p2) c -> b (h p1) (w p2) c', b=B_, h=(H // self.window_size),
#                         w=(W // self.window_size), p1=self.window_size, p2=self.window_size)
#         out = out.reshape(B_, -1, C_)
#
#         # # print('9) Attn output : ', out.shape)
#         # # print('------------------------')
#         return out
#
#     def forward(self, x):
#         '''
#         input x : [B, C, H, W]
#         1) permute and view : [B, HW, C] (=[B, -1, C] 이후 norm)
#         2) qkv : [3, B, HW, C]
#         2-1) Attn input : [3, B, HW, C//2]
#         3) Attend_x : cat(x1, x2) : [B, HW, C]
#         '''
#         B_, C_, H_, W_ = x.shape
#
#         assert H_ >= self.window_size, 'window should be less than feature map size'
#         assert W_ >= self.window_size, 'window should be less than feature map size'
#
#         # H, W,가 window_size의 배수가 아닐 경우, 패딩
#         Padding = False
#         if min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ % self.window_size != 0:
#             # # print('padding condition : ', H_, W_, self.split_size)
#             Padding = True
#             # # print(f'img_size {min(H_, W_)} is less than (or not divided by) split_size {self.split_size}, Padding.')
#             pad_r = (self.window_size - W_ % self.window_size) % self.window_size
#             pad_b = (self.window_size - H_ % self.window_size) % self.window_size
#             x = F.pad(x, (0, pad_r, 0, pad_b))
#         # # print('X after padding : ', x.shape)
#
#         B, C, H, W = x.shape
#
#         x = x.permute(0, 2, 3, 1).contiguous().view(B_, H * W, C)
#
#         # # print('1) x [B HW C] - ', x.shape)
#
#         # 우선 qkv를 만든 다음, channel을 1/2로 나눠, 하나는 가로방향, 하나는 세로방향으로 진행해야 됨.
#         img = self.norm1(x)
#         qkv = self.qkv(img).reshape(B_, H * W, 3, C).permute(2, 0, 1, 3)  # [3, 1, 64, 4]
#         # # print('2) qkv ', qkv.shape)
#
#         x1 = self.Attn(qkv[:, :, :, :C // 2], 'x_axis', H, W)  # x-axis such as (2, 6). [3, 1, 64, 2]
#         x2 = self.Attn(qkv[:, :, :, C // 2:], 'y_axis', H, W)  # y-axis such as (6, 2). [3, 1, 64, 2]
#
#         attened_x = torch.cat([x1, x2], dim=2)
#         attened_x = self.proj(attened_x)
#         x = x + self.drop_path(attened_x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         # change shape into 4 size, [batch, embed, height, width]
#         x = x.permute(0, 2, 1).contiguous()
#
#         # # print('x shape after permute : ', x.shape)
#         x = x.view(-1, C, H, W)  # b c h w
#
#         # # print(Padding)
#         # # print(x.shape)
#         # reverse padding
#         if Padding:
#             x = x[:, :, :H_, :W_]
#
#         # # print('10) Final output : ', x.shape)
#         return x

#######################################
