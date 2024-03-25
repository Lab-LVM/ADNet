from functools import partial

import math
import torch
from timm import create_model
from torch import nn
import torch.nn.functional as F

from timm.models import ResNet, register_model
from timm.models.layers import DropPath, trunc_normal_
from timm.models.resnet import Bottleneck


class ConvBNAct(nn.Sequential):
    """Convolution-Normalization-Activation Module"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups, norm_layer, act, conv_layer=nn.Conv2d):
        super(ConvBNAct, self).__init__(
            conv_layer(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False),
            norm_layer(out_channel),
            act()
        )


class SEUnit(nn.Module):
    """Squeeze-Excitation Unit
    paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper
    """
    def __init__(self, in_channel, reduction_ratio=4, act1=partial(nn.SiLU, inplace=True), act2=nn.Sigmoid):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio, dw=False):
        super().__init__()
        self.dw = dw
        if self.dw:
            self.fc1 = nn.Sequential(
                nn.Conv2d(dim, dim * mlp_ratio, 1),
                nn.GELU(),
                nn.BatchNorm2d(dim * mlp_ratio),
            )
            self.dw_conv = nn.Sequential(
                nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, 1, 1, groups=dim * mlp_ratio),
            )
            self.fc2 = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm2d(dim * mlp_ratio),
                nn.Conv2d(dim * mlp_ratio, dim, 1),
                nn.BatchNorm2d(dim),
            )
        else:
            self.fc1 = nn.Linear(dim, dim * mlp_ratio)
            self.fc2 = nn.Linear(dim * mlp_ratio, dim)
            self.gelu = nn.GELU()

    def forward(self, x):
        if self.dw:
            b, n, c = x.shape
            h = w = int(n ** 0.5)
            x = x.permute(0, 2, 1).reshape(b, c, h, w)
            out = self.fc1(x)
            out = self.dw_conv(out) + out
            out = self.fc2(out)
            out = out.reshape(b, c, n).permute(0, 2, 1)
            return out
        else:
            out = self.fc1(x)
            out = self.gelu(out)
            out = self.fc2(out)
            return out


class MHCA(nn.Module):
    def __init__(self, q_dim, kv_dim, out_dim, head, groups=1, k=1, qkv_bias=False, attn_drop=0.0):
        super().__init__()
        self.k = out_dim // head
        self.div = math.sqrt(self.k)
        self.head = head
        self.groups = groups

        if self.groups > 1:
            self.q = nn.Conv2d(q_dim, out_dim, k, padding=int(k//2), groups=groups, bias=qkv_bias)
            self.kv = nn.Conv2d(kv_dim, out_dim * 2, k, padding=int(k//2), groups=groups, bias=qkv_bias)
        else:
            self.q = nn.Linear(q_dim, out_dim, bias=qkv_bias)
            self.kv = nn.Linear(kv_dim, out_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(out_dim, out_dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, kv):
        if self.groups > 1:
            B, N1, C = q.shape
            B, N2, C2 = kv.shape
            h = w = int(N1 ** 0.5)
            h2 = w2 = int(N2 ** 0.5)

            q = q.permute(0, 2, 1).reshape(B, C, h, w)
            kv = kv.permute(0, 2, 1).reshape(B, C2, h2, w2)

            q = self.q(q)
            kv = self.kv(kv)
            k, v = kv.tensor_split(2, dim=1)

            q = q.reshape(B, C, N1).permute(0, 2, 1).reshape(B, N1, self.head, self.k).permute(0, 2, 1, 3)
            k = k.reshape(B, -1, N2).permute(0, 2, 1).reshape(B, -1, self.head, self.k).permute(0, 2, 1, 3)
            v = v.reshape(B, -1, N2).permute(0, 2, 1).reshape(B, -1, self.head, self.k).permute(0, 2, 1, 3)
        else:
            (B, N1, _), N2 = q.shape, kv.size(1)
            q = self.q(q).reshape(B, N1, self.head, self.k).permute(0, 2, 1, 3)
            k, v = [x.reshape(B, N2, self.head, self.k).permute(0, 2, 1, 3) for x in self.kv(kv).tensor_split(2, dim=-1)]

        attn = q @ k.transpose(-1, -2) / self.div
        attn_prob = F.softmax(attn, dim=-1)
        attn_prob = self.attn_drop(attn_prob)

        out = attn_prob @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N1, -1)
        out = self.proj(out)

        return out


class CrossAttention(nn.Module):
    """Cross Attention

    Details: drop_path_rate is only applied to this module
    """

    def __init__(self, q_dim, kv_dim, out_dim, mlp_ratio, head, groups=1, k=1, dw=False,
                 qkv_bias=False, attn_drop=0.0, drop_path_rate=0.0):
        super().__init__()
        self.add_q = q_dim == out_dim
        self.add_kv = kv_dim == out_dim
        self.attn = MHCA(q_dim, kv_dim, out_dim, head, groups, k, qkv_bias, attn_drop)
        self.mlp = MLP(out_dim, mlp_ratio, dw)
        self.norm1_q = nn.LayerNorm(q_dim, eps=1e-6)
        self.norm1_kv = nn.LayerNorm(kv_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(out_dim, eps=1e-6)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, q, kv):
        x = self.drop_path(self.attn(self.norm1_q(q), self.norm1_kv(kv)))

        if self.add_q:
            x = x + q
        if self.add_kv:
            b, n, c = kv.shape
            h = w = int(n ** 0.5)
            kv = kv.permute(0, 2, 1).reshape(b, c, h, w)
            kv = F.interpolate(kv, scale_factor=2, mode='bilinear')
            kv = kv.reshape(b, c, h * 2 * w * 2).permute(0, 2, 1)
            x = x + kv

        x = self.drop_path(self.mlp(self.norm2(x))) + x

        return x


class DFTTransform(nn.Module):
    """Discrete Fourier Transformation (DFT)
    paper: DFT-based Transformation Invariant Pooling Layer for Visual Classification, ECCV, 2018
    """
    def __init__(self, crop=None):
        super().__init__()
        self.crop = crop

    def forward(self, x):
        # if x.size(-1) < (self.crop[0] - 1) * 2:
        #     x = F.interpolate(x, (self.crop[0] - 1) * 2, mode='bilinear')

        x = x.to(dtype=torch.float, memory_format=torch.contiguous_format)
        out = torch.fft.rfft2(x, norm='forward')
        real = out.real
        imag = out.imag
        magnitude = out.abs()
        phase = out.angle()

        if self.crop:
            # if real.size(-1) < self.crop[-1]:
            #     real = F.pad(real, (0, self.crop[-1] - real.size(-1), 0, self.crop[-2] - real.size(-2)))
            #
            # if imag.size(-1) < self.crop[-1]:
            #     imag = F.pad(imag, (0, self.crop[-1] - imag.size(-1), 0, self.crop[-2] - imag.size(-2)))
            #
            # if magnitude.size(-1) < self.crop[-1]:
            #     magnitude = F.pad(magnitude, (0, self.crop[-1] - magnitude.size(-1), 0, self.crop[-2] - magnitude.size(-2)))
            #
            # if phase.size(-1) < self.crop[-1]:
            #     phase = F.pad(phase, (0, self.crop[-1] - phase.size(-1), 0, self.crop[-2] - phase.size(-2)))

            real = real[:,:,:self.crop[0],: self.crop[1]]
            imag = imag[:,:,:self.crop[0],: self.crop[1]]
            magnitude = magnitude[:,:,:self.crop[0],: self.crop[1]]
            phase = phase[:,:,:self.crop[0],: self.crop[1]]

        return real, imag, magnitude, phase


class HarmonicMagnitudePooling(nn.Module):
    def __init__(self, in_ch, crop=None, mul=1, dft_target='mag'):
        super(HarmonicMagnitudePooling, self).__init__()

        if isinstance(crop, (tuple, list)) and len(crop) == 1:
            crop = (crop[0], crop[0])
        elif not isinstance(crop, (tuple, list)):
            crop = (crop, crop)

        self.dft_target = dft_target
        self.dft = DFTTransform(crop)
        self.mul = mul

        self.spatial_encode = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * mul, crop, bias=False, groups=in_ch),
            nn.BatchNorm2d(in_ch * mul)
        )
        self.channel_encode = nn.Sequential(
            nn.Conv2d(in_ch * mul, in_ch, 1, bias=False, groups=1),
            nn.BatchNorm2d(in_ch),
        )
    def forward(self, x):
        real, imag, magnitude, phase = self.dft(x)

        if self.dft_target == 'mag':
            x = magnitude
        elif self.dft_target == 'pha':
            x = phase
        else:
            raise ValueError(f"Wrong dft_target: {self.dft_target}")


        out = self.spatial_encode(x)
        if self.mul > 1:
            out = self.channel_encode(out)
        else:
            out = out + self.channel_encode(out)
        out = torch.flatten(out, start_dim=1)

        return out


class DFTMagnitudePooling(nn.Module):
    def __init__(self, in_ch, crop=None, dft_target='mag', pool_channel=None):
        super(DFTMagnitudePooling, self).__init__()

        if isinstance(crop, (tuple, list)) and len(crop) == 1:
            crop = (crop[0], crop[0])
        elif not isinstance(crop, (tuple, list)):
            crop = (crop, crop)

        self.dft_target = dft_target
        self.dft = DFTTransform(crop)
        self.pool_channel = pool_channel

        if self.pool_channel:
            hidden_ch = pool_channel
            self.ch_reduce = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
            )
            self.fc = nn.Sequential(
                nn.Conv2d(hidden_ch, hidden_ch, crop, bias=False),
                nn.BatchNorm2d(hidden_ch),
            )
            self.ch_expand = nn.Sequential(
                nn.Conv2d(hidden_ch, in_ch, 1, bias=False),
            )
        else:
            self.fc = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, crop, bias=False),
                nn.BatchNorm2d(in_ch)
            )

    def forward(self, x):
        if self.pool_channel:
            x = self.ch_reduce(x)

        real, imag, magnitude, phase = self.dft(x)

        if self.dft_target == 'mag':
            x = magnitude
        elif self.dft_target == 'pha':
            x = phase
        else:
            raise ValueError(f"Wrong dft_target: {self.dft_target}")

        out = self.fc(x)

        if self.pool_channel:
            out = self.ch_expand(out)

        out = torch.flatten(out, start_dim=1)

        return out


class VGG(nn.Module):
    def __init__(self, in_ch, size=28, pool_channel=None):
        super().__init__()
        self.size = size
        self.pool_channel = pool_channel

        if self.pool_channel:
            hidden_ch = pool_channel
            self.ch_reduce = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
            )
            self.fc = nn.Sequential(
                nn.Conv2d(hidden_ch, hidden_ch, size, bias=False),
                nn.BatchNorm2d(hidden_ch),
            )
            self.ch_expand = nn.Sequential(
                nn.Conv2d(hidden_ch, in_ch, 1, bias=False),
            )
        else:
            self.fc = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, size, bias=False),
                nn.BatchNorm2d(in_ch)
            )

    def forward(self, x):
        if self.pool_channel:
            x = self.ch_reduce(x)

        out = self.fc(x)

        if self.pool_channel:
            out = self.ch_expand(out)

        out = torch.flatten(out, start_dim=1)

        return out


class ResUNet(ResNet):
    def __init__(self, *args,
                 up_level=None, up_layer_type='attn', up_group=1, keep_dim=False,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
                 last_global_pool='gap', crop_dim=None, mul=1, mlp=3, groups=1, k=1, dw=False,
                 pool_channel=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.channel = [2048, 1024, 512, 256, 64]
        self.up_level = up_level
        self.up_layer_type = up_layer_type
        self.pretrained_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
        self.crop_dim = crop_dim

        if self.up_level:
            if self.up_layer_type == 'attn':
                self.attn_decoder = nn.ModuleList([])

                for level in range(self.up_level):
                    if keep_dim:
                        q_dim = self.channel[level+1]
                        kv_dim = self.channel[0]
                        out_dim = self.channel[0]
                        head = int(out_dim // 32)
                    else:
                        q_dim = self.channel[level+1]
                        kv_dim = self.channel[level]
                        out_dim = self.channel[level+1]
                        head = int(out_dim // 32)

                    cross_attention = CrossAttention(q_dim, kv_dim, out_dim, mlp, head, groups, k, dw)
                    self.attn_decoder.append(cross_attention)

            elif self.up_layer_type == 'conv':
                self.trans_conv_decoder = nn.ModuleList()
                self.conv_concat = nn.ModuleList()

                for level in range(self.up_level):
                    if keep_dim:
                        down_path_dim = self.channel[0]
                        up_path_dim = self.channel[level+1]
                        out_dim = self.channel[0]
                    else:
                        down_path_dim = self.channel[level]
                        up_path_dim = self.channel[level+1]
                        out_dim = self.channel[level+1]

                    trans_conv = nn.Sequential(
                        nn.ConvTranspose2d(down_path_dim, out_dim, 2, 2, groups=up_group, bias=False),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True)
                    )
                    concat_conv = nn.Sequential(
                        nn.Conv2d(up_path_dim + out_dim, out_dim, 1, bias=False),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True)
                    )
                    self.trans_conv_decoder.append(trans_conv)
                    self.conv_concat.append(concat_conv)

            elif self.up_layer_type == 'attn_conv':
                self.attn_decoder = nn.ModuleList([])
                self.trans_conv_decoder = nn.ModuleList()
                self.conv_concat = nn.ModuleList()

                for level in range(self.up_level):
                    if keep_dim:
                        q_dim = self.channel[level + 1]
                        kv_dim = self.channel[0]
                        out_dim = self.channel[0]
                        head = int(out_dim // 32)
                    else:
                        q_dim = self.channel[level + 1]
                        kv_dim = self.channel[level]
                        out_dim = self.channel[level + 1]
                        head = int(out_dim // 32)

                    cross_attention = CrossAttention(q_dim, kv_dim, out_dim, mlp, head, groups, k, dw)
                    trans_conv = nn.Sequential(
                        nn.ConvTranspose2d(kv_dim, out_dim, 2, 2, groups=up_group, bias=False),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True)
                    )
                    concat_conv = nn.Sequential(
                        nn.Conv2d(q_dim + out_dim * 2, out_dim, 1, bias=False),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True)
                    )
                    self.attn_decoder.append(cross_attention)
                    self.trans_conv_decoder.append(trans_conv)
                    self.conv_concat.append(concat_conv)

            self.fc_layer = ConvBNAct(out_dim, self.channel[0], 1, 1, up_group, norm_layer, act_layer)

        self.last_global_pool = last_global_pool
        self.pool_channel = pool_channel
        if last_global_pool == 'hmp':
            self.global_pool = HarmonicMagnitudePooling(self.channel[0], crop_dim, mul)
        elif last_global_pool == 'dft':
            self.global_pool = DFTMagnitudePooling(self.channel[0], crop_dim, pool_channel=pool_channel)
        elif last_global_pool == 'vgg':
            self.crop_dim = int((crop_dim - 1) * 2)
            self.global_pool = VGG(self.channel[0], self.crop_dim, pool_channel=pool_channel)

        self.init_weights()

    def load_state_dict(self, state_dict, strict: bool = True):
        key1 = 'fc_layer.0.weight'
        key2 = 'global_pool.spatial_encode.0.weight'
        key3 = 'global_pool.fc.0.weight'

        if key1 in state_dict and state_dict[key1].size(1) != int(2048 / (2 ** self.up_level)):
            fc_layer = state_dict[key1]
            org_shape = fc_layer.shape
            fc_layer = fc_layer.permute(2, 3, 0, 1).squeeze(0)
            fc_layer = F.interpolate(fc_layer, mode='linear', size=int(2048 / (2 ** self.up_level)))
            fc_layer = fc_layer.unsqueeze(0).permute(2, 3, 0, 1)
            new_shape = fc_layer.shape
            state_dict[key1] = fc_layer

            print(f"change fc_layer shape: {org_shape} -> {new_shape}")

        if key2 in state_dict and state_dict[key2].size(2) != self.crop_dim:
            sp_encode = state_dict[key2]
            org_shape = sp_encode.shape
            h, w = org_shape[2:]
            sp_encode = F.pad(sp_encode, (0, max(self.crop_dim - h, 0), 0, max(self.crop_dim - w, 0)))
            sp_encode = sp_encode[:, :, :self.crop_dim, :self.crop_dim]
            new_shape = sp_encode.shape
            state_dict[key2] = sp_encode

            print(f"change sp_encode shape: {org_shape} -> {new_shape}")

        if key3 in state_dict and state_dict[key3].size(2) != self.crop_dim:
            if self.last_global_pool == 'dft':
                sp_encode = state_dict[key3]
                org_shape = sp_encode.shape
                h, w = org_shape[2:]
                sp_encode = F.pad(sp_encode, (0, max(self.crop_dim - h, 0), 0, max(self.crop_dim - w, 0)))
                sp_encode = sp_encode[:, :, :self.crop_dim, :self.crop_dim]
                new_shape = sp_encode.shape
                state_dict[key3] = sp_encode
            else:
                sp_encode = state_dict[key3]
                org_shape = sp_encode.shape
                sp_encode = F.interpolate(sp_encode, self.crop_dim)
                new_shape = sp_encode.shape
                state_dict[key3] = sp_encode

            print(f"change sp_encode shape: {org_shape} -> {new_shape}")

        if self.last_global_pool in ['vgg', 'dft']:
            one_dim_key_list = [
                'global_pool.fc.1.weight', 'global_pool.fc.1.bias',
                'global_pool.fc.1.running_mean', 'global_pool.fc.1.running_var',
            ]
            two_dim_key_list = [
                ('global_pool.ch_reduce.0.weight', self.pool_channel, 2048),
                ('global_pool.fc.0.weight', self.pool_channel, self.pool_channel),
                ('global_pool.ch_expand.0.weight', 2048, self.pool_channel)
            ]

            for key in one_dim_key_list:
                weight = state_dict[key]
                org_shape = weight.shape
                weight = F.interpolate(weight[None, None, :], self.pool_channel).reshape(-1)
                new_shape = weight.shape
                state_dict[key] = weight

                # print(f"change {key} shape: {org_shape} -> {new_shape}")

            for key, out_ch, in_ch in two_dim_key_list:
                weight = state_dict[key]
                org_shape = weight.shape
                weight = weight.permute(2, 3, 0, 1)
                weight = F.interpolate(weight, (out_ch, in_ch))
                weight = weight.permute(2, 3, 0, 1)
                new_shape = weight.shape
                state_dict[key] = weight

                # print(f"change {key} shape: {org_shape} -> {new_shape}")

        return super().load_state_dict(state_dict, strict)

    def toggle_grad(self, flag):
        for name, param in self.named_parameters():
            if any([x in name for x in self.pretrained_layers]):
                param.requires_grad = flag

    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.act1(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = [x0, x1, x2, x3, x4]

        if self.up_level:
            if self.up_layer_type == 'attn':
                kv = features.pop()
                b, c, h, w = kv.shape
                kv = kv.permute(0, 2, 3, 1).reshape(b, h * w, c)

                for level in range(self.up_level):
                    q = features.pop()
                    b, c, h, w = q.shape
                    q = q.permute(0, 2, 3, 1).reshape(b, h * w, c)
                    kv = self.attn_decoder[level](q, kv)

                b, n, c = kv.shape
                h = w = int(n ** 0.5)
                out = kv.permute(0, 2, 1).reshape(b, c, h, w)

            elif self.up_layer_type == 'conv':
                x = features.pop()

                for level in range(self.up_level):
                    x = self.trans_conv_decoder[level](x)
                    skip = features.pop()
                    x = torch.cat([x, skip], dim=1)
                    x = self.conv_concat[level](x)

                out = x

            elif self.up_layer_type == 'attn_conv':
                x = features.pop()
                b, c, h, w = x.shape
                kv = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

                for level in range(self.up_level):
                    skip = features.pop()
                    b, c, h, w = skip.shape
                    q = skip.permute(0, 2, 3, 1).reshape(b, h * w, c)

                    kv = self.attn_decoder[level](q, kv)
                    b, n, c = kv.shape
                    h = w = int(n ** 0.5)
                    kv = kv.permute(0, 2, 1).reshape(b, c, h, w)

                    x = self.trans_conv_decoder[level](x)

                    x = torch.cat([skip, kv, x], dim=1)
                    x = self.conv_concat[level](x)

                    b, c, h, w = x.shape
                    kv = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

                out = x

            out = self.fc_layer(out)
        else:
            out = x4

        return out

########################
# index
# 1. t-conv - imagenet
# 2. t-conv - nih
# 3. gap - imagenet, nih, other
# 4. hmt - imagenet
# 5. hmt - nih, other
# 6. dft - imagenet
# 7. dft - nih
# 8. vgg - imagenet
# 9. vgg - nih
########################

# t-conv - imagenet
@register_model
def resnet50_up1_conv_hmp(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='conv',
                   last_global_pool='hmp', crop_dim=8,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_conv_hmp(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='conv',
                   last_global_pool='hmp', crop_dim=15,
                   num_classes=kwargs.get('num_classes', 1000))

# t-conv - nih
@register_model
def resnet50_up0_conv_hmp_crop6(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='conv',
                   last_global_pool='hmp', crop_dim=6,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_conv_hmp_crop9(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='conv',
                   last_global_pool='hmp', crop_dim=9,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_conv_hmp_crop12(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='conv',
                   last_global_pool='hmp', crop_dim=12,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_conv_hmp_crop17(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='conv',
                   last_global_pool='hmp', crop_dim=17,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_conv_hmp_crop23(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='conv',
                   last_global_pool='hmp', crop_dim=23,
                   num_classes=kwargs.get('num_classes', 1000))

# org attention model cost too many parameters!
# @register_model
# def resnet50_up1_attn_hmp(pretrained=False, **kwargs):
#     return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
#                    last_global_pool='hmp', crop_dim=8,
#                    num_classes=kwargs.get('num_classes', 1000))
#
# @register_model
# def resnet50_up2_attn_hmp(pretrained=False, **kwargs):
#     return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
#                    last_global_pool='hmp', crop_dim=15,
#                    num_classes=kwargs.get('num_classes', 1000))
# @register_model
# def resnet50_up2_attn_hmp_crop23(pretrained=False, **kwargs):
#     return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
#                    last_global_pool='hmp', crop_dim=23,
#                    num_classes=kwargs.get('num_classes', 1000))

# gap - imagenet
@register_model
def resnet50_up0_attn_gap_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='attn',
                   last_global_pool='gap', crop_dim=8, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_gap_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='gap', crop_dim=15, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_gap_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='gap', crop_dim=15, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

# hmp - imagenet
@register_model
def resnet50_up0_attn_hmp_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=4, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_hmp_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=8, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=15, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

# hmp - different dataset
@register_model
def resnet50_up0_attn_hmp_mlp2dw_group2_crop5(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=5, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up0_attn_hmp_mlp2dw_group2_crop6(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=6, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_hmp_mlp2dw_group2_crop8(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=8, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_hmp_mlp2dw_group2_crop9(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=9, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_hmp_mlp2dw_group2_crop12(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=12, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop3(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=3, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop5(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=5, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop7(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=7, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop9(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=9, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop11(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=11, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop13(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=13, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop15(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=15, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop17(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=17, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop19(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=19, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop21(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=21, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_hmp_mlp2dw_group2_crop23(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='hmp', crop_dim=23, mlp=2, groups=2, dw=True,
                   num_classes=kwargs.get('num_classes', 1000))

# dft - imagenet
@register_model
def resnet50_up0_attn_dft_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='attn',
                   last_global_pool='dft', crop_dim=4, mlp=2, groups=2, dw=True, pool_channel=1024,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_dft_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='dft', crop_dim=8, mlp=2, groups=2, dw=True, pool_channel=512,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_dft_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='dft', crop_dim=15, mlp=2, groups=2, dw=True, pool_channel=384,
                   num_classes=kwargs.get('num_classes', 1000))

# dft - nih
@register_model
def resnet50_up0_attn_dft_mlp2dw_group2_crop5(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='attn',
                   last_global_pool='dft', crop_dim=5, mlp=2, groups=2, dw=True, pool_channel=1024,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up0_attn_dft_mlp2dw_group2_crop6(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='attn',
                   last_global_pool='dft', crop_dim=6, mlp=2, groups=2, dw=True, pool_channel=1024,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_dft_mlp2dw_group2_crop9(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='dft', crop_dim=9, mlp=2, groups=2, dw=True, pool_channel=512,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_dft_mlp2dw_group2_crop12(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='dft', crop_dim=12, mlp=2, groups=2, dw=True, pool_channel=512,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_dft_mlp2dw_group2_crop17(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='dft', crop_dim=17, mlp=2, groups=2, dw=True, pool_channel=384,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_dft_mlp2dw_group2_crop23(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='dft', crop_dim=23, mlp=2, groups=2, dw=True, pool_channel=384,
                   num_classes=kwargs.get('num_classes', 1000))

# vgg - imagenet
@register_model
def resnet50_up0_attn_vgg_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='attn',
                   last_global_pool='vgg', crop_dim=4.5, mlp=2, groups=2, dw=True, pool_channel=512,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_vgg_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='vgg', crop_dim=8, mlp=2, groups=2, dw=True, pool_channel=384,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_vgg_mlp2dw_group2(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='vgg', crop_dim=15, mlp=2, groups=2, dw=True, pool_channel=256,
                   num_classes=kwargs.get('num_classes', 1000))

# vgg - nih
@register_model
def resnet50_up0_attn_vgg_mlp2dw_group2_crop5(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='attn',
                   last_global_pool='vgg', crop_dim=5, mlp=2, groups=2, dw=True, pool_channel=512,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up0_attn_vgg_mlp2dw_group2_crop6(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=0, up_layer_type='attn',
                   last_global_pool='vgg', crop_dim=6.5, mlp=2, groups=2, dw=True, pool_channel=512,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_vgg_mlp2dw_group2_crop9(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='vgg', crop_dim=9, mlp=2, groups=2, dw=True, pool_channel=384,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up1_attn_vgg_mlp2dw_group2_crop12(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=1, up_layer_type='attn',
                   last_global_pool='vgg', crop_dim=12, mlp=2, groups=2, dw=True, pool_channel=384,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_vgg_mlp2dw_group2_crop17(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='vgg', crop_dim=17, mlp=2, groups=2, dw=True, pool_channel=256,
                   num_classes=kwargs.get('num_classes', 1000))

@register_model
def resnet50_up2_attn_vgg_mlp2dw_group2_crop23(pretrained=False, **kwargs):
    return ResUNet(block=Bottleneck, layers=[3, 4, 6, 3], up_level=2, up_layer_type='attn',
                   last_global_pool='vgg', crop_dim=23, mlp=2, groups=2, dw=True, pool_channel=256,
                   num_classes=kwargs.get('num_classes', 1000))


if __name__ == '__main__':
    func_list = [
        # 'resnet50', 'resnet50_up0_attn_hmp_mlp2dw_group2_crop6',
        # 'resnet50_up1_conv_hmp_crop12', 'resnet50_up1_attn_hmp_mlp2dw_group2_crop12',
        # 'resnet50_up2_conv_hmp_crop23', 'resnet50_up2_attn_hmp_mlp2dw_group2_crop23',
        # 'resnet50_up0_attn_gap_mlp2dw_group2',
        # 'resnet50_up1_attn_gap_mlp2dw_group2',
        # 'resnet50_up2_attn_gap_mlp2dw_group2',
        'resnet50_up0_attn_vgg_mlp2dw_group2_crop6',
        'resnet50_up1_attn_vgg_mlp2dw_group2_crop12',
        'resnet50_up2_attn_vgg_mlp2dw_group2_crop23',
        # 'resnet50_up0_attn_dft_mlp2dw_group2_crop6',
        # 'resnet50_up1_attn_dft_mlp2dw_group2_crop12',
        # 'resnet50_up2_attn_dft_mlp2dw_group2_crop23',
    ]
    for func_name in func_list:
        x = torch.rand(2, 3, 352, 352)
        model = create_model(func_name, num_classes=14)
        print(func_name)
        print(sum(param.numel() for param in model.parameters() if param.requires_grad))
    # f = resnet50_up2_conv_hmp_crop23(num_classes=14)
    # for level in [0, 1, 2]:
    #     for pool in ['gap', 'vgg', 'dft', 'hmp']:
    #         print(f"level: {level}, pool:{pool}")
    #         exec(f'f = resnet50_up{level}_attn_{pool}_mlp2dw_group2(num_classes=1000)')
    #         y = f(x)
    #         print(y.shape)
    #         print(sum(param.numel() for param in f.parameters() if param.requires_grad))