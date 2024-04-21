import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torchvision import ops
import numpy as np

from typing import Sequence, Literal, Optional
from functools import partial
from .utils.state import get_seq_len

training = False
# Since Pytorch's interleave is not supported by CoreML, we use this function instead for mobile conversion
def repeat_interleave(x, n):
    x = x.unsqueeze(2)
    x = x.repeat(1, 1, n, 1, 1)
    x = x.reshape(x.shape[0], x.shape[1] * n, x.shape[3], x.shape[4])
    return x


class CCM(nn.Sequential):
    " Convolutional Channel Mixer "
    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, 1, 0)
        )
        trunc_normal_(self[-1].weight, std=0.02)
        

class ICCM(nn.Sequential):
    " Inverted Convolutional Channel Mixer "
    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1)
        )
        trunc_normal_(self[-1].weight, std=0.02)
        

class DCCM(nn.Sequential):
    " Doubled Convolutional Channel Mixer "
    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1)
        )
        trunc_normal_(self[-1].weight, std=0.02)


class PLKConv2d(nn.Module):
    " Partial Large Kernel Convolutional Layer "
    def __init__(self, dim, kernel_size, with_idt):
        super().__init__()
        self.with_idt = with_idt
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim
        self.is_convert = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_convert:
            x[:, :self.idx] = self.conv(x[:, :self.idx])
            return x
            
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            if self.with_idt:
                x1 = self.conv(x1) + x1
            else:
                x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        else:
            if self.with_idt:
                x[:, :self.idx] = x[:, :self.idx] + self.conv(x[:, :self.idx])
            else:
                x[:, :self.idx] = self.conv(x[:, :self.idx])
            return x

    @torch.no_grad()
    def convert(self):
        if not self.is_convert:
            if self.with_idt:
                k = self.conv.weight
                n_pad = (k.shape[2] - 1) // 2
                I = torch.eye(k.shape[0]).reshape(k.shape[0], k.shape[0], 1, 1)
                I = F.pad(I, (n_pad, n_pad, n_pad, n_pad))
                k = k + I
                self.conv.weight.data.copy_(k)
            self.is_convert = True


class RectSparsePLKConv2d(nn.Module):
    " Rectangular Sparse Partial Large Kernel Convolutional Layer (SLaK style) "
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.idx = dim
        m = kernel_size
        n = kernel_size // 3
        self.mn_conv = nn.Conv2d(dim, dim, (m, n), 1, (m // 2, n // 2))
        self.nm_conv = nn.Conv2d(dim, dim, (n, m), 1, (n // 2, m // 2))
        self.nn_conv = nn.Conv2d(dim, dim, (n, n), 1, (n // 2, n // 2))
        
        trunc_normal_(self.mn_conv.weight, std=0.02)
        trunc_normal_(self.nm_conv.weight, std=0.02)
        trunc_normal_(self.nn_conv.weight, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # No reparametrization since this is for a ablative study
        if self.training:
            x1, x2 = x[:, :self.idx], x[:, self.idx:]
            x1 = self.mn_conv(x1) + self.nm_conv(x1) + self.nn_conv(x1)
            return torch.cat([x1, x2], dim=1)
        
        else:
            x[:, :self.idx] = self.mn_conv(x[:, :self.idx]) + self.nm_conv(x[:, :self.idx]) + self.nn_conv(x[:, :self.idx])
            return x


class SparsePLKConv2d(nn.Module):
    " Sparse Partial Large Kernel Convolutional Layer (RepLKNet and UniRepLKNet style) "
    def __init__(self, dim, max_kernel_size, sub_kernel_sizes, dilations, use_max_kernel, with_idt):
        super().__init__()
        self.use_max_kernel = use_max_kernel
        self.max_kernel_size = max_kernel_size
        for k, d in zip(sub_kernel_sizes, dilations):
            m_k = self._calc_rep_kernel_size(k, d)
            if m_k > self.max_kernel_size:
                self.max_kernel_size = m_k
        self.with_idt = with_idt

        convs = [       
            nn.Conv2d(dim, dim, sub_kernel_size, 1, (sub_kernel_size // 2) * d, dilation=d) for sub_kernel_size, d in zip(sub_kernel_sizes, dilations)
        ]
        if use_max_kernel:
            convs.append(
                nn.Conv2d(dim, dim, self.max_kernel_size, 1, self.max_kernel_size // 2)
            )
        self.convs = nn.ModuleList(convs)
        for m in self.convs:
            trunc_normal_(m.weight, std=0.02)
        self.idx = dim
        self.is_convert = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_convert:
            x[:, :self.idx, :, :] = self.conv(x[:, :self.idx, :, :])
            return x
        else:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            if self.with_idt:
                out = x1
            else:
                out = 0.
            for conv in self.convs:
                out = out + conv(x1)
            return torch.cat([out, x2], dim=1)
        
    @staticmethod
    def _calc_rep_kernel_size(ks, dilation):
        return (ks - 1) * dilation + 1

    @staticmethod
    def _get_origin_kernel(kernel, dilation=1, p=0):
        I = torch.ones((1, 1, 1, 1)).to(kernel.device)
        if kernel.size(1) == 1:  # Depth-wise Convolution
            dilated = F.conv_transpose2d(kernel, I, stride=dilation)
        else:
            slices = []  # Dense or Group
            for i in range(kernel.size(1)):
                dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], I, stride=dilation)
                slices.append(dilated)
            dilated = torch.cat(slices, dim=1)
            
        # Pad boundary
        if p != 0:
            dilated = F.pad(dilated, (p, p, p, p))
        return dilated
    
    @staticmethod
    def _dwc_to_dense(kernel):
        n_groups = kernel.size(0)
        
        kernels = []
        for g in range(n_groups):
            kernels.append(
                torch.cat([
                    kernel[g: (g + 1)] if g == i else torch.zeros_like(kernel[g: (g + 1)])
                    for i in range(n_groups)
                ], dim=1)
            )
        return torch.cat(kernels, dim=0)
    
    @torch.no_grad()
    def convert(self):
        if not self.is_convert:
            k = 0.
            b = 0.
            for conv in self.convs:
                _k, _b = conv.weight, conv.bias
                _d = conv.dilation[0]
                
                if _d == 1:
                    ks_ = _k.shape[-1]
                    n_pad = (self.max_kernel_size - ks_) // 2
                    _k = F.pad(_k, (n_pad, n_pad, n_pad, n_pad))          
                else:
                    _ks = conv.kernel_size[0]
                    _ks_rep = self._calc_rep_kernel_size(_ks, _d)
                    _p = (self.max_kernel_size - _ks_rep) // 2
                    _k = self._get_origin_kernel(_k, _d, _p)
                
                if _k.size(1) == 1:
                    _k = self._dwc_to_dense(_k)
                
                k = k + _k
                b = b + _b
            
            device = k.device 
            
            if self.with_idt:
                I = torch.eye(k.size(0)).reshape(k.size(0), k.size(0), 1, 1).to(device)
                n_pad = (k.size(2) - 1) // 2
                I = F.pad(I, (n_pad, n_pad, n_pad, n_pad))
                k = k + I
            
            del self.convs
                
            self.conv = nn.Conv2d(k.size(1), k.size(0), k.size(2), 1, k.size(2) // 2)
            self.conv.weight.data.copy_(k)
            self.conv.bias.data.copy_(b)
            
            self.to(device)
            
            self.is_convert = True
            

class EA(nn.Module):
    " Element-wise Attention "
    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Sigmoid()
        )
        trunc_normal_(self.f[0].weight, std=0.02)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.f(x)


class PLKBlock(nn.Module):
    def __init__(
        self, dim: int, 
        # CCM Rep options
        ccm_type: Literal['CCM', 'ICCM', 'DCCM'],
        # LK Options
        max_kernel_size: int, split_ratio: float, lk_type: Literal['PLK', 'SparsePLK', 'RectSparsePLK'] = 'PLK',
        # Sparse Rep options
        use_max_kernel: bool = False, sparse_kernels: Sequence[int] = [5, 5, 5], sparse_dilations: Sequence[int] = [2, 3, 4],
        with_idt: bool = False,
        # EA ablation
        use_ea: bool = True
    ): 
        super().__init__()
        
        # Local Texture
        if ccm_type == 'CCM':
            self.channe_mixer = CCM(dim)
        elif ccm_type == 'ICCM':
            self.channe_mixer = ICCM(dim)
        elif ccm_type == 'DCCM':
            self.channe_mixer = DCCM(dim)
        else:
            raise ValueError(f'Unknown CCM type: {ccm_type}')
        
        # Long-range Dependency
        pdim = int(dim * split_ratio)
        if lk_type == 'PLK':
            self.lk = PLKConv2d(pdim, max_kernel_size, with_idt)
        elif lk_type == 'SparsePLK':
            self.lk = SparsePLKConv2d(pdim, max_kernel_size, sparse_kernels, sparse_dilations, use_max_kernel, with_idt)
        elif lk_type == 'RectSparsePLK':
            self.lk = RectSparsePLKConv2d(pdim, max_kernel_size)
        else:
            raise ValueError(f'Unknown LK type: {lk_type}')
        
        # Instance-dependent modulation
        if use_ea:
            self.attn = EA(dim)
        else:
            self.attn = nn.Identity()
        
        # Refinement
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        trunc_normal_(self.refine.weight, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.channe_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        return x + x_skip

        
class PLKSR(nn.Module):
    def __init__(
        self, state_dict
    ):
        super().__init__()
        state_keys = state_dict.keys()
        dim = 64
        n_blocks = 28
        # CCM options
        ccm_type = 'DCCM'
        # LK Options
        kernel_size = 17
        split_ratio = 0.5
        lk_type = 'PLK'
        # LK Rep options
        use_max_kernel = True
        sparse_kernels = [5, 5, 5, 5]
        sparse_dilations= [1, 2, 3, 4]
        with_idt = False  # not detect
        # EA ablation
        use_ea = False
        # Mobile Convert options
        is_coreml = False   # not detect
        scale = 4
        num_layers = get_seq_len(state_dict, "feats")
        scale = int((state_dict[f"feats.{num_layers-1}.weight"].shape[0]/3)**0.5)
        n_blocks = num_layers - 2
        ccm_shape = state_dict["feats.1.channe_mixer.2.weight"].shape[3]
        if ccm_shape == 3:
            ccm_type = "DCCM"
        elif ccm_shape == 1:
            ccm_type = "CCM"
        dim = state_dict["feats.0.weight"].shape[0]

        if "feats.1.lk.mn_conv.weight" in state_keys:
            lk_type = "RectSparsePLK"
            split_ratio = state_dict["feats.1.lk.mn_conv.weight"].shape[0]/dim
            kernel_size = state_dict["feats.1.lk.mn_conv.weight"].shape[3]
        elif "feats.1.lk.convs.0.weight" in state_keys:
            lk_type = "SparsePLK"
            split_ratio = state_dict["feats.1.lk.convs.0.weight"].shape[0]/dim
            len_sparse_kernels = get_seq_len(state_dict, f"feats.1.lk.convs")
            kernel_size = state_dict[f"feats.1.lk.convs.{len_sparse_kernels - 1}.weight"].shape[3]
            if kernel_size != 5:
                use_max_kernel = True
                len_sparse_kernels-=1
            sparse_kernels = [5 for _ in range(len_sparse_kernels)]
            sparse_dilations = [i + 1 for i in range(len_sparse_kernels)]
        else:
            split_ratio = state_dict["feats.1.lk.conv.weight"].shape[0]/dim
            kernel_size = state_dict["feats.1.lk.conv.weight"].shape[3]

        if "feats.1.attn.f.0.weight" in state_keys:
            use_ea = True
        upscaling_factor = scale
        self.input_channels = 3
        self.name = "PLKSR"
        self.feats = nn.Sequential(*[
            nn.Conv2d(3, dim, 3, 1, 1)
        ] + [
            PLKBlock(
                dim, ccm_type, kernel_size, split_ratio, lk_type, use_max_kernel, sparse_kernels, sparse_dilations, with_idt, use_ea
            ) for _ in range(n_blocks)
        ] + [
            nn.Conv2d(dim, 3 * upscaling_factor ** 2, 3, 1, 1)
        ])
        trunc_normal_(self.feats[0].weight, std=0.02)
        trunc_normal_(self.feats[-1].weight, std=0.02)
        
        self.to_img = nn.PixelShuffle(upscaling_factor)
        
        self.repeat_op = partial(
            repeat_interleave, n=upscaling_factor ** 2
        ) if is_coreml else partial(
            torch.repeat_interleave, repeats=upscaling_factor ** 2, dim=1
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feats(x) + self.repeat_op(x)
        x = self.to_img(x)
        return x


def pconv_forward_for_coreml(self, x):
    x1 = x[:, :self.idx, :, :]
    x2 = x[:, self.idx:, :, :]
    x1 = self.conv(x1)
    x = torch.cat([x1, x2], dim=1)
    return x


def convert_plk_forward_for_coreml(model):
    for m in model.modules():
        if isinstance(m, (PLKConv2d, SparsePLKConv2d)):
            m.forward = partial(pconv_forward_for_coreml, m)

