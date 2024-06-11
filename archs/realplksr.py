from functools import partial

import torch
from torch import nn
from torch.nn.init import trunc_normal_
from .utils.state import get_seq_len
from .utils.dysample import DySample

training = False


class DCCM(nn.Sequential):
    "Doubled Convolutional Channel Mixer"

    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class PLKConv2d(nn.Module):
    "Partial Large Kernel Convolutional Layer"

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim
        self.training = training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        x[:, : self.idx] = self.conv(x[:, : self.idx])
        return x


class EA(nn.Module):
    "Element-wise Attention"

    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.f(x)


class PLKBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        split_ratio: float,
        norm_groups: int,
        use_ea: bool = True,
    ):
        super().__init__()

        # Local Texture
        self.channel_mixer = DCCM(dim)

        # Long-range Dependency
        pdim = int(dim * split_ratio)

        # Conv Layer
        self.lk = PLKConv2d(pdim, kernel_size)

        # Instance-dependent modulation
        if use_ea:
            self.attn = EA(dim)
        else:
            self.attn = nn.Identity()

        # Refinement
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        trunc_normal_(self.refine.weight, std=0.02)

        # Group Normalization
        self.norm = nn.GroupNorm(norm_groups, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)

        return x + x_skip


class realplksr(nn.Module):
    """Partial Large Kernel CNNs for Efficient Super-Resolution:
    https://arxiv.org/abs/2404.11848
    """

    def __init__(
        self, state_dict
    ):
        super().__init__()
        dim = 64
        n_blocks = 28
        upscaling_factor = 4
        kernel_size = 17
        split_ratio = 0.25
        use_ea = False
        norm_groups = 4
        dropout = 0
        state_keys = state_dict.keys()
        num_layers = get_seq_len(state_dict, "feats")
        upscaling_factor = int((state_dict[f"feats.{num_layers - 1}.weight"].shape[0] / 3) ** 0.5)
        n_blocks = num_layers - 3
        dim = state_dict["feats.0.weight"].shape[0]
        split_ratio = state_dict["feats.1.lk.conv.weight"].shape[0] / dim
        kernel_size = state_dict["feats.1.lk.conv.weight"].shape[3]
        self.name = "realplksr"
        self.input_channels = 3
        if "feats.1.attn.f.0.weight" in state_keys:
            use_ea = True

        self.feats = nn.Sequential(
            *[nn.Conv2d(3, dim, 3, 1, 1)]
            + [
                PLKBlock(dim, kernel_size, split_ratio, norm_groups, use_ea)
                for _ in range(n_blocks)
            ]
            + [nn.Dropout2d(dropout)]
            + [nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1)]
        )
        trunc_normal_(self.feats[0].weight, std=0.02)
        trunc_normal_(self.feats[-1].weight, std=0.02)

        self.repeat_op = partial(
            torch.repeat_interleave, repeats=upscaling_factor**2, dim=1
        )

        if "to_img.init_pos" in state_keys:

            out_channels = state_dict["to_img.end_conv.weight"].shape[0]
            in_channels = state_dict["to_img.end_conv.weight"].shape[1]
            scale = int((in_channels / out_channels) ** 0.5)

            offset_shape = state_dict["to_img.offset.weight"].shape
            style = 'pl' if offset_shape[1] == out_channels else 'lp'
            if style == "pl":
                groups = offset_shape[0]//2
            else:
                groups = int(offset_shape[0]/2/scale**2)
            dyscope = "to_img.scope.weight" in state_keys
            self.to_img = DySample(in_channels, scale, style,groups, dyscope)
        else:
            self.to_img = nn.PixelShuffle(upscaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feats(x) + self.repeat_op(x)
        return self.to_img(x)

