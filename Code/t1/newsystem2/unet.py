import math
from typing import Union, Tuple, List

import torch
from torch import nn, Tensor
from torch.nn import Identity, Module, ModuleList, Linear, Sigmoid, GroupNorm, Conv2d, ConvTranspose2d, Dropout

class TimeEmbed(Module):
    """A class that handles sinusoidal position embeddings.
        Utilised as parameters of the network are shared across time.
        Tells the network which timestep/noise level it is operating at.
        tensor (shape = batchsize, 1) -> tensor (shape = batchsize, dim)"""

    def __init__(self, dim: int):
        super().__init__()
        # dimensionality of sinusoidal position embedding
        self.dim = dim
        self.lin1 = Linear(self.dim//4, self.dim)
        self.act = Sigmoid()
        self.lin2 = Linear(self.dim, self.dim)

    def forward(self, t: Tensor):
        half_dim = self.dim // 8

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.lin2(self.act(self.lin1(emb)))
        return emb


class ResBlock(Module):
    """A class for residual blocks. A basic ResNet block is composed by two layers of 3x3 conv/batchnorm/activation."""

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32, conv_shortcut: bool = False, dropout: float = 0.1):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        # x -> group norm -> sigmoid -> conv2d
        self.norm1 = GroupNorm(n_groups, in_channels)
        self.act1 = Sigmoid()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1))

        # x -> group norm -> sigmoid -> dropout -> conv2d
        self.norm2 = GroupNorm(n_groups, out_channels)
        self.act2 = Sigmoid()
        self.dropout = Dropout(p=dropout)
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1))

        # if in channels != out_channels, we project the shorcut connection
        # if self.should_apply_shortcut:
        #     if conv_shortcut:
        #         self.shortcut = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        #     else:
        #         # TODO: decide on linear or identity shortcut
        #         # self.shortcut = nn.Linear(in_features=in_channels, out_features=out_channels)
        #         self.shortcut = Identity()

        if self.in_channels == self.out_channels:
            self.shortcut = Identity()
        elif conv_shortcut:
            self.shortcut = Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        else:
            self.shortcut = Conv2d(self.in_channels, self.out_channels, 1)

        self.time_emb = Linear(in_features=time_channels, out_features=out_channels)

    def forward(self, x: Tensor, t: Tensor):
        # x -> group norm -> sigmoid -> conv2d
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # add time embeddings
        h += self.time_emb(t)[:, :, None, None]

        # x -> group norm -> sigmoid -> dropout -> conv2d
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # add shorcut connections
        # h += self.shortcut(x)

        return self.shortcut(x) + h

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class AttnBlock(nn.Module):
    """An attention function can be described as mapping a query and a set of key-value pairs 
        to an output, where the query, keys, values, and output are all vectors. - Attention
        is all you need. attention(q, k ,v) = softmax(dot(q,k)/root(d_k)) * v"""

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super().__init__()
        self.n_heads = n_heads
        # d_k is dimensionality of the keys - the default is the number of channels 
        if d_k is None: d_k = n_channels
        self.d_k = d_k

        # normalisation
        self.norm = GroupNorm(num_groups=n_groups, num_channels=n_channels)
        # projection of query, key and values
        self.proj = Linear(in_features=n_channels, out_features=(n_heads * d_k * 3))
        # transformation of shape
        self.out = Linear(in_features=(n_heads * d_k), out_features=n_channels)
        # scale to counteract growth of dot products to regions where softmax has small gradients
        self.scale = d_k ** -0.5

    def forward(self, x: Tensor, t: Tensor = None):
        # project out shape
        B, C, H, W = x.shape
        # reshape x to [B, S, C] - flattens the image
        x = x.view(B, C, -1).permute(0, 2, 1)
        # get q, k, v, with lengths B, S, (3*d_k)
        q, k, v = torch.chunk(self.proj(x).view(B, -1, self.n_heads, 3*self.d_k), 3, dim=-1)
        # calculate scaled dot product = dot(q,k)/root(d_k)
        dotted = torch.einsum('bihd, bjhd -> bijh', q, k) * self.scale
        # softmax along S dimension
        soft_maxed = dotted.softmax(dim = 2)
        # multiply by v
        attn = torch.einsum('bijh, bjhd -> bihd', soft_maxed, v)
        # reshape
        attn = attn.view(B, -1, self.n_heads * self.d_k)
        # transform shape
        attn = self.out(attn)
        # add skip connection
        # TODO: decide order here attn + x or x + attn
        attn += x
        # transform shape
        attn = attn.permute(0,2,1).view(B, C, H, W)
        # return
        return attn

class ResAttnDown(Module):
    """Residual Block followed by optional Attention Block for the 1st half of the UNet"""

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()

        # Residual block
        self.res = ResBlock(in_channels, out_channels, time_channels)
        # Optional Attention block
        self.attn = AttnBlock(out_channels) if has_attn else Identity()

    def forward(self, x: Tensor, t: Tensor):
        return self.attn(self.res(x, t))

class DownSample(Module):
    """Downsample the feature map using scale factor 2"""

    def __init__(self, n_channels: int):
        super().__init__()
        # conv layer reduces size by factor 2
        self.conv = Conv2d(n_channels, n_channels, kernel_size=(3,3), stride=(2,2), padding=(1,1))

    def forward(self, x: Tensor, t: Tensor = None):
        return self.conv(x)

class ResAttnUp(Module):
    """Residual Block followed by optional Attention Block for the 2nd half of the UNet"""

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()

        # Residual block (in + out due to shorcut concatenation)
        self.res = ResBlock(in_channels + out_channels, out_channels, time_channels)
        # Optional Attention block
        self.attn = AttnBlock(out_channels) if has_attn else Identity()

    def forward(self, x: Tensor, t: Tensor):
        return self.attn(self.res(x, t))

class UpSample(Module):
    """Upsample the feature map using scale factor 2"""

    def __init__(self, n_channels: int):
        super().__init__()
        # transpose conv increases size by scale factor 2
        self.conv = ConvTranspose2d(n_channels, n_channels, kernel_size=(4,4), stride=(2,2), padding=(1,1))

    def forward(self, x: Tensor, t: Tensor = None):
        return self.conv(x)

class BottleNeck(Module):
    """Bottom of the UNet, acts as the bottleneck. ResBlock -> AttnBlock -> ResBlock."""

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.resDown = ResBlock(n_channels, n_channels, time_channels)
        self.attn = AttnBlock(n_channels)
        self.resUp = ResBlock(n_channels, n_channels, time_channels)

    def forward(self, x: Tensor, t: Tensor):
        return self.resUp(self.attn(self.resDown(x, t)), t)

class UNet(Module):
    """UNet"""

    def __init__(self, image_channels: int = 3, n_channels: int = 64, 
                       ch_multipliers: Union[Tuple[int, ...], List[int]] = (1, 2, 4, 8),
                       is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, False),
                       n_blocks: int = 2):
        super().__init__()

        # calculate number of resolutions
        n_resolutions = len(ch_multipliers)

        # project image into feature map
        self.image_proj = Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # time embedding. dim is n_channels * 4 channels
        self.time_emb = TimeEmbed(n_channels * 4)

        # --------------------------------------------------------------------------------------------------

        # 1st half of UNet
        self.down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_multipliers[i]
            for _ in range(n_blocks):
                self.down.append(ResAttnDown(in_channels, out_channels, time_channels=n_channels*4, has_attn=is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                self.down.append(DownSample(in_channels))
        
        self.down = ModuleList(self.down)

        # Bottleneck
        self.bottleneck = BottleNeck(n_channels=out_channels, time_channels=n_channels*4)

        # 2nd Half of UNet

        self.up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                self.up.append(ResAttnUp(in_channels, out_channels, time_channels=n_channels*4, has_attn=is_attn[i]))
            out_channels = in_channels // ch_multipliers[i]
            self.up.append(ResAttnUp(in_channels, out_channels, time_channels=n_channels*4, has_attn=is_attn[i]))
            in_channels = out_channels
            if i > 0:
                self.up.append(UpSample(in_channels))

        self.up = ModuleList(self.up)

        # --------------------------------------------------------------------------------------------------

        # Normalisation and output convolution
        self.norm = GroupNorm(num_groups=32, num_channels=n_channels)
        self.act = Sigmoid()
        self.out = Conv2d(in_channels, image_channels, kernel_size=(3,3), padding=(1,1))

    def forward(self, x: Tensor, t: Tensor):
        t = self.time_emb(t)
        x = self.image_proj(x)

        # we store outputs for skip connections
        outputs = [x]
        # 1st half
        for module in self.down:
            x = module(x, t)
            outputs.append(x)
        
        # bottleneck
        x = self.bottleneck(x, t)

        # 2nd half
        for module in self.up:
            if isinstance(module, UpSample):
                x = module(x, t)
            else:
                s = outputs.pop()
                x = torch.cat([x, s], dim=1)
                x = module(x, t)
        
        # output
        return self.out(self.act(self.norm(x)))