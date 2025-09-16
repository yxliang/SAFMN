import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class SimpleSAFM(nn.Module):
    """
    Simplified SAFM (Simple Spatial-Frequency Modulation) Block.
    This version uses adaptive_max_pool2d and variance for spatial feature modulation.
    """
    def __init__(self, dim, split_dim=12):
        super().__init__()

        dim2 = dim - split_dim
        self.conv1 = nn.Conv2d(dim, split_dim, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim2, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(split_dim, split_dim, 1, 1, 0, bias=False)
        
        self.dwconv = nn.Conv2d(split_dim, split_dim, 3, 1, 1, groups=split_dim, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        # Split features
        x0 = self.conv1(x)
        x1 = self.conv2(x)

        # Spatial feature modulation branch
        x2 = F.max_pool2d(x0, kernel_size=8, stride=8)
        x2 = self.dwconv(x2)
        # Add variance as a global descriptor
        x2 = self.conv3(x2 + torch.var(x0, dim=(-2, -1), keepdim=True))
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=False)
        x2 = self.act(x2) * x0

        # Concatenate and output
        x = torch.cat([x1, x2], dim=1)
        x = self.out(self.act(x))
        return x


class FFN(nn.Module):
    """
    Feed-Forward Network implemented with 1x1 convolutions.
    """
    def __init__(self, dim, ffn_scale):
        super().__init__()
        hidden_dim = int(dim * ffn_scale)

        self.conv1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x


class AttBlock(nn.Module):
    """
    Attention Block combining SimpleSAFM and FFN.
    """
    def __init__(self, dim, ffn_scale):
        super().__init__()

        self.conv1 = SimpleSAFM(dim, split_dim=12)
        self.conv2 = FFN(dim, ffn_scale)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


@ARCH_REGISTRY.register()
class SAFMNV3(nn.Module):
    """
    Simple and Fast Multi-scale Network Version 3 (SAFMNV3)
    """
    def __init__(self, dim=40, n_blocks=8, ffn_scale=2.0, upscaling_factor=5):
        super().__init__()
        self.scale = upscaling_factor

        # Feature extraction
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1, bias=False)

        # Deep feature processing
        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        # Image reconstruction
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1, bias=False),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        # Initial bicubic upsampling for residual connection
        res = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        x = self.to_feat(x)
        x = self.feats(x)
        
        # Add residual connection
        return self.to_img(x) + res
    

if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 3, 768, 560)
    # x = torch.randn(1, 3, 256, 256)

    model = SAFMNV3(dim=40, n_blocks=6, ffn_scale=2.0, upscaling_factor=5)
    # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)