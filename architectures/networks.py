import torch
from torch import nn


class ConvBlock(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=nn.InstanceNorm3d, 
                 norm_kwargs={}, nonlin=nn.LeakyReLU, nonlin_kwargs={}, num_convs=2) -> None:
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
                                                   norm(out_channels, **norm_kwargs))])
        for _ in range(num_convs - 1):
            self.convs.append(nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding=1),
                                            norm(out_channels, **norm_kwargs)))
        self.nonlin = nonlin(**nonlin_kwargs)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.nonlin(x)
        return x


class ResBlock(nn.Module):
    """Residual block for ResNet like architectures"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=nn.InstanceNorm3d, 
                 norm_kwargs={}, nonlin=nn.LeakyReLU, nonlin_kwargs={}, num_convs=2) -> None:
        super().__init__()
        self.nonlin = nonlin(**nonlin_kwargs)
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
                                                   norm(out_channels, **norm_kwargs))])
        for _ in range(num_convs - 1):
            self.convs.append(nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding=1),
                                            norm(out_channels, **norm_kwargs)))
        self.identity = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) \
                        if in_channels != out_channels else nn.Identity()
        self.nonlin = nonlin(**nonlin_kwargs)

    def forward(self, x):
        conv = self.convs[0]
        res = conv(x)
        if len(self.convs) > 1:
            res = self.nonlin(res)
        for i, conv in enumerate(self.convs[1:], 1):
            res = conv(res)
            if i < len(self.convs) - 1:
                res = self.nonlin(res)
        x = res + self.identity(x)
        x = self.nonlin(x)
        return x


class MyResNet(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, depth: int, num_base_channels: int, 
                 num_max_channels: int, blocks_per_layer: int|list=2) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, num_base_channels, kernel_size=3, stride=1)
        if isinstance(blocks_per_layer, int):
            blocks_per_layer = [blocks_per_layer] * depth
        assert len(blocks_per_layer) == depth, "blocks_per_layer must be either an integer or a list of integers of length depth"
        self.downs = nn.ModuleList()
        for d in range(depth):
            in_channels = min(num_base_channels * 2**d, num_max_channels)
            out_channels = min(num_base_channels * 2**(d+1), num_max_channels)
            self.downs.append(nn.Sequential(nn.MaxPool3d(2), ResBlock(in_channels, out_channels, kernel_size=3, stride=1, num_convs=blocks_per_layer[d]), 
                                            *[ResBlock(out_channels, out_channels, kernel_size=3, stride=1, num_convs=blocks_per_layer[d]) 
                                            for _ in range(blocks_per_layer[d] - 1)]))
        self.final = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(out_channels, num_actions))
        # keep out_channels for binary input
        self.out_channels = out_channels

    def forward(self, x):
        x = self.stem(x)
        for down in self.downs:
            x = down(x)
        return self.final(x)


class MyResNetBinary(MyResNet):
    """
    This class inherits from MyResNet and allows for additional binary input to the network.
    """
    def __init__(self, in_channels: int, num_actions: int, depth: int, num_base_channels: int, 
                 num_max_channels: int, blocks_per_layer: int|list=2, num_binary: int=1) -> None:
        super().__init__(in_channels, num_actions, depth, num_base_channels, num_max_channels, blocks_per_layer)
        self.final = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(self.out_channels + num_binary, num_actions))

    def forward(self, x, x_bin):
        x = self.stem(x)
        for down in self.downs:
            x = down(x)
        x = torch.cat((x, x_bin), dim=1)
        return self.final(x)


if __name__ == "__main__":
    model = MyResNet(1, 2, 4, 8, 64, [4, 7, 7, 4]).to("cuda")
    print(model)
    x = torch.randn(1, 1, 64, 64, 64).to("cuda")
    y = model(x)
    print(y.shape)
