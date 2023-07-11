import torch
import torch.nn as nn
import torch.nn.functional as F


class dwise(nn.Module):
    def __init__(self, inChans, kernel_size=3, stride=1, padding=1):
        super(dwise, self).__init__()
        self.conv1 = nn.Conv2d(inChans, inChans, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=inChans)

    def forward(self, x):
        out = self.conv1(x)
        return out


class pwise(nn.Module):
    def __init__(self, inChans, outChans, kernel_size=1, stride=1, padding=0):
        super(pwise, self).__init__()
        self.conv1 = nn.Conv2d(
            inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv1(x)
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, domain_num=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.dwise1 = nn.ModuleList([dwise(in_channels)
                                     for i in range(domain_num)])
        self.dwise2 = nn.ModuleList([dwise(mid_channels)
                                     for i in range(domain_num)])
        self.pwise1 = pwise(in_channels, mid_channels)
        self.pwise2 = pwise(mid_channels, out_channels)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(mid_channels)
                                  for i in range(domain_num)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_channels)
                                  for i in range(domain_num)])
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, domain_idx=0):
        x = self.pwise1(self.dwise1[domain_idx](x))
        x = self.relu1(self.bn1[domain_idx](x))
        x = self.pwise2(self.dwise2[domain_idx](x))
        x = self.relu2(self.bn2[domain_idx](x))
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, domain_num=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, domain_num=domain_num)

    def forward(self, x, domain_idx=0):
        return self.conv(self.maxpool(x), domain_idx)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, domain_num=1, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, domain_num)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels, out_channels, domain_num=domain_num)

    def forward(self, x1, x2, domain_idx=0):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, domain_idx)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class U2Net(nn.Module):
    def __init__(self, in_channels, out_channel_list, num_layers=4, use_actionmap=False, use_offset=False):
        super().__init__()
        base_channels = 64
        self.in_channels = in_channels
        domain_num = len(out_channel_list)
        self.outconv = nn.ModuleList([OutConv(base_channels, out_chan) for out_chan in out_channel_list])

        self.first_conv = OutConv(in_channels, base_channels)
        self.conv = DoubleConv(base_channels, base_channels, domain_num=domain_num)
        self.num_layers = num_layers
        self.down_layers = []
        self.up_layers = []
        for i in range(num_layers):
            in_chan = base_channels*(2**i)
            out_chan = in_chan if i==num_layers-1 else in_chan*2
            self.down_layers.append(Down(in_chan, out_chan, domain_num))

            up_in_chan = in_chan * 2
            up_out_chan = in_chan if i==0 else in_chan//2
            self.up_layers.append(Up(up_in_chan, up_out_chan, domain_num))
        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)

    def forward(self, x, domain_idx=0):
        x1 = self.conv(self.first_conv(x), domain_idx)
        xs = [x1]
        for i in range(self.num_layers):
            xs.append(self.down_layers[i](xs[-1], domain_idx))
        x = xs[-1]
        for i in range(self.num_layers-1, -1, -1):
            x = self.up_layers[i](x, xs[i], domain_idx)
        out = self.outconv[domain_idx](x)
        return {'heatmap': out}


if __name__ == '__main__':
    x = torch.randn(2, 1, 256, 256)
    model = U2Net(1, {'head':19})
    out = model(x, 'head')
    print(out['heatmap'].shape)
