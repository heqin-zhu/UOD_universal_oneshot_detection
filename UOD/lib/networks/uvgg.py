import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

from .base_vgg19 import vgg19
from .mdn import UniDecoderCup, Conv
from .unet_pretrained import Up


class Universal_Decoder(nn.Module):
    def __init__(self, in_channels=[512, 512, 256, 128, 64], out_channels=[256, 128, 64, 128, 128], domain_num=1):
        super().__init__()
        assert len(out_channels)==len(in_channels)
        self.initial = Conv(inp_dim=3, out_dim=32, bn=True)
        skip_channels = in_channels
        in_feature_channels = [in_channels[0], *out_channels[:-1]]
        embed_dims = [i+j for i, j in zip(in_feature_channels, skip_channels[1:])]

        self.decoder = UniDecoderCup(
                          in_channels = embed_dims, 
                          out_channels = out_channels, 
                          skip_channels = skip_channels,
                          domain_num = domain_num,
                    )
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, features, domain_idx=0, get_features=False):
        initial = self.initial(x)
        x, dec_features = self.decoder(features, domain_idx=domain_idx, get_features=get_features)
        x = self.upsampling(x)
        x = torch.cat([x, initial], dim=1)
        if get_features:
            return x, dec_features
        else:
            return x


class UVGG(nn.Module):
    def __init__(self, in_channels, non_local=False, emb_len=16, domain_num=1):
        super().__init__()
        self.in_channels = in_channels
        length_embedding = emb_len

        in_feature_channels = [512, 512, 256, 128, 64]
        self.decoder = Universal_Decoder(in_channels=in_feature_channels, out_channels=[256, 128, 64, 128, 128], domain_num=domain_num)
        self.Up = Up(128, 128, bilinear=True)

        self.encoders = nn.ModuleList([vgg19(pretrained=True) for i in range(domain_num)])
        self.trans_list = nn.ModuleList([nn.Conv2d(chan, length_embedding, kernel_size=1) for chan in [512, 256, 128, 64, 128, 128]])

        if non_local:
            self.non_local_list = nn.ModuleList([RFB_modified(chan, chan) for chan in [512, 256, 128]])
        self.non_local = non_local




    def forward(self, x, domain_idx=0, get_features=False):
        _, features = self.encoders[domain_idx](x, get_features=True)

        _, dec_features = self.decoder(x, features[::-1], domain_idx=domain_idx, get_features=True)
        dec_features.insert(0, features[4])
        dec_features.append(self.Up(dec_features[-1]))


        if self.non_local:
            for i in range(3):
                dec_features[i] = self.non_local_list[i](dec_features[i])
        for i, f in enumerate(dec_features):
            dec_features[i] = self.trans_list[i](f)

        out = dec_features[-1]
        if get_features:
            return out, dec_features
        else:
            return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


if __name__ == '__main__':
    model = UVGG(3, emb_len=16, domain_num=3, non_local=True)
    x = torch.zeros([4, 3, 384, 384], dtype=torch.float)
    out, features = model(x, domain_idx=1, get_features=True)
    for f in features:
        print(f.shape)
