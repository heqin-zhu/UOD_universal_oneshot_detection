import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .u2net import DoubleConv


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class UniDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, domain_num=1):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, out_channels, domain_num=domain_num)

    def forward(self, x, skip, domain_idx=0):
        x = self.upsampling(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, domain_idx=domain_idx)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels,
            use_batchnorm=True,
    ):
        super().__init__()

        self.conv1 = Conv(
            in_channels,
            out_channels,
            kernel_size=3,
            bn=use_batchnorm
        )
        self.conv2 = Conv(
            out_channels,
            out_channels,
            kernel_size=1,
            bn=use_batchnorm
        )
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, skip):

        x = self.upsampling(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class DecoderCup(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.Blocks = nn.ModuleList()
        for in_chan, out_chan, skip_chan in zip(in_channels, out_channels, skip_channels):
            self.Blocks.append(DecoderBlock(in_chan,out_chan,skip_chan))

    def forward(self, features):
        x = features[0]
        for i, block in enumerate(self.Blocks):
            skip = features[i+1]
            x = block(x, skip)
        return x

class UniDecoderCup(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, domain_num=1):
        super().__init__()
        self.domain_num = domain_num
        self.Blocks = nn.ModuleList()
        for in_chan, out_chan, skip_chan in zip(in_channels, out_channels, skip_channels):
            self.Blocks.append(UniDecoderBlock(in_chan, out_chan, domain_num=domain_num))

    def forward(self, features, domain_idx=0, get_features=False):
        x = features[0]
        dec_features = []
        for i, block in enumerate(self.Blocks):
            skip = features[i+1]
            x = block(x, skip, domain_idx=domain_idx)
            dec_features.append(x)
        if get_features:
            return x, dec_features
        else:
            return x


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)

        self.first_conv = self.encoder.conv1
        if in_channels !=3 and in_channels !=1:
            self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        inputs = x

        if inputs.size(1) == 1:
            inputs = torch.cat([inputs, inputs, inputs], dim=1)
        x0 = self.first_conv(inputs)

        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)
        c0 = x0
        x0 = self.encoder.maxpool(x0)
        c1 = self.encoder.layer1(x0)
        c2 = self.encoder.layer2(c1)
        c3 = self.encoder.layer3(c2)
        c4 = self.encoder.layer4(c3)

        features = [c4, c3, c2, c1, c0]
        return features


class Decoder(nn.Module):
    def __init__(self, in_dim, feature_channels=[1024, 512, 256, 64]):
        super().__init__()
        assert len(feature_channels)==4
        self.initial = Conv(inp_dim=in_dim, out_dim=32, bn=True)
        skip_channels = [1024, 512, 256, 64]
        in_feature_channels = [2048, *feature_channels[:-1]]
        embed_dims = [i+j for i, j in zip(in_feature_channels, skip_channels)]

        self.decoder = DecoderCup(
                          in_channels = embed_dims, 
                          out_channels = feature_channels, 
                          skip_channels = skip_channels
                    )
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, features):
        initial = self.initial(x)
        x = self.decoder(features)
        x = self.upsampling(x)
        x = torch.cat([x, initial], dim=1)
        return x

class UniDecoder(nn.Module):
    def __init__(self, in_dim, feature_channels=[1024, 512, 256, 64], domain_num=1):
        super().__init__()
        assert len(feature_channels)==4
        self.initial = Conv(inp_dim=in_dim, out_dim=32, bn=True)
        skip_channels = [1024, 512, 256, 64]
        in_feature_channels = [2048, *feature_channels[:-1]]
        embed_dims = [i+j for i, j in zip(in_feature_channels, skip_channels)]

        self.decoder = UniDecoderCup(
                          in_channels = embed_dims, 
                          out_channels = feature_channels, 
                          skip_channels = skip_channels,
                          domain_num = domain_num,
                    )
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, features, domain_idx=1):
        initial = self.initial(x)
        x = self.decoder(features, domain_idx)
        x = self.upsampling(x)
        x = torch.cat([x, initial], dim=1)
        return x



class MESD(nn.Module):
    ''' multiple encoder, single decoder'''
    def __init__(self, in_channels, out_channel_list):
        super().__init__()
        self.encoder_lst = nn.ModuleList([ResNetEncoder(in_channels) for _ in out_channel_list])
        self.decoder = Decoder(in_channels)
        self.head_lst = nn.ModuleList([
                          nn.Sequential(OrderedDict([
                            ('conv3', nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)),
                            ('norm', nn.BatchNorm2d(96)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('conv1', nn.Conv2d(96, out_chan, kernel_size=1))
                            ]))
                           for out_chan in out_channel_list
        ])


    def forward(self, x, domain_idx=0, get_features=False):
        features = self.encoder_lst[domain_idx](x)
        out_x = self.decoder(x, features)

        ret = {}
        ret['heatmap'] = self.head_lst[domain_idx](out_x)
        if get_features:
            return ret['heatmap'], features
        else:
            return ret

class MEUD(nn.Module):
    ''' multiple encoder, single decoder'''
    def __init__(self, in_channels, out_channel_list):
        super().__init__()
        self.encoder_lst = nn.ModuleList([ResNetEncoder(in_channels) for _ in out_channel_list])
        domain_num = len(out_channel_list)
        self.decoder = UniDecoder(in_channels, domain_num=domain_num)
        self.head_lst = nn.ModuleList([
                          nn.Sequential(OrderedDict([
                            ('conv3', nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)),
                            ('norm', nn.BatchNorm2d(96)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('conv1', nn.Conv2d(96, out_chan, kernel_size=1))
                            ]))
                           for out_chan in out_channel_list
        ])


    def forward(self, x, domain_idx=0, get_features=False):
        features = self.encoder_lst[domain_idx](x)
        out_x = self.decoder(x, features, domain_idx=domain_idx)

        ret = {}
        ret['heatmap'] = self.head_lst[domain_idx](out_x)
        if get_features:
            return ret['heatmap'], features
        else:
            return ret



class MEMD(nn.Module):
    ''' multiple encoder, multiple decoder'''
    def __init__(self, in_channels, out_channel_list):
        super().__init__()
        self.encoder_lst = nn.ModuleList([ResNetEncoder(in_channels) for _ in out_channel_list])
        self.decoder_lst = nn.ModuleList([Decoder(in_channels) for _ in out_channel_list])
        self.head_lst = nn.ModuleList([
                          nn.Sequential(OrderedDict([
                            ('conv3', nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)),
                            ('norm', nn.BatchNorm2d(96)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('conv1', nn.Conv2d(96, out_chan, kernel_size=1))
                            ]))
                           for out_chan in out_channel_list
        ])


    def forward(self, x, domain_idx=0, get_features=False):
        features = self.encoder_lst[domain_idx](x)
        out_x = self.decoder_lst[domain_idx](x, features)

        ret = {}
        ret['heatmap'] = self.head_lst[domain_idx](out_x)
        if get_features:
            return ret['heatmap'], features
        else:
            return ret


class Multi_Domain_Module(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.convs = nn.ModuleList([Conv(chan, chan, bn=True) for chan in feature_channels])

    def forward(self, features):
        return [self.convs[i](f)+f for i, f in enumerate(features)]


if __name__ == '__main__':
    img_size = (384, 384)
    out_channel_list = [19, 38]
    img = torch.randn(2, 1 , *img_size)
    gt_map = torch.randn(2, out_channel_list[0] , *img_size)
    domain_idx = 1
    MODEL_LST = [MESD, MEMD, MEUD]
    MODEL_LST = [MEUD]
    for Model in MODEL_LST:
        print(Model.__name__)
        model = Model(1, out_channel_list)
        out = model(img, domain_idx)
        print(out.keys())
        for k in out:
            print(k, out[k].shape)

        # print('>> features')
        # heatmap, features = model(img, domain_idx, get_features=True)
        # for f in features:
        #     print(f.shape)
