import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .u2net import Up, OutConv
from .SwinTransformer import SwinTransformer, Mlp, BasicLayer

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential, nn.ModuleList)):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.GELU, nn.Softmax, nn.ReLU, nn.Dropout, nn.LayerNorm)):
            pass
        else:
            pass


def embedding(x):
    B, C, H, W = x.size()
    x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
    return x


def unembedding(x, H, W):
    B, L, C = x.size()
    x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    return x


class myTransBlk(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim=None,
            mlp_ratio=1,
            n_head=8,
            attn_pdrop=0,
            resid_pdrop=0,
            act_func=nn.GELU,
            depth=1,
            domain_num=1,
            use_layerscale=False,
              ):
        super().__init__()
        self.layers = BasicLayer(
                 dim=in_dim,
                 depth=depth,
                 num_heads=n_head,
                 window_size=7,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 domain_num=domain_num,
                 use_layerscale=use_layerscale,
                )
        if out_dim is not None and out_dim!=in_dim:
            self.final=nn.Sequential(
                                     nn.Linear(in_dim, out_dim),
                                     act_func(),
                                    )
        else:
            self.final = nn.Identity()
    def forward(self, x, H=None, W=None, domain_idx=0):
        if H is None or W is None:
            B, L, C = x.size()
            H = W = round(L**0.5)
        x_out, H, W, x, H, W = self.layers(x, H, W, domain_idx=domain_idx)
        return self.final(x_out)


class FAM(nn.Module):
    ''' Feature aggregation module

            x1: B x 4L x embed_dim
            x2: B x L x 2embed_dim
            out: B x 4L x embed_dim
    '''
    def __init__(self, embed_dim, mlp_ratio=4,n_head=8,attn_pdrop=0,resid_pdrop=0,depth=1,residual=False, act_func=nn.GELU, multi_feature=True, domain_num=1, use_layerscale=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.residual = residual
        self.multi_feature = multi_feature

        self.in_trans = myTransBlk(
            in_dim=embed_dim, 
            out_dim=2*embed_dim, 
            mlp_ratio=mlp_ratio,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            act_func=act_func,
            depth=depth,
            domain_num=domain_num,
            use_layerscale=use_layerscale,
          )
        self.fuse_trans = myTransBlk(
            in_dim=2*(2+multi_feature)*embed_dim, 
            out_dim=embed_dim, 
            mlp_ratio=mlp_ratio,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            act_func=act_func,
            depth=depth,
            domain_num=domain_num,
            use_layerscale=use_layerscale,
          )

    def forward(self, x1, x2, domain_idx=0):
        ''' 
            x1: B x 4L x C
            x2: B x L x 2C
            out: B x 4L x C
        '''
        res = x1
        B, L1, C1 = x1.size()
        B, L2, C2 = x2.size()
        assert L1==4*L2 and 2*C1 == C2 == 2*self.embed_dim, 'dimension mismatch: {}, {}'.format(x1.size(), x2.size())

        # x1: (B, 4L, C) -> (B, 4L, 2C)
        x1 = self.in_trans(x1, domain_idx=domain_idx)

        # x2: (B, L, 2C) -> (B, 4L, 2C)
        H = W = round(L2**0.5)
        x2 = unembedding(x2, H, W)
        x2 = F.interpolate(x2, size=(2*H, 2*W), mode='bilinear', align_corners=True)
        x2 = embedding(x2)

        x = torch.cat([x1,x2],dim=2).contiguous()
        if self.multi_feature:
            x = torch.cat((x, x1*x2),dim=2).contiguous()
        x = self.fuse_trans(x, domain_idx=domain_idx)
        if self.residual:
            x += res
        return x


class ConvDecoder(nn.Module):
    def __init__(self, embed_dim, out_channels, domain_num=1):
        super().__init__()
        self.up0 = Up(embed_dim*16, embed_dim*4, domain_num=domain_num)
        self.up1 = Up(embed_dim*8, embed_dim*2, domain_num=domain_num)
        self.up2 = Up(embed_dim*4, embed_dim , domain_num=domain_num)
        self.up3 = Up(embed_dim*2, 64, domain_num=domain_num)
        for i, out_chan in enumerate(out_channels):
            setattr(self, 'out{i}'.format(i=i), OutConv(64, out_chan))
    def forward(self, xs, H, W, domain_idx=0):
        x = xs[-1]
        for i, feat in enumerate(xs[::-1]):
            x = getattr(self, 'up{}'.format(i))(x, feat, domain_idx)
        x = getattr(self, 'out{}'.format(domain_idx))(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

class TransDecoder(nn.Module):
    def __init__(self, out_channels, embed_dim, depths, down_scale, attn_drop_rate, drop_rate, fusion_depth, residual, module_residual, domain_num, use_layerscale):
        super().__init__()
        self.mask_trans = nn.ModuleList()
        dim = embed_dim
        for i in range(len(depths)):
            self.mask_trans.append(nn.Sequential(nn.Linear(dim, dim//down_scale), nn.GELU()))
            dim*=2

        self.mask_fam = nn.ModuleList()

        dim = embed_dim//down_scale
        for i in range(len(depths)-1):
            self.mask_fam.append(FAM(dim, mlp_ratio=2,n_head=8,attn_pdrop=attn_drop_rate, resid_pdrop=drop_rate, depth=fusion_depth, residual=module_residual,act_func=nn.GELU,multi_feature=False, domain_num=domain_num, use_layerscale=use_layerscale))
            dim *=2

        self.residual = residual
        for idx, out_chan in enumerate(out_channels):
            setattr(self, 'mask_linear_lst_{}'.format(idx), nn.ModuleList([nn.Linear(embed_dim//down_scale*(2**i), out_chan) for i in range(len(depths))]))
    def forward(self, xs, H, W, domain_idx=0):
        mask_features = [down(embedding(f)) for f, down in zip(xs, self.mask_trans)]

        mask_x = mask_features[-1]

        mask_out = [mask_x]

        for i in range(len(mask_features)-2,-1,-1):
            mask_x = self.mask_fam[i](mask_features[i], mask_x, domain_idx=domain_idx)
            res_mask = mask_x
            if self.residual:
                mask_x += res_mask
            mask_out.append(mask_x)
        mask_out_unembed = []
        for i, mask  in enumerate(mask_out):
            m = unembedding(getattr(self, 'mask_linear_lst_{}'.format(domain_idx))[len(mask_out)-i-1](mask), H*(2**i)//32, W*(2**i)//32)
            m = F.interpolate(m, size=(H,W), mode='bilinear', align_corners=True)
            mask_out_unembed.append(m)
        return mask_out_unembed[-1]


class DATR(nn.Module):
    def __init__(self,
                 img_size=256,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 out_channel_list=[1], 
                 residual=False,
                 multi_feature=True,
                 down_scale=4,
                 checkpoint_path='',
                 fusion_depth=1,
                 base_path='',
                 module_residual=False,
                 domain_num=1,
                 attn_type='origin',
                 use_layerscale=False,
                 decoder='conv',
                ):
        super().__init__()
        domain_num = len(out_channel_list)

        self.encoder = SwinTransformer(
                 img_size=img_size,
                 patch_size=patch_size,
                 in_chans=in_channels,
                 embed_dim=embed_dim,
                 depths=depths,
                 num_heads=num_heads,
                 window_size=window_size,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate,
                 norm_layer=nn.LayerNorm,
                 ape=ape,
                 patch_norm=patch_norm,
                 out_indices=out_indices,
                 domain_num=domain_num,
                 attn_type=attn_type,
                 use_layerscale=use_layerscale,
                  )
        if decoder == 'transformer':
            self.decoder =  TransDecoder(out_channel_list, embed_dim, depths, down_scale, attn_drop_rate, drop_rate, fusion_depth, residual, module_residual, domain_num, use_layerscale)
            print('building transformer decoder')
        else:
            print('building convolution decoder')
            self.decoder = ConvDecoder(embed_dim, out_channel_list, domain_num)
        
        self.base_path = base_path
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.domain_num = domain_num
        self.my_init()


    def forward(self, x, domain_idx=0):
        if x.size(1)==1:
            x = x.repeat(1,3,1,1)

        B, C, H, W = x.shape
        features = self.encoder(x, domain_idx=domain_idx)
        x = self.decoder(features, H, W, domain_idx)
        return {'heatmap': torch.sigmoid(x)}

    def my_init(self):
        def gen_para_dic(swin_paras, domain_num=3):
            new_dic = {}
            for para, val in swin_paras.items():
                if 'qkv' in para:
                    dim = val.size(0)//3
                    for idx, tp in enumerate(['q', 'k', 'v']):
                        para_str_lst = [tp]
                        for num in range(domain_num):
                            para_str_lst.append('{}.{}'.format(tp,num))
                        for para_str in para_str_lst:
                            new_para = para.replace('qkv', para_str)
                            new_dic[new_para] = val[dim*idx:dim*(idx+1)]
                else:
                    new_dic[para] = val
            return new_dic
        if os.path.exists(self.checkpoint_path):
            print('loading: {}'.format(self.checkpoint_path))
            self.load_state_dict(torch.load(self.checkpoint_path),strict=True)
        else:
            weight_init(self)
            if os.path.exists(self.base_path):
                print('initialize: {}'.format(self.base_path))
                para_dic = torch.load(self.base_path)['model']
                new_para_dic = gen_para_dic(para_dic, self.domain_num)
                
                # key = 'layers.2.blocks.3.attn.q.weight'
                # print('before', self.encoder.state_dict()[key][:3,:3])
                # print('before', self.encoder.state_dict()[key.replace('weight', 'bias')][:3])
                self.encoder.load_state_dict(new_para_dic,strict=False)
                # print('after', self.encoder.state_dict()[key][:3,:3])
                # print('after', self.encoder.state_dict()[key.replace('weight', 'bias')][:3])
            else:
                print('Swin Transformer checkpoint doesn\'t exist: {}'.format(self.base_path))


if __name__ == '__main__':
    img = torch.randn(4,1,256,256)
    model = DATR(domain_num=3, out_channel_list=[1,100,2], decoder='conv')
    out = model(img,1 )['heatmap']
    print(out.shape)
