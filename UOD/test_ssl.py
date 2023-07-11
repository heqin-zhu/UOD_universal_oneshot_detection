import argparse
import datetime
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import os
import json
import yaml
from PIL import Image, ImageDraw, ImageFont

from lib.datasets import Testset_Head_SSL, Testset_Hand_SSL, Testset_JSRT_SSL

from lib.utils import np2py, toYaml
from lib.utils.eval import Evaluater
from lib.utils.utils import to_Image, pred_landmarks, visualize, make_dir


def gray_to_PIL(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor  * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images

def gray_to_PIL2(tensor, pred_lm ,landmark, row=6, width=384):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(images)
    red = (255, 0, 0)
    green = (0, 255, 0)
    # red = 255
    for i in range(row):
        draw.rectangle((pred_lm[0]+i*width-2, pred_lm[1]-2, pred_lm[0]+i*width+2, pred_lm[1]+2), fill=green)
        draw.rectangle((landmark[0]+i*width-2, landmark[1]-2, landmark[0]+i*width+2, landmark[1]+2), fill=red)
    draw.line([tuple(pred_lm), tuple(landmark)], fill='green', width=0)
    # import ipdb; ipdb.set_trace()
    return images

def match_cos(feature, template):
    feature = feature.permute(1, 2, 0)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    return torch.clamp(cos_similarity, 0, 1)

def print_vars(vars:np.ndarray):
    for i, var in enumerate(vars):
        # print(var.shape)
        var = var / np.max(var) * 255
        image = Image.fromarray(var.astype(np.uint8))
        image.save(f"visuals/vars/var_{i}.jpg")
        import ipdb;ipdb.set_trace()

def get_dataset(config, dataname, mode, **kargs):
    if dataname == 'head':
        return Testset_Head_SSL(config['dataset']['head']['pth'], mode, **kargs)
    elif dataname == 'hand':
        return Testset_Hand_SSL(config['dataset']['hand']['pth'], mode, **kargs)
    elif dataname == 'jsrt':
        return Testset_JSRT_SSL(config['dataset']['jsrt']['pth'], mode, **kargs)
    else:
        raise Exception('Unknown dataset: {}'.format(datanme))

class Tester(object):
    def __init__(self, data_list, logger, config, net=None):
        self.data_loaders = []
        self.evaluaters = []
        self.data_list = data_list
        self.num_landmark_list = []
        self.size_dic = {}
        for data_name in data_list:
            dataset = get_dataset(config, data_name, mode='Train')
            self.size_dic[data_name] = dataset.size
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
            self.data_loaders.append(loader)
            self.evaluaters.append(Evaluater())
            num_landmark = config['dataset'][data_name]['num_landmarks']
            self.num_landmark_list.append(num_landmark)
        
        self.config = config
        self.model = net 
        self.logger = logger


    def test(self, net, data_name, epoch, dump_label=True, oneshot_id=None, draw=True):
        ratios = (1, 1)
        size = self.size_dic[data_name]
        if not oneshot_id:
            if data_name=='head':
                oneshot_id = '126'
            elif data_name=='hand':
                oneshot_id = '3188'
            elif data_name=='jsrt':
                oneshot_id = 'JPCLN035'
            else:
                raise Exception('Unknown data_name "{}"'.format(data_name))
        if data_name=='head':
            ratios = (0.1*2400/size[0], 0.1*1935/size[1])
        elif data_name=='hand':
            ratios = (50.0/1, 50.0/1)
        elif data_name=='jsrt':
            ratios = (0.175*2048/size[0], 0.175*2048/size[1])
        else:
            raise Exception('Unknown data_name "{}"'.format(data_name))

        idx = self.data_list.index(data_name)

        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        net.eval()
        config = self.config

        for i in range(1):
            one_shot_loader = get_dataset(self.config, data_name, mode='Oneshot', id_oneshot=oneshot_id)
            self.logger.info(f'ID Oneshot : {oneshot_id}')
            self.evaluaters[idx].reset()
            image, landmarks, template_patches, landmarks_ori = one_shot_loader.__getitem__(0)
            feature_list = list()
            image = image.cuda()
            _, features_tmp = net(image.unsqueeze(0), get_features=True, domain_idx=idx)

            # Depth
            feature_list = dict()
            for id_depth in range(6):
                tmp = list()
                for id_mark, landmark in enumerate(landmarks_ori):
                    tmpl_y, tmpl_x = landmark[1] // (2 ** (5-id_depth)), landmark[0] // (2 ** (5-id_depth))
                    mark_feature = features_tmp[id_depth][0, :, tmpl_y, tmpl_x]
                    tmp.append(mark_feature.detach().squeeze().cpu().numpy())
                tmp = np.stack(tmp)
                one_shot_feature = torch.tensor(tmp).cuda()
                feature_list[id_depth] = one_shot_feature

            for img_name_wo_ext, img, landmark_list in self.data_loaders[idx]:
                img_name_wo_ext = img_name_wo_ext[0] # extract from 1-length min batch

                img = img.cuda()
                _, features = net(img, get_features=True, domain_idx=idx)
                
                pred_landmarks_y, pred_landmarks_x = list(), list()
                for id_mark in range(one_shot_feature.shape[0]):
                    cos_lists = []
                    cos_ori_lists = []
                    final_cos = torch.ones_like(img[0,0]).cuda()
                    for id_depth in range(5):
                        cos_similarity = match_cos(features[id_depth].squeeze(),\
                             feature_list[id_depth][id_mark])
                        cos_similarity = torch.nn.functional.upsample(\
                            cos_similarity.unsqueeze(0).unsqueeze(0), \
                            scale_factor=2**(5-id_depth), mode='nearest').squeeze()
                        final_cos = final_cos * cos_similarity
                        cos_lists.append(cos_similarity)
                    final_cos = (final_cos - final_cos.min()) / (final_cos.max() - final_cos.min())
                    cos_lists.append(final_cos)
                    chosen_landmark = final_cos.argmax().item()
                    pred_landmarks_y.append(chosen_landmark // 384)
                    pred_landmarks_x.append(chosen_landmark % 384)
                    debug = torch.cat(cos_lists, 1).cpu()
                    a_landmark = landmark_list[id_mark]
                    pred_landmark = (chosen_landmark % 384, chosen_landmark // 384)

                preds = [np.array(pred_landmarks_y), np.array(pred_landmarks_x)] 
                if data_name == 'hand':
                    x1, y1 = pred_landmarks_x[0], pred_landmarks_y[0]
                    x2, y2 = pred_landmarks_x[4], pred_landmarks_y[4]
                    dis = ((x1-x2)**2+(y1-y2)**2)**0.5
                    ratios = (50.0/dis, 50.0/dis)
                self.evaluaters[idx].record(preds, landmark_list, ratios)
                
                # Optional Save viusal results
                if draw:
                    image_pred = visualize(img, preds, landmark_list)
                    draw_pth = os.path.join(config['base']['run_dir'], 'visuals', img_name_wo_ext)
                    if not os.path.exists(draw_pth):
                        os.makedirs(draw_pth)
                    image_pred.save(os.path.join(draw_pth, 'pred.png'))

                if dump_label:
                    inference_marks = {id:[int(preds[1][id]), int(preds[0][id])] for id in range(self.num_landmark_list[idx])}
                    dir_pth = os.path.join(config['base']['run_dir'], f"oneshot_id_{oneshot_id}_ep{epoch}", 'pseudo-labels_init')
                    if not os.path.exists(dir_pth):
                        os.makedirs(dir_pth)
                    with open(os.path.join('{0}/{1}.json'.format(dir_pth, img_name_wo_ext)), 'w') as f:
                        json.dump(inference_marks, f)
                    print("Dumped JSON file:" , '{0}/{1}.json'.format(dir_pth, img_name_wo_ext))

            metric_data = self.evaluaters[idx].cal_metrics()

            self.logger.info("ep{:03d} {} MRE {}".format(epoch, data_name, metric_data['MRE']))
            for th, sdr in metric_data['SDR'].items():
                self.logger.info("ep{:03d} {} SDR={} {}".format(epoch, data_name, th, sdr))

            yaml_path = os.path.join(config['base']['run_dir'], 'metrics.yaml')

            all_data = {}
            if os.path.exists(yaml_path):
                with open(yaml_path) as f:
                    all_data = yaml.load(f.read())
            all_data[data_name] = np2py(metric_data)
            toYaml(yaml_path, all_data)

            return metric_data
