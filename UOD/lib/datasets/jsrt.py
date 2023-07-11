import os
import json
import random
from PIL import Image

import pandas as pd
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage

from ..utils import gen_heatmap, gen_actionmap, heatmapOffset


def read_landmark(path):
    landmarks = []
    with open(path) as f:
        lines = list(f.readlines())
        idx = 0
        n = len(lines)
        while idx<n:
            line = lines[idx]
            if 'right lung' in line: # 44/4
                # for i in range(idx+2, idx+2+44, 4): # TODO
                for i in range(idx+2, idx+2+44):
                    if i-idx-2 not in [0, 21, 29]:
                        continue
                    line = ''.join([c for c in lines[i] if c not in ' \n{}'])
                    x, y = line.rstrip(',').split(',')
                    x = float(x)
                    y = float(y)
                    landmarks.append((x,y))
                idx += 2+44
            elif 'left lung' in line: # 50/5
                # for i in range(idx+2, idx+2+50, 4):
                for i in range(idx+2, idx+2+50):
                    if i-idx-2 not in [0, 21, 27]:
                        continue
                    line = ''.join([c for c in lines[i] if c not in ' \n{}'])
                    x, y = line.rstrip(',').split(',')
                    x = float(x)
                    y = float(y)
                    landmarks.append((x,y))
                idx += 2+50
            else:
                idx+=1
    # !!! TODO
    landmarks = [(2*x, 2*y) for x, y in landmarks]
    return landmarks


class JSRT(Dataset):
    def __init__(self, img_dir, gt_dir, img_size, num_landmark, sigma, alpha, phase='train', cache_dir='.cache_jsrt', pseudo_path='', train_with_pseudo=False, few_shot=None, **kargs):
        # pseudo_label
        self.pseudo_path = pseudo_path

        self.phase = phase

        self.train_with_existed_pseudo = False
        if self.phase == 'train' and train_with_pseudo:
            if os.path.isdir(pseudo_path):
                print('[NOTICE] Training with Pseudo Labels from "{}"'.format(pseudo_path))
                self.train_with_existed_pseudo = True
            else:
                print('[NOTICE] Pseudo path "{}" does not exist'.format(pseudo_path))


        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.num_landmark = num_landmark
        self.sigma = sigma
        self.alpha = alpha

        self.original_size = (2048, 2048)

        img_names = sorted(os.listdir(img_dir))
        # 247 images  2048x2048
        total_num = len(img_names)
        nn_lst = [f for f in img_names if f.startswith('JPCNN')]
        ln_lst = [f for f in img_names if f.startswith('JPCLN')]
        assert total_num == 247 == len(ln_lst) + len(nn_lst)
        test_ratio = 0.2
        test_num = 50

        nn_test_num = int(len(nn_lst)*test_ratio)
        ln_test_num = test_num - nn_test_num

        if phase == 'train':
            self.img_names = nn_lst[:-nn_test_num] + ln_lst[:-ln_test_num]
            if few_shot:
                assert 0 < few_shot <= len(self.img_names), 'few_shot: {}, len: {}'.format(few_shot, len(self.img_names))
                self.img_names = self.img_names[:few_shot]

        else:
            self.img_names = nn_lst[-nn_test_num:] + ln_lst[-ln_test_num:]

        self.img_size = img_size
        self.height, self.width = self.img_size

        self.transform = iaa.Sequential([
            # iaa.PadToAspectRatio(self.width/self.height, position='right-bottom'),
            iaa.Resize({"width": self.width, "height": self.height}),
        ])
        self.transform_aug = iaa.Sequential([
            # iaa.PadToAspectRatio(width/height, position='right-bottom'),
            # iaa.Resize({'width':width, 'height':height}),
            # iaa.Fliplr(0.5), # horizontal flips 50%
            # iaa.Flipud(0.2), # vertically flip 20% of all images
            # iaa.Crop(percent=(0, 0.1)), # random crops

            # Small gaussian blur with random sigma between 0 and 3.
            # But we only blur about 50% of all images.
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 3))
            ),

            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),

            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            # iaa.Affine(
            #       translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
            #       # scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            #       # rotate=(-25, 25),
            #       # shear=(-4, 4)
            # )
        ], random_order=False) # if apply augmenters in random order?


        self.norm_trans = transforms.Compose([
            transforms.Normalize([0.40305], [0.27195])
        ]
        )

        self.cache_prefix = cache_dir

    def __len__(self):
        return len(self.img_names)


    def get_image_and_landmark(self, img_name, img_origin, transform, gt_origin, cur_size):
        cache_dir = os.path.join(self.cache_prefix, '{}_{}'.format(*cur_size))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_img_path = os.path.join(cache_dir, img_name+'.npy')
        cache_label_path = os.path.join(cache_dir, img_name+'_landmark.npy')

        img = landmark = None

        if os.path.exists(cache_img_path):
            img = np.load(cache_img_path)
        else:
            img = transform(image=img_origin)
            np.save(cache_img_path, img)

        img = np.expand_dims(img, axis=0)
        img = self.norm_trans(torch.FloatTensor(img))

        if os.path.exists(cache_label_path) and False:
            raise Excpetion('Don\'t use cache label')
            landmark = np.load(cache_label_path)
        else:
            keypoints = KeypointsOnImage.from_xy_array(gt_origin, shape=img_origin.shape)
            landmark = transform(keypoints=keypoints).to_xy_array()
            # np.save(cache_label_path, landmark)
        return img, landmark
        



    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_name_wo_ext = img_name.split('.')[0]

        # get image
        img_file = os.path.join(self.img_dir, img_name)
        img_origin = io.imread(img_file, as_gray=True)

        ori_size = img_origin.shape

        # origin landmark
        gt_origin = read_landmark(os.path.join(self.gt_dir, img_name_wo_ext+'.pfs'))


        # pseudo label
        if self.train_with_existed_pseudo:
            pseudo_file = os.path.join(self.pseudo_path, img_name_wo_ext + '.json')
            gt_origin = np.zeros((self.num_landmark, 2), dtype=float)
            with open(pseudo_file) as f:
                point_dic = json.loads(f.read())
                for n in range(self.num_landmark):
                    x, y = point_dic[str(n)]
                    gt_origin[n] = x/384*ori_size[0], y/384*ori_size[1]

        # stg1 size, low res
        img, gt = self.get_image_and_landmark(img_name_wo_ext, img_origin, self.transform, gt_origin, self.img_size)

        ret = {
               'name': img_name, 
               'image':img,
               'landmark': gt,
              }

        # assert len(gt)==6,f'{len(gt)}, {img_file}'
        ret['heatmap'] = np.stack([gen_heatmap(self.img_size, gt[i], self.sigma, self.alpha) for i in range(self.num_landmark)], axis=0)
        for k in ret:
            if k not in ['name']:
                ret[k] = torch.FloatTensor(ret[k])
        return ret
