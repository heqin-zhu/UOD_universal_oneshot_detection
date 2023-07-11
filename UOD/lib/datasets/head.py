import os
import json
import random
from PIL import Image

from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage

from ..utils import gen_heatmap, gen_actionmap, heatmapOffset


class Head(Dataset):
    def __init__(self, img_dir, gt_dir, img_size, num_landmark=19, sigma=10, alpha=40, use_actionmap=False, use_offset=False, img_size_highres=None, highres_factor=1, phase='train', cache_dir='.cache', test_aug_dic={}, pseudo_path='', train_with_pseudo=False, few_shot=None):

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
        self.highres_factor = highres_factor

        self.img_names = os.listdir(img_dir)

        if self.phase == 'train' and few_shot:
            assert 0 < few_shot <= len(self.img_names), 'few_shot: {}, len: {}'.format(few_shot, len(self.img_names))
            self.img_names = self.img_names[:few_shot]


        self.img_size = img_size
        self.height, self.width = self.img_size
        if  img_size_highres is None:
            self.img_size_highres = (2400, 2400)
        else:
            self.img_size_highres = img_size_highres
        self.height_highres, self.width_highres = self.img_size_highres

        self.transform = iaa.Sequential([
            iaa.PadToAspectRatio(self.width/self.height, position='right-bottom'),
            iaa.Resize({"width": self.width, "height": self.height}),
        ])
        self.transform_highres = iaa.Sequential([
            iaa.PadToAspectRatio(self.width_highres/self.height_highres, position='right-bottom'),
            iaa.Resize({"width": self.width_highres, "height": self.height_highres}),
        ])


        aug_lst = []
        for aug, val in test_aug_dic.items():
            if aug == 'blur':
                aug_lst.append(iaa.GaussianBlur(sigma=val))
            else:
                raise KeyError
        self.test_aug = iaa.Sequential(aug_lst)
        self.test_aug_dic = test_aug_dic

        self.transform_aug = iaa.Sequential([
            # iaa.PadToAspectRatio(width/height, position='right-bottom'),
            # iaa.Resize({'width':width, 'height':height}),
            # iaa.Fliplr(0.5), # horizontal flips 50%
            # iaa.Flipud(0.2), # vertically flip 20% of all images
            # iaa.Crop(percent=(0, 0.1)), # random crops

            # Small gaussian blur with random sigma between 0 and 3.
            # But we only blur about 50% of all images.
            # iaa.Sometimes(
            #     0.5,
            #     iaa.GaussianBlur(sigma=(0.1, 3))
            # ),

            # Strengthen or weaken the contrast in each image.
            # iaa.LinearContrast((0.75, 1.5)),

            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5),

            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            # iaa.Multiply((0.8, 1.2), per_channel=0.2),

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
            transforms.Normalize([0.482208], [0.292954])
        ]
        )

        self.use_actionmap = use_actionmap
        self.use_offset = use_offset
        if use_offset:
            self.heatmapOffset = heatmapOffset(int(self.height*0.1))

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

        # TODO
        # if self.phase == 'train':
        #     # augmentation
        #     img = self.transform_aug(image=(img*255).astype(np.uint8))
        #     img = (img - img.min())/(img.max()-img.min())

        if len(self.test_aug_dic)!=0:
            img = self.test_aug(image=(img*255).astype(np.uint8))
            img = (img - img.min())/(img.max()-img.min())

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

        # origin landmark
        gt_file = os.path.join(self.gt_dir, img_name_wo_ext + '.txt')
        gt_origin = np.loadtxt(gt_file, delimiter=",")[:-1] # the last row is spacing


        # pseudo label
        if self.train_with_existed_pseudo:
            pseudo_file = os.path.join(self.pseudo_path, img_name_wo_ext + '.json')
            gt_origin = np.zeros((self.num_landmark, 2), dtype=float)
            with open(pseudo_file) as f:
                point_dic = json.loads(f.read())
                for n in range(self.num_landmark):
                    x, y = point_dic[str(n)]
                    gt_origin[n] = x/384*1935, y/384*2400


        # stg1 size, low res
        img, gt = self.get_image_and_landmark(img_name_wo_ext, img_origin, self.transform, gt_origin, self.img_size)

        # stg2 size, high res 
        img_highres, gt_highres = self.get_image_and_landmark(img_name_wo_ext, img_origin, self.transform_highres, gt_origin, self.img_size_highres)

        ret = {
               'name': img_name, 
               'image':img,
               'image_highres':img_highres,
               'landmark': gt,
               'landmark_highres': gt_highres,
              }

        ret['heatmap'] = np.stack([gen_heatmap(self.img_size, gt[i], self.sigma, self.alpha) for i in range(self.num_landmark)], axis=0)
        ret['heatmap_highres'] = np.stack([gen_heatmap(self.img_size_highres, gt_highres[i], self.sigma*self.highres_factor, self.alpha) for i in range(self.num_landmark)], axis=0)
        if self.use_actionmap:
            act_ratio = 1
            cur_size = tuple([sz//act_ratio for sz in self.img_size])
            ret['actionmap'] = np.concatenate([gen_actionmap(tuple([p/act_ratio for p in gt[i]]), cur_size, onehot=True) for i in range(self.num_landmark)], axis=0)
        if self.use_offset:
            mask, heatmap_2, offset_y, offset_x = self.heatmapOffset(gt.astype(np.int), self.img_size, is_single=False)
            C, *size = offset_y.shape
            offset = np.array([(y, x) for y, x in zip(offset_y, offset_x)])
            ret['offset'] = offset.reshape(-1, *size)

        for k in ret:
            if k not in ['name']:
                ret[k] = torch.FloatTensor(ret[k])
        return ret
