import os
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .jsrt import read_landmark
from .augment import cc_augment
from .entropy_loss import get_guassian_heatmaps_from_ref


def to_PIL(tensor):
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8).transpose(1, 2, 0))
    return images


def augment_patch(tensor, aug_transform):
    image = to_PIL(tensor)
    aug_image = aug_transform(image)
    return aug_image


class JSRT_SSL(data.Dataset):
    def __init__(self, pathDataset, mode, size=[384, 384], be_consistency=False, patch_size=192, pre_crop=False, rand_psize=False, ref_landmark=None, prob_map=None, retfunc=1, use_prob=True, ret_name=False, num_repeat=0):
        assert not (ref_landmark is not None and prob_map is not None), f"Got both ref_landmark: \n{ref_landmark} \nand prob_map: \n{prob_map}"

        self.size = size
        self.retfunc = retfunc
        self.original_size = (2048, 2048)
        if retfunc > 0:
            print("Using new ret Function!")
            self.new_ret = True
        self.pth_Image = os.path.join(pathDataset, 'imgs')
        self.gt_dir = os.path.join(pathDataset, 'labels')

        img_names = sorted(os.listdir(self.pth_Image))
        # 247 images  2048x2048
        total_num = len(img_names)
        nn_lst = [f for f in img_names if f.startswith('JPCNN')]
        ln_lst = [f for f in img_names if f.startswith('JPCLN')]
        assert total_num == 247 == len(ln_lst) + len(nn_lst)
        test_ratio = 0.2
        test_num = 50

        nn_test_num = int(len(nn_lst)*test_ratio)
        ln_test_num = test_num - nn_test_num

        cap_mode = mode.capitalize()
        if cap_mode in ['Oneshot', 'Train']:
            self.img_names = nn_lst[:-nn_test_num] + ln_lst[:-ln_test_num]
        elif cap_mode == 'Test':
            self.img_names = nn_lst[-nn_test_num:] + ln_lst[-ln_test_num:]
        else:
            raise Exception('Unkown mode "{}"'.format(mode))

        self.patch_size = patch_size
        self.pre_crop = pre_crop
        self.ref_landmark = ref_landmark
        self.prob_map = prob_map
        self.ret_name = ret_name
        if rand_psize:
            self.rand_psize = rand_psize
            self.patch_size = -1

        normalize = transforms.Normalize([0], [1])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        # transforms.ColorJitter(brightness=0.15, contrast=0.25),
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        transform_list = [
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ]
        self.aug_transform = transforms.Compose(transform_list)

        self.mode = mode
        self.base = 16
        self.be_consistency = be_consistency

        if use_prob: # prob map setting
            print("Using Retfunc2() and Prob map")
            assert retfunc == 2, f" Retfunc Error, Got {retfunc}"
            if prob_map is None:
                self.prob_map = self.prob_map_from_landmarks(self.size)

    def set_rand_psize(self):
        self.patch_size = np.random.randint(6, 8) * 32


    def prob_map_from_landmarks(self, size=(384, 384), kernel_size=192):
        """
        Guassion Prob map from landmarks
        landmarks: [(x,y), (), ()....]
        size: (384,384)
        """
        landmarks = self.ref_landmark
        prob_maps = get_guassian_heatmaps_from_ref(landmarks=landmarks, num_classes=len(landmarks), image_shape=size, kernel_size=kernel_size, sharpness=0.2)
        prob_map = np.sum(prob_maps, axis=0)
        prob_map = np.clip(prob_map, 0, 1)
        print("====== Save Prob map to ./imgshow")
        cv2.imwrite(f"imgshow/prob_map_ks{kernel_size}.jpg", (prob_map*255).astype(np.uint8) )
        return prob_map

    def select_point_from_prob_map(self, prob_map, size=(192, 192)):
        size_x, size_y = prob_map.shape
        assert size_x == size[0]
        assert size_y == size[1]
        chosen_x1 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_x2 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y1 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        chosen_y2 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        if prob_map[chosen_x1, chosen_y1] * np.random.random() > prob_map[chosen_x2, chosen_y2] * np.random.random() :
            return chosen_x1, chosen_y1
        else:
            return chosen_x2, chosen_y2

    def __getitem__(self, index):
        if self.retfunc == 1:
            return self.retfunc1(index)
        elif self.retfunc == 2:
            return self.retfunc2(index)
        elif self.retfunc == 3:
            return self.retfunc3(index)
        elif self.retfunc == 0:
            return self.retfunc_old(index)

    def __len__(self):
        return len(self.img_names)

    def retfunc1(self, index):
        """
        New Point Choosing Function without prob map
        """
        np.random.seed()
        img_name = self.img_names[index]
        pth_img = os.path.join(self.pth_Image, img_name)
        img = Image.open(pth_img).convert('RGB')
        if self.transform != None:
            img = self.transform(img)

        pad_scale = 0.05
        padding = int(pad_scale*self.size[0])
        patch_size = self.patch_size
        raw_x = np.random.randint(int(pad_scale * self.size[0]), int((1-pad_scale) * self.size[0]))
        raw_y = np.random.randint(int(pad_scale * self.size[1]), int((1-pad_scale) * self.size[1]))

        b1_left = 0
        b1_top = 0
        b1_right = self.size[0] - patch_size
        b1_bot = self.size[1] - patch_size
        b2_left = raw_x-patch_size+1
        b2_top = raw_y-patch_size+1
        b2_right = raw_x
        b2_bot = raw_y
        b_left = max(b1_left, b2_left)
        b_top  = max(b1_top, b2_top)
        b_right = min(b1_right, b2_right)
        b_bot = min(b1_bot, b2_bot)
        left = np.random.randint(b_left, b_right)
        top = np.random.randint(b_top, b_bot)

        margin_x = left
        margin_y = top
        cimg = img[:, margin_y:margin_y + patch_size, margin_x:margin_x + patch_size]
        crop_imgs = augment_patch(cimg, self.aug_transform)
        chosen_x, chosen_y = raw_x - margin_x, raw_y - margin_y

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y, chosen_x] = 1
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]
        chosen_y, chosen_x = temp.argmax() // patch_size, temp.argmax() % patch_size
        if self.ret_name:
            return img, crop_imgs, chosen_y, chosen_x, raw_y, raw_x,  img_name
        return img, crop_imgs, chosen_y, chosen_x, raw_y, raw_x

    def retfunc2(self, index):
        """
        New Point Choosing Function with 'PROB MAP'
        """
        np.random.seed()
        img_name = self.img_names[index]
        pth_img = os.path.join(self.pth_Image, img_name)
        img = Image.open(pth_img).convert('RGB')
        if self.transform != None:
            img = self.transform(img)

        padding = int(0.1*self.size[0])
        patch_size = self.patch_size
        raw_x, raw_y = self.select_point_from_prob_map(self.prob_map, size=self.size)
        
        b1_left = 0
        b1_top = 0
        b1_right = self.size[0] - patch_size
        b1_bot = self.size[1] - patch_size
        b2_left = raw_x-patch_size+1
        b2_top = raw_y-patch_size+1
        b2_right = raw_x
        b2_bot = raw_y
        b_left = max(b1_left, b2_left)
        b_top  = max(b1_top, b2_top)
        b_right = min(b1_right, b2_right)
        b_bot = min(b1_bot, b2_bot)
        left = np.random.randint(b_left, b_right)
        top = np.random.randint(b_top, b_bot)
        
        margin_x = left
        margin_y = top
        cimg = img[:, margin_y:margin_y + patch_size, margin_x:margin_x + patch_size]
        crop_imgs = augment_patch(cimg, self.aug_transform)
        chosen_x, chosen_y = raw_x - margin_x, raw_y - margin_y

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y, chosen_x] = 1
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]
        chosen_y, chosen_x = temp.argmax() // patch_size, temp.argmax() % patch_size
        return img, crop_imgs, chosen_y, chosen_x, raw_y, raw_x

    def retfunc3(self, index):
        """
        Two Crops Consistency
        """
        padsize = (32,32)
        np.random.seed()
        img_name = self.img_names[index]
        pth_img = os.path.join(self.pth_Image, img_name)
        img = Image.open(pth_img).convert('RGB')
        if self.transform != None:
            img = self.transform(img)

        p0, p1, (x0, x1, y0, y1) = self.select_dual_points(self.size, (32,32))
        crop_img1 = TF.crop(img, *p0, *padsize)
        crop_img2 = TF.crop(img, *p1, *padsize)

        crop_img1 = self.transform_tensor(crop_img1)
        crop_img2 = self.transform_tensor(crop_img2)
        return crop_img1, crop_img2, p0, p1, (x0, x1, y0, y1)

    def retfunc_old(self, index):
        img_name = self.img_names[index]
        pth_img = os.path.join(self.pth_Image, img_name)
        img = Image.open(pth_img).convert('RGB')

        if self.transform != None:
            img = self.transform(img)

        # Crop 192 x 192 Patch
        # patch_size = int(0.5 * self.size[0])
        patch_size = self.patch_size
        margin_x = np.random.randint(0, self.size[0] - patch_size)
        margin_y = np.random.randint(0, self.size[0] - patch_size)
        crop_imgs = augment_patch(img[:, margin_y:margin_y + patch_size, margin_x:margin_x + patch_size], self.aug_transform)

        chosen_x_raw = np.random.randint(int(0.1 * patch_size), int(0.9 * patch_size))
        chosen_y_raw = np.random.randint(int(0.1 * patch_size), int(0.9 * patch_size))
        raw_y, raw_x = chosen_y_raw + margin_y, chosen_x_raw + margin_x

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y_raw, chosen_x_raw] = 1
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]
        chosen_y, chosen_x = temp.argmax() // patch_size, temp.argmax() % patch_size

        to_PIL(img).save('Raw.png')
        to_PIL(crop_imgs).save('Crop.png')

        if self.be_consistency:
            crop_imgs_aug = augment_patch(img[:, margin_y:margin_y + patch_size, margin_x:margin_x + patch_size], self.aug_transform)
            temp = torch.zeros([1, patch_size, patch_size])
            temp[:, chosen_y_raw, chosen_x_raw] = 1
            temp = cc_augment(torch.cat([crop_imgs_aug, temp], 0))
            crop_imgs_aug = temp[:3]
            temp = temp[3]
            chosen_y_aug, chosen_x_aug = temp.argmax() // patch_size, temp.argmax() % patch_size
            return img, crop_imgs, chosen_y, chosen_x, raw_y, raw_x, crop_imgs_aug, chosen_y_aug, chosen_x_aug
        return img, crop_imgs, chosen_y, chosen_x, raw_y, raw_x

class Testset_JSRT_SSL(data.Dataset):
    def __init__(self, pathDataset, mode, size=[384, 384], id_oneshot='JPCLN035', pre_crop=False):

        self.num_landmark = 6
        self.size = size
        if pre_crop:
            self.size[0] = 480 #int(size[0] / 0.8)
            self.size[1] = 480 #int(size[1] / 0.8)
        self.pth_Image = os.path.join(pathDataset, 'imgs')
        self.gt_dir = os.path.join(pathDataset, 'labels')

        self.img_names = sorted(os.listdir(self.pth_Image))

        img_names = sorted(os.listdir(self.pth_Image))
        self.original_size = (2048, 2048)
        # 247 images  2048x2048
        total_num = len(img_names)
        nn_lst = [f for f in img_names if f.startswith('JPCNN')]
        ln_lst = [f for f in img_names if f.startswith('JPCLN')]
        assert total_num == 247 == len(ln_lst) + len(nn_lst)
        test_ratio = 0.2
        test_num = 50

        nn_test_num = int(len(nn_lst)*test_ratio)
        ln_test_num = test_num - nn_test_num

        cap_mode = mode.capitalize()
        if cap_mode == 'Oneshot':
            print("JSRT One shot ID: ", id_oneshot)
            id_oneshot = str(id_oneshot).upper()
            self.img_names = [id_oneshot if id_oneshot.endswith('.png' ) else id_oneshot + '.png']
        elif cap_mode == 'Train':
            self.img_names = nn_lst[:-nn_test_num] + ln_lst[:-ln_test_num]
        elif cap_mode == 'Test':
            self.img_names = nn_lst[-nn_test_num:] + ln_lst[-ln_test_num:]
        else:
            raise Exception('Unkown mode "{}"'.format(mode))


        normalize = transforms.Normalize([0], [1])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        self.mode = mode
        self.base = 16

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_name_wo_ext = img_name[:img_name.rfind('.')]
        pth_img = os.path.join(self.pth_Image, img_name)
        img = Image.open(pth_img).convert('RGB')
        ori_size = img.width, img.height

        if self.transform != None:
            img = self.transform(img)

        gt_xys = read_landmark(os.path.join(self.gt_dir, img_name_wo_ext+'.pfs'))
        landmark_list = [(int(x*self.size[0]/ori_size[0]), int(y*self.size[1]/ori_size[1])) for x, y in gt_xys]

        if self.mode not in ['Oneshot']:
            return img_name_wo_ext, img, landmark_list

        template_patches = torch.zeros([self.num_landmark, 3, 192, 192])
        landmark_list2 = []
        for id, landmark in enumerate(landmark_list):
            left = min(max(landmark[0] - 96, 0), self.size[0]-192)
            bottom = min(max(landmark[1] - 96, 0), self.size[0]-192)
            template_patches[id] = img[:, bottom:bottom+192, left:left+192]
            landmark_list2.append([landmark[0] - left, landmark[1] - bottom])
        return img, landmark_list2, template_patches, landmark_list

    def __len__(self):
        return len(self.img_names)
