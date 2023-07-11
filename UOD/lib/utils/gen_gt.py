import os
import math
import yaml
import json
from itertools import product
from functools import partial
from collections.abc import Iterable

import torch
import numpy as np


def gen_heatmap(image_size,coords,sigma,alpha, dtype=np.float32):

    heatmap = np.zeros(image_size, dtype=dtype)
    size_sigma_factor = 10

    # flip point from [x, y, z] to [z, y, x]
    flipped_coords = np.flip(coords, 0)
    region_start = (flipped_coords - sigma * size_sigma_factor / 2).astype(int)
    region_end = (flipped_coords + sigma * size_sigma_factor / 2).astype(int)
    region_start = np.maximum(0, region_start).astype(int)
    region_end = np.minimum(image_size, region_end).astype(int)

    # return zero landmark, if region is invalid, i.e., landmark is outside of image
    if np.any(region_start >= region_end):
        return heatmap

    region_size = (region_end - region_start).astype(int)

    dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
    x_diff = dx + region_start[0] - flipped_coords[0]
    y_diff = dy + region_start[1] - flipped_coords[1]

    squared_distances = x_diff * x_diff + y_diff * y_diff

    cropped_heatmap = np.exp(-squared_distances / (2 * (sigma** 2)))

    heatmap[region_start[0]:region_end[0],
    region_start[1]:region_end[1]] = cropped_heatmap[:, :]

    #
    heatmap = np.power(alpha, heatmap)
    heatmap[heatmap <= 1.05] = 0

    return heatmap

def gaussianHeatmap(sigma, dim: int = 2, nsigma: int = 3):
    if nsigma <= 2:
        print('[Warning]: nsigma={} is recommended to be greater than 2'.format(nsigma))
    radius = round(nsigma*sigma)
    center = tuple([radius for i in range(dim)])
    mask_shape = tuple([2*radius for i in range(dim)])
    mask = np.zeros(mask_shape, dtype=np.float)
    sig2 = sigma**2
    coef = sigma*np.sqrt(2*np.pi)
    for p in product(*[range(i) for i in mask_shape]):
        d2 = sum((i-j)**2 for i, j in zip(center, p))
        mask[p] = np.exp(-d2/sig2/2)/coef
    mask = (mask-mask.min())/(mask.max()-mask.min()) # necessary? the output heatmap is processed with sigmoid
    # mask = mask/mask.sum() # necessary? the output heatmap is processed with sigmoid

    def genHeatmap(points, shape, is_single=True):
        if is_single:
            points = [points]
        chan = 1 if is_single else len(points)
        ret = np.zeros((chan,*shape), dtype=np.float)
        for i, point in enumerate(points):
            if sigma==1:
                ret[i][point] = 1
            else:
                bboxs = [(max(0, point[ax]-radius), min(shape[ax], point[ax]+radius))
                         for ax in range(dim)]
                img_sls = tuple([slice(i, j) for i, j in bboxs])
                mask_begins = [max(0, radius-point[ax]) for ax in range(dim)]
                mask_sls = tuple([slice(beg, beg+j-i)
                                  for beg, (i, j) in zip(mask_begins, bboxs)])
                ret[i][img_sls] = mask[mask_sls]
        if is_single:
            ret = ret[0]
        return ret
    return genHeatmap


def heatmapOffset(Radius):
    mask_tmp = np.zeros((2*Radius, 2*Radius), dtype=np.float)
    heatmap_tmp = np.zeros((2*Radius, 2*Radius), dtype=np.float)
    for i in range(2*Radius):
        for j in range(2*Radius):
            distance = np.linalg.norm([i+1 - Radius, j+1 - Radius])
            if distance < Radius:
                mask_tmp[i][j] = 1
                heatmap_tmp[i][j] = math.exp(-0.5 * math.pow(distance, 2) /math.pow(Radius, 2))
    
    # gen offset
    x_tmp = np.zeros((2*Radius, 2*Radius), dtype=np.float)
    y_tmp = np.zeros((2*Radius, 2*Radius), dtype=np.float)
    for i in range(2*Radius):
        x_tmp[:, i] = Radius - i
        y_tmp[i, :] = Radius - i
    x_tmp = x_tmp * mask_tmp / Radius
    y_tmp = y_tmp * mask_tmp / Radius

    def genGT(landmarks, shape, is_single=True):
        y, x = shape
        chan = 1 if is_single else len(landmarks)
        if is_single:
            landmarks = [landmarks]
        mask = np.zeros((chan, y, x), dtype=np.float)
        heatmap = np.zeros((chan, y, x), dtype=np.float)
        offset_x = np.zeros((chan, y, x), dtype=np.float)
        offset_y = np.zeros((chan, y, x), dtype=np.float)
        for i, landmark in enumerate(landmarks):

            margin_x_left = max(0, landmark[0] - Radius)
            margin_x_right = min(x, landmark[0] + Radius)
            margin_y_bottom = max(0, landmark[1] - Radius)
            margin_y_top = min(y, landmark[1] + Radius)

            mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = mask_tmp[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            heatmap[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = heatmap_tmp[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = x_tmp[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = y_tmp[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        if is_single:
            mask = mask[0]
            heatmap = heatmap[0]
            offset_y = offset_y[0]
            offset_x = offset_x[0]
        return mask, heatmap, offset_y, offset_x
    return genGT


def gen_actionmap(point, shape, onehot=False):
    ''' Left=0, Up=1, Right=2, Down=3 '''
    assert len(shape)==len(point)==2, 'dim should be 2, not {}'.format(len(shape))
    actionmap = np.zeros(shape, dtype=np.int)
    for idx in product(*[range(i) for i in shape]):
        judge_add = (idx[0]-point[0]) + (idx[1]-point[1]) #  line: idx[0]+idx[1]=0
        judge_sub = (idx[0]-point[0]) - (idx[1]-point[1]) #  line: idx[0]-idx[1]=0
        if judge_sub>=0 and judge_add<0:
            actionmap[idx]=0
        elif judge_sub<0 and judge_add<=0:
            actionmap[idx]=1
        elif judge_sub<=0 and judge_add>0:
            actionmap[idx]=2
        elif judge_sub>0 and judge_add>=0:
            actionmap[idx]=3
        elif judge_add==judge_sub==0: # the location of landmark
            actionmap[idx]=4
        else:
            raise Exception("Error: {}, {}".format(judge_sub, judge_add))
    if onehot:
        actionmap = np.stack([actionmap==i for i in [0,1,2,3]],axis=0).astype(np.int)
    return actionmap


def aggregate_actionmap(actionmap):
    ''' actionmap np.array: HxW, contains {0,1,2,3} '''
    dire_num = 4
    H, W = actionmap.shape

    flagmap = np.stack([actionmap.astype(np.int)==i for i in range(dire_num)], axis=0)
    countmap = np.zeros((dire_num, H, W), dtype=np.int)

    # initial first row and column
    for i, j in  [(0, j) for j in range(W)] + [(i, 0) for i in range(H)]:
        for dire in range(dire_num):
            sls_pos = (i, slice(j+1,None) ) if dire in [0, 2] else (slice(i+1,None), j )  # row or column
            sls_neg = (i, slice(0, j) ) if dire in [0, 2] else (slice(0, i), j )
            countmap[dire][i,j] = flagmap[dire][sls_pos].sum() - flagmap[dire][sls_neg].sum()

    # dp: dynamic programming
    for i in range(H):
        for j in range(W):
            if i==0 or j==0:
                continue
            for dire in range(dire_num):
                sls = (i, j-1) if dire in [0, 2] else (i-1, j)  # row or column
                countmap[dire][i,j] = countmap[dire][sls] - flagmap[dire][sls] - flagmap[dire][i,j]

    countmap[0] = countmap[0] * -1
    countmap[1] = countmap[1] * -1
    countmap[2] = countmap[2] * 1
    countmap[3] = countmap[3] * 1
    return countmap


def get_offset_from_heatmap(heatmaps, points):
    ''' heatmaps: NxHxWxD '''
    N, *img_size = heatmaps.shape
    dim = len(img_size)
    offsets = np.zeros((dim*N, *img_size))
    for i, (pt, heatmap) in enumerate(zip(points, heatmaps)):
        for idx in zip(*np.where(heatmap>0)):
            for d in range(dim):
                offsets[dim*i+d][idx] = idx[d]-pt[d]
    return offsets

def weight_coord(arr, multi_channel=True, use_ratio=True, threshold=0):
    '''
        arr::[channel]xWxHxDx...
        => coord_list::[(W,H,D,...)]
    '''
    arr = np.array(arr)
    if not multi_channel:
        arr = np.expand_axis(arr, axis=0)
    arr_uni = np.array([chan/chan.sum() for chan in arr])

    channel_num, *shape = arr_uni.shape
    coord_list = np.zeros((channel_num, len(shape)))
    for c in range(channel_num):
        for idx in zip(*np.where(arr_uni[c]>threshold)):
            coord_list[c] += arr_uni[c][idx]*np.array(idx)
    if use_ratio:
        coord_list = [tuple((coord/np.array(shape)).tolist()) for coord in coord_list]
    return coord_list


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def argmax_coord(arr, multi_channel=True, use_ratio=True):
    ''' 
        arr: numpy.ndarray, channel x imageshape
        coord_lst: [(x,y..)]* channel
    '''
    arr = np.array(arr)
    if not multi_channel:
        arr = np.expand_axis(arr, axis=0)
    shape = arr.shape[1:]
    coord_lst = []
    for img in arr:
        index = img.argmax()
        coord = unravel_index(index, img.shape)
        if use_ratio: 
            coord = tuple([p/1.0/s for p,s in zip(coord, shape)])
        coord_lst.append(coord)
    return coord_lst

