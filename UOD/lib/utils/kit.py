import os
import math
import yaml
import json
from itertools import product
from functools import partial
from collections.abc import Iterable

import torch
import numpy as np


def mkdir(path):
    def _mkdir(p):
        if p and not os.path.exists(p):
            _mkdir(p[:p.rfind(os.path.sep)])
            os.mkdir(p)
    _mkdir(os.path.abspath(path))


def getName(path):
    path = os.path.basename(path)
    n = path.rfind('.')
    if n != -1:
        path = path[:n]
    return path


def get_partition(path, phase):
    ''' 
        path: yaml file contains: {'train': [], 'validate':[], 'test':[]}
    '''
    with open(path) as f:
        return yaml.load(f.read())[phase]


def readDir(path, recursive=True, isSorted=True, filter_func=None):
    def _readDir(path, recursive=recursive):
        li = os.listdir(path)
        fs = []
        ds = []
        for i in li:
            f = os.path.join(path, i)
            if os.path.isfile(f):
                if filter_func is None or filter_func(f):
                    fs.append(f)
            else:
                ds.append(f)
        if recursive:
            for d in ds:
                fs += _readDir(d, recursive)
        return fs
    fs = _readDir(path, recursive)
    if isSorted:
        fs.sort()
    return fs


def norm(x, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        vmin, vmax = x.min(), x.max()
    else:
        x[x < vmin] = vmin
        x[x > vmax] = vmax
    if vmin == vmax:
        return x
    else:
        return (x-vmin)/(vmax-vmin)


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


def genPoints(start, end, n=6, min_ratio=0, max_ratio=1):
    '''
        start,end are n-dim points of a line
    '''
    start, end = np.array(start), np.array(end)
    diff = end-start
    for i in np.linspace(min_ratio, max_ratio, n):
        yield tuple((start+diff*i+0.5).astype(np.int16).tolist())


def np2py(obj):
    if isinstance(obj, dict):
        return {k:np2py(v) for k,v in obj.items()}
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, np.generic):
        return np.asscalar(obj)
    elif isinstance(obj, Iterable):
        return [np2py(x) for x in obj]
    else:
        return obj
