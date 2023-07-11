import os
import yaml
import argparse
import random
import pickle
from collections.abc import Iterable
from itertools import product
from functools import partial
from PIL import Image
from PIL import ImageDraw, ImageFont

import warnings
warnings.filterwarnings("ignore")

import cv2
import cc3d
from tqdm import tqdm
import scipy.io as sio
import numpy as np
import SimpleITK as sitk

from lib.utils import mkdir, toYaml, radial, get_config, color, colorRGB
from lib.utils import np2py, genPoints, argmax_coord, weight_coord


DATA_PREFIX = '../data'
PATH_DIC = {
    'head': 'head/RawImage',
    'hand': 'hand/jpg',
    'jsrt': 'JSRT/imgs',
    'pelvis': 'pelvis/pngs',
    'spineweb16': 'SpineWeb16/data/test',
    'spineweb3': 'SpineWeb3/spine_uni',
    'verse': 'VerSe/verse2019_miccai_uni',
    'cthead': 'CTHead/resample_data',
}
for k in PATH_DIC:
    PATH_DIC[k] = os.path.join(DATA_PREFIX, PATH_DIC[k])

DATASET_IMAGE_SUFFIX_DIC = {
    'head': '.bmp',
    'hand': '.jpg',
    'jsrt': '.png',
    'pelvis': '.png',
    'spineweb16': '',
    'spineweb3': '.nii.gz',
    'verse': '.nii.gz',
    'cthead': '.nii',
}

GT_LABEL_SUFFIX = '_gt.txt'

FONT_PATH = './times.ttf'
SDR_THRESHOLD = [2, 2.5, 3, 4, 6, 8, 9, 10, 15, 20, 30]
CEPH_PHYSICAL_FACTOR = 0.1
JSRT_PHYSICAL_FACTOR = 0.175
WRIST_WIDTH = 50  # mm

def parse_dataname_from_path(path):
    dic = {'head': ['head', 'ceph'],
           'hand': ['hand'],
           'jsrt': ['jsrt'],
           'pelvis': ['pelvis'],
           'spineweb16': ['spineweb16'],
           'spineweb3': ['spioneweb3'],
           'verse': ['verse'],
           'cthead': ['cthead'],
          }
    base_dir = os.path.basename(path).lower()
    for k, lst in dic.items():
        if any([pat in base_dir for pat in lst]):
            return k
    raise Exception('Can not parse data name from "{}"'.format(path))


def draw_text(image, text, factor=1):
    width = round(40 * factor)
    padding = round(10 * factor)
    margin = round(5 * factor)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, width)
    text_size = draw.textsize(text, font)
    text_w = padding
    text_h = image.height - width - padding
    text_w = text_h = padding
    pos = [text_w, text_h, text_w + text_size[0], text_h + text_size[1]]
    # draw.rectangle(pos, fill='#000000')  # 用于填充
    draw.text((text_w, text_h), text, fill='#00ffff', font=font)  # blue
    return image


def cal_sdr(distance_list, threshold=SDR_THRESHOLD):
    ''' successfully detection rate
    '''
    if distance_list:
        ret = {}
        n = len(distance_list)
        for th in threshold:
            ret[th] = sum(dist <= th for dist in distance_list) / n
        return ret
    else:
        return {th:float('inf') for th in threshold}


def read_landmark_ratio_from_txt(path):
    dic = {}
    with open(path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            ratios = line.strip('\n ').split()
            exist, num, ratios = ratios[0], ratios[1], ratios[2:]
            assert int(num[3:]) == idx, 'num={}, idx={}'.format(num, idx)
            if exist=='True':
                dic[idx] = [float(r) for r in ratios]
    return dic


def get_candidate(path, existences, percentile, weighted, use_cache=True, min_area=300, max_area=5000):
    filename = os.path.basename(path)
    dest = '.cache_data'
    if not os.path.exists(dest):
        os.mkdir(dest)
    flag = '_weight' if weighted else '_argmax'
    pickle_path = os.path.join(dest, filename + flag + '.pkl')

    if os.path.exists(pickle_path) and use_cache:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    maps = np.load(path)

    # percentile along channel
    th_lst = [np.percentile(arr, percentile) for arr in maps]
    mask = [np.where(arr>=th, 1, 0).astype(np.uint8) for arr, th in zip(maps, th_lst)]

    '''connectivity, find candidates'''

    # 2d connected components
    # connected_components = {idx: cv2.connectedComponentsWithStats(mask[idx], connectivity=8) for idx in range(len(existences)) if existences[idx]}
    # 3d connected components
    connected_components = {idx: cc3d.connected_components(mask[idx], connectivity=26) for idx in range(len(existences)) if existences[idx]}
    candidate_ratios = {}
    for idx, label_map in connected_components.items():
        num_label = np.max(label_map)
        candidate_ratios[idx] = []
        for label in range(1, num_label+1):
            label_flag = label_map==label
            cur_labelmap = np.where(label_flag, maps[idx], 0)
            ratio = weight_coord([cur_labelmap])[0] if weighted else argmax_coord([cur_labelmap])[0]
            area = (label_map==label).sum()
            if min_area<=area<=max_area:
                candidate_ratios[idx].append(ratio)
        if candidate_ratios[idx] == []:
            # TODO weighted coord
            candidate_ratios[idx].append( argmax_coord([maps[idx]])[0] )
    with open(pickle_path, 'wb') as f:
        pickle.dump(candidate_ratios, f)

    return candidate_ratios

def extract_landmark_ratio_from_npy(path, existences, img_size, dataset, percentile=99.5, weighted=True, use_cache=True, min_area=300, max_area=5000, verbose=False):
    assert 0<=percentile<=100, 'percentile={}'.format(percentile)
    num_spine = sum(existences)
    max_candidate = 8
    if num_spine > 14:
        max_candidate = 3
    elif num_spine > 10:
        max_canidate = 4
    elif num_spine > 6:
        max_candidate = 5
    else:
        max_candidate = 8

    maps = np.load(path) # NxHxWxD

    candidate_ratios = get_candidate(path, existences, percentile, weighted, use_cache, min_area, max_area)

    # TODO
    # avg ratios for each channel
    avg_ratios = weight_coord(maps)
    max_ratios = argmax_coord(maps)
    gt_ratios = read_landmark_ratio_from_txt(path[:path.rfind('_')]+'_gt.txt')

    for idx, cands in candidate_ratios.items():
        if len(cands)>max_candidate:
            candidate_ratios[idx] = random.sample(cands, min(len(cands), max_candidate))
            # TODO
            candidate_ratios[idx].insert(0, avg_ratios[idx])

    # init candidate score
    candidate_scores = {idx: [[cand, 0] for cand in cand_lst] for idx, cand_lst in candidate_ratios.items()}

    # assign score according to dataset statistics
    with open('gt_dis_{}_analysis.yaml'.format(dataset)) as f:
        data = yaml.load(f.read())
        gt_z_dis = data['summary_z_dis']
        gt_dis = data['summary_dis']
    spine_idxs = sorted(list(candidate_scores.keys()))

    ranges = list(product(*[range(len(candidate_scores[idx])) for idx in spine_idxs]))
    if len(ranges)>100000:
        ranges = random.sample(ranges, 100000)
    idx_scores = {k: 0 for k in ranges}
    sign = sum([max_ratios[i][0]-max_ratios[-i-1][0] for i in range(len(spine_idxs)//2)])
    for cand_idxs in idx_scores:
        cur_z_dis_lst = []
        cur_dis_lst = []
        cur_score = 0
        for i in range(len(spine_idxs)-1):
            spine_idx = spine_idxs[i]
            cand_score1 = candidate_scores[spine_idx][cand_idxs[i]]
            cand_score2 = candidate_scores[spine_idx+1][cand_idxs[i+1]]
            cand1 = cand_score1[0]
            cand2 = cand_score2[0]
            cur_dis = radial(cand1, cand2, img_size)
            cur_dis_lst.append(cur_dis)
            cur_z_dis = radial(cand1, cand2, (img_size[0], 0, 0))
            cur_z_dis_lst.append(cur_z_dis) # TODO check, img_size order, which axis is z
            if cur_z_dis > gt_z_dis[spine_idx]['max']:
                cur_score -= 1
            elif cur_z_dis > gt_z_dis[spine_idx]['mean']:
                cur_score -= 0.5
            elif cur_z_dis < gt_z_dis[spine_idx]['min']:
                cur_score -= 1
            if sign*(cand1[0]-cand2[0])<0:
                cur_score -= 10


        idx_scores[cand_idxs] = cur_score
    best_cand_idxs = max(idx_scores, key=lambda idx: idx_scores[idx])
    ret = {spine_idx: candidate_scores[spine_idx][cand_idx][0] for spine_idx, cand_idx in zip(spine_idxs, best_cand_idxs)}

    if verbose:
        argmax_ratios = argmax_coord(maps)
        print('cand_idxs', best_cand_idxs)
        print('idx\t gt\t argmax \tcand')
        for idx, flag in enumerate(existences):
            if flag and idx in gt_ratios:
                print(idx, gt_ratios[idx], argmax_ratios[idx], ret[idx])
                print('candidates', candidate_ratios[idx])
    return ret


def evaluate(input_path, output_path, save_img=False, IS_DRAW_TEXT=False, percentile=99.5, use_npy=False):
    mkdir(output_path)
    data_name = parse_dataname_from_path(input_path)
    image_path_pre = PATH_DIC[data_name]
    image_path_suf = DATASET_IMAGE_SUFFIX_DIC[data_name]

    print('\n' + '-' * 20 + data_name + '-' * 20)
    print('input : ', input_path)
    print('output: ', output_path)
    print('image : ', image_path_pre)
    print('use_npy: {}, percentile: {}'.format(use_npy, percentile))

    out_label_path = os.path.join(output_path, 'labels')
    mkdir(out_label_path)
    if save_img:
        out_img_path = os.path.join(output_path, 'images')
        mkdir(out_img_path)
    physical_dis = {}
    undetected = {}

    gt_files = [gt_p for gt_p in os.listdir(input_path) if gt_p.endswith(GT_LABEL_SUFFIX)]

    pbar = tqdm(gt_files, ncols=80)
    data_num = len(gt_files)

    for i, gt_p in enumerate(pbar):
        pbar.set_description('{:03d}/{:03d}: {}'.format(i + 1, data_num, gt_p))
        name = gt_p[:gt_p.rfind(GT_LABEL_SUFFIX)]

        img_path = None
        if data_name == 'head':
            cur_name = name.split('_')[0]
            mid_dir = 'Test1Data' if int(cur_name)<=300 else 'Test2Data'
            img_path = os.path.join(image_path_pre, mid_dir, cur_name+image_path_suf)
        else:
            img_path = os.path.join(image_path_pre, name + image_path_suf)

        img = None
        img_size = None

        if data_name in {'cthead', 'spineweb3', 'verse'}:
            img = sitk.ReadImage(img_path)
            img_size = img.GetSize()[::-1]
        else:
            img = Image.open(img_path)

            if data_name in ['head', 'hand', 'jsrt']: # origin size, cal physical dis
                pass
            elif data_name in ['spineweb16']:
                img = img.resize((256, 512))
            else:
                raise Exception("Unkown dataset: {}".format(data_name))
            img_size =  img.size #(img.width, img.height)

        # read landmarks
        gt_ratios = read_landmark_ratio_from_txt(os.path.join(input_path, gt_p))
        pred_ratios = read_landmark_ratio_from_txt(os.path.join(input_path, name + '.txt'))

        if data_name in {'spineweb3', 'verse'} and use_npy:
            existences = [int(i in pred_ratios) for i in range(24)]
            pred_ratios = extract_landmark_ratio_from_npy(os.path.join(input_path, name+'_heatmap.npy'), existences, img_size, data_name, percentile=percentile, weighted=True, use_cache=False, min_area=300, max_area=5000)

            # ratios = argmax_coord(np.load(os.path.join(input_path, name + '_heatmap.npy')))
            # pred_ratios = {idx: ratios[idx] for idx in range(len(existences)) if existences[idx]}

        landmark_size = [1800, 1800] if 'head3' in  input_path else img_size

        gt_points = {num: tuple([round(ratio*X) for ratio, X in zip(ratios, landmark_size)]) for num, ratios in gt_ratios.items()}
        pred_points = {num: tuple([round(ratio*X) for ratio, X in zip(ratios, landmark_size)]) for num, ratios in pred_ratios.items()}

        physical_factor = 1
        if data_name in {'head'}:
            physical_factor = CEPH_PHYSICAL_FACTOR
        elif data_name == 'hand':
            physical_factor = WRIST_WIDTH / radial(gt_points[0], gt_points[4])
        elif data_name == 'jsrt':
            physical_factor = JSRT_PHYSICAL_FACTOR
        elif data_name in {'cthead', 'spineweb3','verse'}:
            physical_factor = img.GetSpacing()[::-1]
        else:
            raise Exception('Unknown dataset: {}'.format(data_name))

        detected_keys = sorted([key for key in gt_points if key in pred_points])
        pred_gt_pairs = [(pred_points[key], gt_points[key]) for key in detected_keys]

        undetected_keys = sorted([key for key in gt_points if key not in pred_points])
        undetected.update({name:{'path':img_path, 'landmark': undetected_keys}})

        if len(detected_keys) != 0:
            cur_dis_dic = {key: radial(p,q, physical_factor) for key, (p,q) in zip(detected_keys, pred_gt_pairs)}
            mre_value = sum(list(cur_dis_dic.values()))/len(cur_dis_dic)
            info_dic = {'path': os.path.abspath(img_path), 
                        'distance': cur_dis_dic, 
                        'MRE': mre_value
                       }
            physical_dis[name] = info_dic
        else:
            raise Exception('No landmark detected')


        if save_img:
            mre_str = '{:.4f}'.format(mre_value)

            # mark landmark:  green, red dots
            if data_name in {'cthead', 'spineweb3',  'verse'}:
                arr = np.zeros(img_size, dtype=np.uint8)
                axis_factor = tuple([1]*len(img_size))
                if data_name in {'spineweb3'}:
                    axis_factor = tuple([512/sz for sz in img_size]) # ellipsoid, NOTICE!!!
                factor_radial = partial(radial, axis_factor=axis_factor)


                marker_dis = 4
                if data_name =='spineweb3':
                    marker_dis=8
                for gt in gt_points.values():
                    color(arr, [gt], partial(factor_radial, gt), marker_dis, 2)
                for pred in pred_points.values():
                    color(arr, [pred], partial(factor_radial, pred), marker_dis, 1)
                for pred, gt in pred_gt_pairs:
                    for point in genPoints(pred, gt):
                        point = tuple([round(i) for i in point])
                        color(arr, [point], partial(factor_radial, point), marker_dis//2, 4)

                mask = sitk.GetImageFromArray(arr)
                mask.SetDirection(img.GetDirection())
                mask.SetOrigin(img.GetOrigin())
                mask.SetSpacing(img.GetSpacing())
                sitk.WriteImage(mask, out_img_path + '/' + name + '_' + mre_str + '_draw.nrrd')
            else:
                img.save(out_img_path + '/' + name + '_' + mre_str + '.png')

                marker_dis = 5
                # save resized image
                if data_name == 'spineweb16':
                    img_size = (256, 512)
                    marker_dis = 5
                else:
                    img_size = (512, 512)
                img2 = img.resize(img_size)
                img2.save(out_img_path + '/' + name + '_' + mre_str + '.png')

                arr = np.array(img2)
                if len(arr.shape) == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                arr = np.transpose(arr, (1, 0, 2))
                arr_2 = arr.copy()
                for key in detected_keys:
                    pred_r = pred_ratios[key]
                    gt_r = gt_ratios[key]
                    pred = [round(r*sz) for r, sz in zip(pred_r, img_size)]
                    gt = [round(r*sz) for r, sz in zip(gt_r, img_size)]

                    # arr
                    colorRGB(arr, [pred], partial(radial, pred), marker_dis, [255, 0, 0])
                    colorRGB(arr, [gt], partial(radial, gt), marker_dis, [0, 255, 0])

                    # arr with connected line between pred and gt points
                    num_of_points = max(int(2*radial(pred, gt)/marker_dis), 10)
                    for point in genPoints(pred, gt, n=num_of_points):
                        point = tuple([round(i) for i in point])
                        colorRGB(arr_2, [point], partial(radial, point), marker_dis//2, [0, 0, 255])
                    colorRGB(arr_2, [pred], partial(radial, pred), marker_dis, [255, 0, 0])
                    colorRGB(arr_2, [gt], partial(radial, gt), marker_dis, [0, 255, 0])

                arr = np.transpose(arr, (1, 0, 2))
                arr = Image.fromarray(arr)

                arr_2 = np.transpose(arr_2, (1, 0, 2))
                arr_2 = Image.fromarray(arr_2)

                # 512 mark
                arr.save(out_img_path + '/' + name + '_' + mre_str + '_mark.png')

                arr_2.save(out_img_path + '/' + name + '_' + mre_str + '_line_mark.png')

                # draw mre text
                if IS_DRAW_TEXT:
                    arr = draw_text(arr, mre_str)
                    arr.save(out_img_path + '/' + name + '_' + mre_str + '_draw.png')

                    arr_2 = draw_text(arr_2, mre_str)
                    arr_2.save(out_img_path + '/' + name + '_' + mre_str + '_line_draw.png')

    physical_dis = np2py(physical_dis)
    undetected = np2py(undetected)
    toYaml(os.path.join(output_path, 'undetected.yaml'), undetected)
    toYaml(os.path.join(output_path, 'distance.yaml'), physical_dis)
    physical_dis_list = sum([list(dic['distance'].values()) for dic in physical_dis.values()], [])
    undetected_list = sum([dic['landmark'] for dic in undetected.values()],[])
    return physical_dis_list, undetected_list


def evaluate_main(src, dest=None, save_img=True, draw_text=False, percentile=99.5, SDR=(2,4,10), use_npy=False):
    src = os.path.abspath(src)
    phy_dic = {}
    undetected_dic = {}
    if not dest:
        dest = os.path.join(os.path.dirname(src), 'eval_{}'.format(os.path.basename(src)))
    if use_npy:
        dest = dest + '_npy_percent_{}'.format(percentile)
    for dataset in os.listdir(src):
        inp = os.path.join(src, dataset)
        if os.path.isdir(inp):
            phy_dis_list, undetected_list = evaluate(inp, os.path.join(dest, dataset), save_img, draw_text, percentile, use_npy)
            phy_dic[dataset] = phy_dis_list
            undetected_dic[dataset] = undetected_list
    if 'test1_head' in phy_dic and 'test2_head' in phy_dic:
        phy_dic['test_head'] = [0.6*d1+0.4*d2 for d1, d2 in zip(phy_dic['test1_head'], phy_dic['test2_head'])] 
        undetected_dic['test_head'] = []


    for radial_th in [None, 20]:
        print('*' * 20 + 'th={}'.format(radial_th) + '*' * 20)
        summary = {}
        all_dis = []
        all_undetected = []
        for dataset in phy_dic:
            undet = undetected_dic[dataset]
            phy_dis = phy_dic[dataset]
            all_dis += phy_dis
            all_undetected += undet
            print('-' * 20 + dataset + '-' * 20)
            summary[dataset] = analysis(phy_dis.copy(), undet.copy(), radial_th, SDR)
        # print('-' * 20 + 'avg' + '-' * 20)
        summary['avg'] = analysis(all_dis.copy(), all_undetected.copy(), radial_th, SDR, verbose=False)
        savename = 'summary.yaml'
        if radial_th is not None:
            savename = 'summary_th{}.yaml'.format(radial_th)
        toYaml(os.path.join(dest, savename), summary)


def analysis(detected, undet, radial_threshold=None, SDR=(2, 2.5, 3, 4), verbose=True):
    summary = {}
    summary['ORI_NUM'] = len(detected) + len(undet)
    summary['DET_NUM'] = len(detected)
    if radial_threshold is None:
        li_th = detected
    else:
        li_th = [i for i in detected if i<radial_threshold]
    summary['TH_DET_NUM'] = len(li_th)
    summary['ID'] = summary['DET_NUM']/summary['ORI_NUM']
    summary['TH_ID'] = summary['TH_DET_NUM']/summary['ORI_NUM']

    mean1, std1, = np.mean(li_th), np.std(li_th)
    sdr1 = cal_sdr(li_th)
    summary['MRE'] = np2py(mean1)
    summary['STD'] = np2py(std1)
    summary['SDR'] = {k: np2py(v) for k, v in sdr1.items()}
    if verbose:
        print('MRE:', mean1, '+-', std1)
        print('ORI:', summary['ORI_NUM'])
        print('DET:', summary['DET_NUM'])
        print('TH_DET:', summary['TH_DET_NUM'])
        print('ID :', summary['ID'])
        print('TH_ID :', summary['TH_ID'])
        print('SDR:')
        for k in sorted(sdr1.keys()):
            if k in SDR:
                print('     {}: {:.4f}'.format(k, sdr1[k]))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # optional
    parser.add_argument("-s", "--save_img", action='store_true')
    parser.add_argument("-d", "--draw_text", action='store_true')
    parser.add_argument('--SDR', nargs='+', type=float, default=[2, 2.5, 3, 4])
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-C", "--config", type=str)
    parser.add_argument("-p", "--percentile", type=float, default=99.8)
    parser.add_argument("--use_npy", action='store_true')
    # required
    parser.add_argument("-i", "--input", type=str, required=True)

    args = parser.parse_args()

    evaluate_main(args.input, args.output, args.save_img, args.draw_text, args.percentile, args.SDR, args.use_npy)
