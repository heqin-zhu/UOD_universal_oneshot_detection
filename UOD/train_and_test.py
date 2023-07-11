import os
import datetime
import random
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.datasets import get_dataset
from lib.networks import get_net
from lib.utils import *
from lib.utils import MixIter
from evaluate import evaluate_main

torch.autograd.set_detect_anomaly(True)


def get_dataset_ratios(data_name, size=(576, 576)):
    ratios = (1, 1)
    if data_name=='head':
        ratios = (0.1*2400/size[0], 0.1*2400/size[1])
    elif data_name=='hand':
        ratios = (1, 1)
    elif data_name=='jsrt':
        ratios = (0.175*2048/size[0], 0.175*2048/size[1])
    else:
        raise Exception('Unknown data_name "{}"'.format(data_name))
    return ratios


def save_landmark(dest, img_name, pred_landmarks, gt_landmarks, size, landmark_idx=None, pred_heatmap=None, gt_heatmap=None, to_json=False):
    if '.' in img_name:
        img_name = img_name.split('.')[0]
    for suf, points in {'':pred_landmarks, '_gt':gt_landmarks}.items():
        point_ratios = [[p/s for p, s in zip(pt, size)] for pt in points]
        save_path = os.path.join(dest, img_name + suf)
        if landmark_idx is not None:
            save_path = os.path.join(dest, img_name + '_{:02d}'.format(landmark_idx) + suf )
        data_str = None
        if to_json:
            save_path = save_path + '.json'
            data_dic = {num: list(np2py(coords)) for num, coords in enumerate(point_ratios)}
            data_str = json.dumps(data_dic)
        else:
            save_path = save_path + '.txt'
            lines = []
            for num, point_ratio in enumerate(point_ratios):
                flag = True
                coord_str = ' '.join(['{:.6f}'.format(r) for r in point_ratio])
                line = '{flag} num{num:02d} {s}'.format(flag=flag, num=num, s=coord_str)
                lines.append(line)
            data_str = '\n'.join(lines)
        with open(save_path, 'w') as f:
            f.write(data_str)
    if pred_heatmap is not None:
        save_path = os.path.join(dest, img_name + '_heatmap' + '.npy')
        np.save(save_path, pred_heatmap)
    if gt_heatmap is not None:
        save_path = os.path.join(dest, img_name + '_heatmap_gt' + '.npy')
        np.save(save_path, gt_heatmap)


def get_landmark_from_output(pred_heatmap):
    pred_landmarks = get_landmark_by_component(pred_heatmap)
    return pred_landmarks


def get_bbox_from_landmark(landmarks, ratios=1, radius=64, size=(2400, 2400)):
    ''' landmark: C x imgsize, ndarray '''
    landmarks = (landmarks*ratios).astype(np.int).tolist()
    bbox_lst = []
    for x, y in landmarks:
        x0 = x1 = y0 = y1 = None
        if x<radius:
            x0 = 0
            x1 = 2*radius
        elif x+radius>size[0]:
            x1 = size[0]
            x0 = size[0] - 2*radius
        else:
            x0 = x-radius
            x1 = x+radius
        if y<radius:
            y0 = 0
            y1 = 2*radius
        elif y+radius>size[0]:
            y1 = size[0]
            y0 = size[0] - 2*radius
        else:
            y0 = y-radius
            y1 = y+radius

        bbox_lst.append((x0, x1, y0, y1))
    return bbox_lst


class Runner(object):
    def __init__(self, args):
        self.args = args
        self.phase = self.args.phase

    def run(self):
        self.get_opts()
        self.initialize_loaders()

        self.start_epoch = 0
        model_stg1, criterion, optimizer, scheduler = self.get_model(self.opts.model, self.opts.checkpoint)

        model_stg2 = None
        criterion2 = optimizer2 = scheduler2 = None
        if self.opts.model_stg2:
            model_stg2, criterion2, optimizer2, scheduler2 = self.get_model(self.opts.model_stg2, self.opts.checkpoint_stg2)

        assert self.phase in ['train', 'test', 'train_stg2'], 'Unknown phase: {}'.format(self.phase)

        best_path = self.opts.checkpoint
        best_path2 = self.opts.checkpoint_stg2
        if self.phase == 'train':
            # train stg1
            best_path = self.train_stg1(model_stg1, criterion, optimizer, scheduler)

            # freeze model_stg1
            for (name, param) in model_stg1.named_parameters():
                param.requires_grad = False
            self.phase = 'train_stg2'

        if self.phase == 'train_stg2':
            # train_stg2
            if model_stg2:
                best_path2 = self.train_stg2(model_stg1, model_stg2, criterion2, optimizer2, scheduler2)

        # test after training
        self.phase = 'test'
        if best_path!=self.opts.checkpoint:
            self.load_checkpoint(model_stg1, best_path)

        if model_stg2 and best_path2 != self.opts.checkpoint_stg2:
            self.load_checkpoint(model_stg2, best_path2)

        self.test(model_stg1, model_stg2, self.start_epoch)

    def get_opts(self):
        self.opts = get_config(self.args.config)
        update_config(self.opts, self.args)
        self.run_name = self.opts.run_name if self.opts.run_name else self.opts.model
        self.run_name = self.run_name.strip(os.sep)
        self.run_dir = os.path.join(self.opts.run_dir, self.run_name)
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        self.result_dir = os.path.join(self.run_dir, 'results')
        print('>> output: {}'.format(self.run_dir))

        for d in [self.run_dir, self.checkpoint_dir, self.result_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        toYaml("{rd}/config_{ph}.yaml".format(rd=self.run_dir, ph=self.phase), self.opts)

        origin_config = '{run_dir}/config_origin.yaml'.format(run_dir=self.run_dir)
        if not os.path.exists(origin_config):
            shutil.copy(self.args.config, origin_config)

        self.setup_seed(self.opts.seed)

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_model(self, modelname, checkpoint_path, **kargs):
        net_params = {}

        if modelname in self.opts:
            for k, v in self.opts[modelname].items():
                net_params[k] = v


        if 'in_channels' not in net_params:
            net_params['in_channels'] = 1
        net_params['out_channel_list'] = []
        for data_name in  self.opts.dataset.data_list:
            net_params['out_channel_list'].append(self.opts.dataset[data_name]['paras']['num_landmark'])


        for k,v in kargs.items():
           net_params[k] = v

        if modelname.upper().startswith('DATR'):
            for k, v in self.opts['swin_'+self.opts.swin_type].items():
                net_params[k] = v

        print('building model...')
        print(modelname, net_params)

        model = None
        if modelname == 'gu2net':
            model = get_net('gln')(get_net('u2net'), net_params, {}) 
        else:
            model = get_net(modelname)(**net_params)
        model = model.cuda()

        lr = self.opts.learning.lr
        opt_grad_params = [ {'params':filter(lambda p: p.requires_grad, model.parameters()), 'lr':lr} ]

        self.load_checkpoint(model, checkpoint_path)

        # loss optim, sche
        criterion = torch.nn.SmoothL1Loss(beta=1.5).cuda()

        optimizer = torch.optim.Adam(opt_grad_params)

        scheduler_name = self.opts.learning.scheduler
        scheduler = None
        if scheduler_name == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ** self.opts.learning[scheduler_name])
        elif scheduler_name == 'clr':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **self.opts.learning[scheduler_name])
        else:
            raise NotImplementedError
        return model, criterion, optimizer, scheduler

    def initialize_loaders(self):
        def get_one_loader(phase, data_name):
            d_opts = self.opts.dataset[data_name]

            img_dir = gt_dir = None
            if data_name == 'hand':
                img_dir = os.path.join(d_opts['prefix'], 'jpg')
                gt_dir = d_opts['prefix']
            elif data_name == 'jsrt':
                img_dir = os.path.join(d_opts['prefix'], 'imgs')
                gt_dir = os.path.join(d_opts['prefix'], 'labels')
            else: # head
                phase_dir = {
                             'train': 'TrainingData',
                             'validate': 'validate',
                             'test1': 'Test1Data',
                             'test2': 'Test2Data',
                             'gt': 'gt',
                            }
                img_dir = os.path.join(d_opts['prefix'], phase_dir[phase])
                gt_dir = os.path.join(d_opts['prefix'], phase_dir['gt'])

            shuffle = False
            drop_last = False
            cur_batch_size = 1
            num_workers = 1
            if phase == 'train':
                shuffle = True
                drop_last = True
                cur_batch_size = self.opts.batch_size
                num_workers = 2

            test_aug_dic = {}
            if phase!='train' and self.opts.dataset.blur!=0:
                print('[{}] blur = {}'.format(phase, blur))
                test_aug_dic['blur'] = self.opts.dataset.blur

            dataset = get_dataset(data_name)(img_dir=img_dir, gt_dir=gt_dir,  **d_opts['paras'], phase=phase, test_aug_dic=test_aug_dic)
            loader = DataLoader(dataset=dataset, batch_size=cur_batch_size, shuffle=shuffle, num_workers=num_workers)
            return loader

        self.data_list = self.opts.dataset.data_list
        self.raw_name_dic = {k:k for k in self.data_list}

        # test loader
        self.test_loader_dic = {}
        for data_name in self.data_list:
            if 'head' == data_name:
                self.test_loader_dic['test1_'+data_name] = get_one_loader('test1', data_name)
                self.test_loader_dic['test2_'+data_name] = get_one_loader('test2', data_name)
                self.raw_name_dic['test1_'+data_name] = data_name
                self.raw_name_dic['test2_'+data_name] = data_name
            else:
                self.test_loader_dic['test_'+data_name] = get_one_loader('test', data_name)
                self.raw_name_dic['test_'+data_name] = data_name

        if self.phase == 'train':
            # initialize train loader and validate loader

            # train loader
            self.train_name_list = self.data_list[:]
            self.train_loader_list = [get_one_loader('train', data_name) for data_name in self.data_list]
            step = int(self.opts.mix_step)
            if step > 0:
                self.train_name_list = ['mix']
                self.train_loader_list = [MixIter(self.train_loader_list, step)]

            # validate loader
            # self.validate_loader_dic = {data_name:get_one_loader('validate', data_name) for data_name in self.data_list}
            self.validate_loader_dic = {k:v for k, v in self.test_loader_dic.items()}


    def get_train_loaders(self, epoch):
        def get_one_pseudo_train_loader(phase, data_name):
            d_opts = self.opts.dataset[data_name]
            d_opts['paras']['train_with_pseudo'] = True
            if epoch > self.opts.epochs/2:
                dir_name = 'mix' if int(self.opts.mix_step)>0 else data_name
                d_opts['paras']['pseudo_path'] = os.path.join(self.run_dir, 'generated_labels', 'ep{:03d}'.format(epoch-1), dir_name)

            img_dir = gt_dir = None
            if data_name == 'hand':
                img_dir = os.path.join(d_opts['prefix'], 'jpg')
                gt_dir = d_opts['prefix']
            elif data_name == 'jsrt':
                img_dir = os.path.join(d_opts['prefix'], 'imgs')
                gt_dir = os.path.join(d_opts['prefix'], 'labels')
            else: # head
                phase_dir = {
                             'train': 'TrainingData',
                             'validate': 'validate',
                             'test1': 'Test1Data',
                             'test2': 'Test2Data',
                             'gt': 'gt',
                            }
                img_dir = os.path.join(d_opts['prefix'], phase_dir[phase])
                gt_dir = os.path.join(d_opts['prefix'], phase_dir['gt'])

            shuffle = True
            drop_last = True
            cur_batch_size = self.opts.batch_size
            num_workers = 2

            dataset = get_dataset(data_name)(img_dir=img_dir, gt_dir=gt_dir, phase=phase,  **d_opts['paras'])
            loader = DataLoader(dataset=dataset, batch_size=cur_batch_size, shuffle=shuffle, num_workers=num_workers)
            return loader


        if self.args.train_with_pseudo:
            loaders = [get_one_pseudo_train_loader('train', data_name) for data_name in self.data_list]
            step = int(self.opts.mix_step)
            if step > 0:
                loaders = [MixIter(loaders, step)]
            return loaders
        else:
            return self.train_loader_list


    def load_checkpoint(self, model, checkpoint_path, flag=''):
        if os.path.isdir(checkpoint_path):
            paths = [p for p in os.listdir(checkpoint_path) if flag in p]
            if paths:
                checkpoint_path = os.path.join(checkpoint_path, sorted(paths)[-1])

        if os.path.isfile(checkpoint_path):
            epoch_str = [i for i in os.path.basename(checkpoint_path).split('_') if 'epoch' in i][0]
            self.start_epoch = int(epoch_str[len('epoch'):])
            print('[{}] loading checkpoint ep{:03d}: {}'.format(self.phase, self.start_epoch, checkpoint_path))
            state_dict = torch.load(checkpoint_path)

            model.load_state_dict(state_dict, strict=True)
        elif checkpoint_path == '':
            print('No checkpoint')
        else:
            print('[{}] Warning: invalid checkpoint path "{}"'.format(self.phase, checkpoint_path))

    def train_stg1(self, model, criterion, optimizer, scheduler):
        best_path = ''
        val_err = float('inf')
        best_error = float('inf')

        learn_opts = self.opts.learning

        for epoch in range(self.start_epoch, self.opts.epochs+1):
            epoch = epoch + 1
            model.train()

            loss_temp = 0
            loss_count = 0
            train_loaders = self.get_train_loaders(epoch)
            for data_idx, (data_name, loader) in enumerate(zip(self.train_name_list, train_loaders)):
                generated_path = os.path.join(self.run_dir, 'generated_labels', 'ep{:03d}'.format(epoch), data_name)
                if not os.path.exists(generated_path):
                    os.makedirs(generated_path)

                for i, data_dic in enumerate(loader):
                    if isinstance(data_dic, tuple):
                        data_dic, data_idx = data_dic
                        data_name = self.data_list[data_idx]
                    for k in data_dic:
                        if k != 'name':
                            data_dic[k] = data_dic[k].cuda()
                    out_dic = model.forward(data_dic['image'], domain_idx=data_idx)

                    loss = criterion(out_dic['heatmap'], data_dic['heatmap'])
                    if self.opts.learning.learn_similarity:
                        cur_input = data_dic['image']
                        if cur_input.shape[1]!=3:
                            cur_input = torch.cat([cur_input, cur_input, cur_input], dim=1)
                        loss_sim = self.similarity_learner(cur_input)
                        loss += self.opts.learning.weight_similarity * loss_sim
                    loss_temp += loss.item()
                    loss_count +=1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step() # TODO

                    # saving generated labels of trainset
                    gt_landmark_list = data_dic['landmark'].cpu().numpy() # batch_size  = or != 1
                    pred_heatmap_list = out_dic['heatmap'].detach().cpu().numpy()
                    for pred_heatmap, gt_landmarks, img_name in zip(pred_heatmap_list, gt_landmark_list, data_dic['name']):
                        pred_landmarks = get_landmark_from_output(pred_heatmap)
                        assert gt_landmarks.shape == pred_landmarks.shape
                        # size == (1, 1) for saving direct coordinates
                        save_landmark(generated_path, img_name, pred_landmarks, gt_landmarks, size=(1,1), to_json=True)

                # scheduler.step()

            if loss_count == 0 :
                raise Exception('Empty loader: {}'.format('_'.join(self.data_list)))
            epoch_loss = loss_temp / loss_count
            if epoch % self.opts.val_freq==0 or epoch == self.opts.epochs or epoch % 20 == 0:
                model.eval()
                val_err_dic = {}
                for raw_name, loader in self.validate_loader_dic.items():
                    data_name = self.raw_name_dic[raw_name]
                    data_idx = self.data_list.index(data_name)
                    dis_lst = self.val_stg1(model, loader, data_idx)
                    val_err_dic[raw_name] = np.mean(dis_lst)
                val_info = '-'.join(['{}{:.4f}'.format(k,v) for k,v in val_err_dic.items()])
                file_name = 'epoch{:03d}_bestval{:.6f}_val-{}_train{:.6f}.pth'.format(epoch, best_error, val_info , epoch_loss)
                val_err = sum(val_err_dic.values())
                if val_err < best_error:
                    best_error = val_err
                    file_name = 'best_'+file_name

                    save_file_path = os.path.join(self.checkpoint_dir,file_name)
                    torch.save(model.state_dict(), save_file_path)
                    if self.opts.learning.learn_similarity:
                        torch.save(self.similarity_learner.state_dict(), os.path.join(self.checkpoint_dir, 'sim_'+file_name))
                    best_path = save_file_path
                elif epoch % 20 ==0:
                    save_file_path = os.path.join(self.checkpoint_dir, file_name)
                    torch.save(model.state_dict(), save_file_path)
                    if self.opts.learning.learn_similarity:
                        torch.save(self.similarity_learner.state_dict(), os.path.join(self.checkpoint_dir, 'sim_'+file_name))


            tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('[{}][{}][train stg1] bestval={:.6f}, lastval={:.6f}, train={:.6f}'.format(tm, epoch, best_error, val_err, epoch_loss))
        return best_path

    def val_stg1(self, model, val_loader, data_idx, save_dir=None):
        map_types = ['heatmap']

        arr_dest = os.path.join(self.result_dir, 'arr')
        for map_type in map_types:
            dest = os.path.join(arr_dest, map_type)
            if not os.path.exists(dest):
                os.makedirs(dest)

        dis_lst = []

        data_name = self.data_list[data_idx]
        size = self.opts.dataset[data_name]['paras']['img_size']
        ratios = get_dataset_ratios(data_name, size)

        for index, data_dic in enumerate(val_loader):
            landmark_idx = data_dic['landmark_idx'].squeeze(dim=0).item() if 'landmark_idx' in data_dic else None
            img_names = [name.split('.')[0] for name in data_dic['name']]

            # gt landmark
            gt_landmarks = data_dic['landmark'].squeeze(dim=0).cpu().numpy() # batch_size==1, squeeze
            
            # pred landmark
            for k in data_dic:
                if k != 'name':
                    data_dic[k] = data_dic[k].cuda()
            out_dic = model.forward(data_dic['image'], domain_idx=data_idx)

            # save arr
            for key in map_types:
                if key in out_dic:
                    for i, img_name in enumerate(img_names):
                        arr = out_dic[key][i].detach().cpu().numpy()
                        if key == 'actionmap':
                            pred_actionmap = np.stack([arr[4*i:4*(i+1)].argmax(axis=0) for i in range(19)], axis=0)
                            agg_map = np.stack([aggregate_actionmap(act_map).sum(axis=0) for act_map in pred_actionmap], axis=0)
                            out_dic['agg_map'] = agg_map
                            npy_name = img_names[i]+'.npy' if landmark_idx is None else '{}_{:02d}.npy'.format(img_names[i], landmark_idx)
                            np.save(os.path.join(arr_dest, key, 'agg_'+npy_name), arr)
                        npy_name = img_names[i]+'.npy' if landmark_idx is None else '{}_{:02d}.npy'.format(img_names[i], landmark_idx)
                        np.save(os.path.join(arr_dest, key, npy_name), arr)

            pred_heatmap = out_dic['heatmap'][0].detach().cpu().numpy()
            gt_heatmap = data_dic['heatmap'][0].detach().cpu().numpy()
            
            pred_landmarks = get_landmark_from_output(out_dic['heatmap'].detach().cpu().numpy()[0]) # batch_size = 1

            # cal dis (pred, gt) for validation
            assert gt_landmarks.shape == pred_landmarks.shape
            for gt, pred in zip(gt_landmarks, pred_landmarks):
                if data_name == 'hand':
                    wrist_width = sum((p-q)**2 for p, q in zip(gt_landmarks[0], gt_landmarks[4]))**0.5
                    ratios = tuple([50.0/wrist_width]* len(gt))
                dis = sum(((p-q)*r)**2  for p, q, r in zip(pred, gt, ratios))**0.5
                dis_lst.append(dis)

            if save_dir:
                assert len(img_names)==1
                save_landmark(save_dir, img_names[0], pred_landmarks, gt_landmarks, size, landmark_idx, pred_heatmap, gt_heatmap)
        return dis_lst


    def test(self, model_stg1, model_stg2=None, start_epoch=0):
        model_stg1.eval()
        for i, (raw_name, loader) in enumerate(self.test_loader_dic.items()):
            data_name = self.raw_name_dic[raw_name]
            data_idx = self.data_list.index(data_name)

            dest = os.path.join(self.result_dir, 'ep{:03d}'.format(start_epoch), raw_name)
            if not os.path.exists(dest):
                os.makedirs(dest)
            self.val_stg1(model_stg1, loader, data_idx, save_dir=dest)
            if model_stg2:
                self.val_stg2(model_stg1, model_stg2, loader, save_dir=dest)
        evaluate_main(os.path.join(self.result_dir, 'ep{:03d}'.format(start_epoch)))
