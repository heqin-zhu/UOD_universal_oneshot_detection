import argparse
import os
import yaml
import logging

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from PIL import Image
import numpy as np

from lib.networks import UVGG
from lib.datasets import Head_SSL, Hand_SSL, JSRT_SSL
from lib.utils import get_config, update_config, toYaml, MixIter

from test_ssl import Tester

import warnings
warnings.filterwarnings("ignore")

# torch.autograd.set_detect_anomaly(True)

def cos_visual(tensor):
    tensor = torch.clamp(tensor, 0, 10)
    tensor = tensor * 25.5
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def gray_to_PIL(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def ce_loss(cos_map, gt_y, gt_x, nearby=None):
    b, h, w = cos_map.shape
    total_loss = list()
    for id in range(b):
        cos_map[id] = cos_map[id].exp()
        gt_value = cos_map[id, gt_y[id], gt_x[id]].clone()
        if nearby is not None:
            min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
            min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
            chosen_patch = cos_map[id, min_y:max_y, min_x:max_x]
        else:
            chosen_patch = cos_map[id]
        id_loss = - torch.log(gt_value / chosen_patch.sum())
        total_loss.append(id_loss)
    return torch.stack(total_loss).mean()


def match_inner_product(feature, template):
    feature = feature.permute(0, 2, 3, 1)
    template = template.unsqueeze(1).unsqueeze(1)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    cos_similarity = torch.clamp(cos_similarity, 0., 1.)
    assert torch.max(cos_similarity) <= 1.0, f"Maximum Error, Got max={torch.max(cos_similarity)}"
    assert torch.min(cos_similarity) >= 0.0, f"Maximum Error, Got max={torch.min(cos_similarity)}"
    return cos_similarity 


cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
def match_inner_product2(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    return cos_sim


def get_dataset(config, dataname):
    if dataname == 'head':
        return Head_SSL(config['dataset']['head']['pth'], 'Train', patch_size=32*8, retfunc=1, use_prob=False)
    elif dataname == 'hand':
        return Hand_SSL(config['dataset']['hand']['pth'], 'Train', patch_size=32*8, retfunc=1, use_prob=False)
    elif dataname == 'jsrt':
        return JSRT_SSL(config['dataset']['jsrt']['pth'], 'Train', patch_size=32*8, retfunc=1, use_prob=False)
    else:
        raise Exception('Unknown dataset: {}'.format(datanme))


def get_model(model, **kargs):
    net_name = model.lower()
    if net_name == 'uvgg':
        return UVGG(**kargs)
    else:
        raise Exception('Unkown model: {}'.format(model))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train landmark detection network")
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--phase", choices=['train','test'], required=True)

    parser.add_argument("--config", default="config_ssl.yaml")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--oneshot_id_list", nargs='+', type=str, help='should be consistent with data_list', required=True)
    parser.add_argument("--data_list", nargs='+', type=str, help='dataset names', required=True)
    parser.add_argument("--model", type=str, default='unet_pretrained')

    parser.add_argument('-b', "--batch_size", type=int, default=8)
    parser.add_argument('-x', "--mix_step", type=int, default=0)
    parser.add_argument('-e', "--epoch", type=int, default=500)
    parser.add_argument('-g', "--gpu", type=str, default='0')
    args = parser.parse_args()

    config = get_config(args.config)
    update_config(config, args)

    assert len(args.oneshot_id_list) == len(args.data_list)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # should be in front of  "import torch"
    run_dir = os.path.join(args.run_dir, args.run_name)
    config['base']['run_dir'] = run_dir
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    toYaml("{rd}/config_{ph}.yaml".format(rd=run_dir, ph=args.phase), config)

    logging.basicConfig(filename=os.path.join(run_dir, 'log.log'), level=logging.ERROR)
    logger = logging


    # Tester
    tester = Tester(args.data_list, logger, config)
    train_loaders = []

    for dataname in args.data_list:
        dataset = get_dataset(config, dataname)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=4)
        train_loaders.append(dataloader)

    # mix step
    if args.mix_step>0:
        train_loaders = [MixIter(train_loaders, args.mix_step)]

    net_paras = config[args.model.lower()]
    if args.model.lower() in ['uvgg', 'uvgg_ori']:
        net_paras['domain_num'] = len(args.data_list)
    net = get_model(args.model, **net_paras)

    if args.model.lower() == 'unet_pretrained':
        net_paras['non_local'] = False

    net_patch = get_model(args.model, **net_paras)


    start_epoch = 0
    if args.checkpoint_path is not None:
        check_name = os.path.basename(args.checkpoint_path)
        start_epoch = int(check_name[:-4].split('_')[-1])
        logger.info(f'Load epoch={start_epoch} CKPT {args.checkpoint_path}')
        net.load_state_dict(torch.load(args.checkpoint_path))
        path2 = os.path.join(os.path.dirname(args.checkpoint_path), check_name.replace('model', 'model_patch'))
        assert os.path.exists(path2)
        net_patch.load_state_dict(torch.load(path2))

    net = net.cuda()
    net_patch = net_patch.cuda()

    if args.phase.lower()=='test':
        assert args.checkpoint_path is not None and os.path.exists(args.checkpoint_path)
        for oneshot_id, data_name in zip(args.oneshot_id_list, args.data_list):
            tester.test(net, data_name=data_name, epoch=start_epoch, dump_label=True, oneshot_id=oneshot_id)
        exit()

    optimizer = optim.Adam(params=net.parameters(), lr=config['training']['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = StepLR(optimizer, config['training']['decay_step'], gamma=config['training']['decay_gamma'])

    optimizer_patch = optim.Adam(params=net_patch.parameters(), lr=config['training']['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler_patch = StepLR(optimizer_patch, config['training']['decay_step'], gamma=config['training']['decay_gamma'])

    # loss
    loss_logic_fn = torch.nn.CrossEntropyLoss()
    mse_fn = torch.nn.MSELoss()
    
    # Best MRE record
    best_mre = 100.0
    best_epoch = -1
    best_net_path = best_net_patch_path = ''

    for epoch in range(start_epoch, config['training']['epoch']):
        net.train()
        net_patch.train()
        logic_loss_list = list()
        for domain_idx, dataloader in enumerate(train_loaders):
            for index, items in enumerate(dataloader):
                if args.mix_step>0:
                    items, domain_idx = items
                raw_img, crop_imgs, chosen_y, chosen_x, raw_y, raw_x = items
                with torch.autograd.set_detect_anomaly(False):
                    raw_img = raw_img.cuda()
                    crop_imgs = crop_imgs.cuda()

                    _, raw_fea_list = net(raw_img, get_features=True, domain_idx=domain_idx)
                    _, crop_fea_list = net_patch(crop_imgs, get_features=True, domain_idx=domain_idx)

                    gt_y, gt_x = raw_y // (2 ** 5), raw_x // (2 ** 5)
                    tmpl_y, tmpl_x = chosen_y // (2 ** 5), chosen_x // (2 ** 5)
                    
                    tmpl_feature = torch.stack([crop_fea_list[0][[id], :, tmpl_y[id], tmpl_x[id]] for id in range(gt_y.shape[0])]).squeeze()
                    ret_inner_5 = match_inner_product(raw_fea_list[0], tmpl_feature)  # shape [8,12,12]

                    loss_5 = ce_loss(ret_inner_5, gt_y, gt_x)

                    gt_y, gt_x = raw_y // (2 ** 4), raw_x // (2 ** 4)
                    tmpl_y, tmpl_x = chosen_y // (2 ** 4), chosen_x // (2 ** 4)
                    tmpl_feature = torch.stack([crop_fea_list[1][[id], :, tmpl_y[id], tmpl_x[id]] \
                                                for id in range(gt_y.shape[0])]).squeeze()
                    ret_inner_4 = match_inner_product(raw_fea_list[1], tmpl_feature)
                    loss_4 = ce_loss(ret_inner_4, gt_y, gt_x, nearby=config['training']['nearby'])

                    gt_y, gt_x = raw_y // (2 ** 3), raw_x // (2 ** 3)
                    tmpl_y, tmpl_x = chosen_y // (2 ** 3), chosen_x // (2 ** 3)
                    tmpl_feature = torch.stack([crop_fea_list[2][[id], :, tmpl_y[id], tmpl_x[id]] \
                                                for id in range(gt_y.shape[0])]).squeeze()
                    ret_inner_3 = match_inner_product(raw_fea_list[2], tmpl_feature)
                    loss_3 = ce_loss(ret_inner_3, gt_y, gt_x, nearby=config['training']['nearby'])

                    gt_y, gt_x = raw_y // (2 ** 2), raw_x // (2 ** 2)
                    tmpl_y, tmpl_x = chosen_y // (2 ** 2), chosen_x // (2 ** 2)
                    tmpl_feature = torch.stack([crop_fea_list[3][[id], :, tmpl_y[id], tmpl_x[id]] \
                                                for id in range(gt_y.shape[0])]).squeeze()
                    ret_inner_2 = match_inner_product(raw_fea_list[3], tmpl_feature)
                    loss_2 = ce_loss(ret_inner_2, gt_y, gt_x, nearby=config['training']['nearby'])

                    gt_y, gt_x = raw_y // (2 ** 1), raw_x // (2 ** 1)
                    tmpl_y, tmpl_x = chosen_y // (2 ** 1), chosen_x // (2 ** 1)
                    tmpl_feature = torch.stack([crop_fea_list[4][[id], :, tmpl_y[id], tmpl_x[id]] \
                                                for id in range(gt_y.shape[0])]).squeeze()
                    ret_inner_1 = match_inner_product(raw_fea_list[4], tmpl_feature)
                    loss_1 = ce_loss(ret_inner_1, gt_y, gt_x, nearby=config['training']['nearby'])

                    loss = loss_5 + loss_4 + loss_3 + loss_2 + loss_1

                    optimizer.zero_grad()
                    optimizer_patch.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer_patch.step()

                    logic_loss_list.append(np.array([loss_5.cpu().item(), loss_4.cpu().item(), loss_3.cpu().item(), loss_2.cpu().item(), loss_1.cpu().item()]))

        losses = np.stack(logic_loss_list).transpose()
        logger.info("Epoch {} Training logic loss 5 {:.3f} 4 {:.3f} 3 {:.3f} 2 {:.3f} 1 {:.3f}". \
                    format(epoch, losses[0].mean(), losses[1].mean(), losses[2].mean(), \
                           losses[3].mean(), losses[4].mean()))

        scheduler.step()
        scheduler_patch.step()

        if (epoch) % config['training']['save_seq'] == 0:
            net.eval()
            net_patch.eval()
            mre_lst = []
            for oneshot_id, data_name in zip(args.oneshot_id_list, args.data_list):
                metric_data = tester.test(net, data_name=data_name, epoch=epoch, dump_label=False, oneshot_id=oneshot_id)
                mre_lst.append(metric_data['MRE'])
            mre = sum(mre_lst)/len(mre_lst)

            if mre < best_mre:
                best_mre = mre
                best_epoch = epoch
                best_net_path = run_dir + "/best_model_epoch_{}.pth".format(epoch)
                torch.save(net.state_dict(), best_net_path)
                best_net_patch_path = run_dir + "/best_model_patch_epoch_{}.pth".format(epoch)
                torch.save(net_patch.state_dict(), best_net_patch_path)
            logger.info(f"tag:{args.run_name} ***********  Best MRE:{best_mre} in Epoch {best_epoch} || Epoch:{epoch}:{mre} ***********")
            logger.info(run_dir + "/model_epoch_{}.pth".format(epoch))
            logger.info(f"tag:{args.run_name} ***  Best MRE:{best_mre} in Epoch {best_epoch} || Epoch:{epoch}:{mre} *** " + run_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), run_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net_patch.state_dict(), run_dir + "/model_patch_epoch_{}.pth".format(epoch))
            config['training']['last_epoch'] = epoch


    ckpt = torch.load(best_net_path)
    net.load_state_dict(ckpt)
    for oneshot_id, data_name in zip(args.oneshot_id_list, args.data_list):
        tester.test(net, data_name=data_name, epoch=best_epoch, dump_label=True, oneshot_id=oneshot_id)
