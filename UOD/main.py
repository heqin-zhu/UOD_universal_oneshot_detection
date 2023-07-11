import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument('-p', "--phase", choices=['train', 'test', 'train_stg2'], required=True)
    parser.add_argument('-r', "--run_name", type=str, required=True)
    parser.add_argument('-d', "--run_dir", type=str, required=True)
    parser.add_argument('-m', "--model", type=str, required=True)

    # optional
    parser.add_argument('-C', "--config", default='config.yaml', type=str)

    parser.add_argument("--model_stg2")
    parser.add_argument('-c', "--checkpoint", type=str)
    parser.add_argument("--checkpoint_stg2", type=str)

    parser.add_argument("--data_list", nargs='+', type=str, help='dataset names')
    parser.add_argument('-e', "--epochs", type=int)
    parser.add_argument('-g', "--gpu", type=str, default='0')
    parser.add_argument('-b', "--batch_size", type=int)
    parser.add_argument('-s', "--sigma", type=float)
    parser.add_argument('-a', "--alpha", type=float)

    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument("--encoder", type=str, default='resnet', help='encoder for model_stg2')
    parser.add_argument("--highres_factor", type=int, default=1)
    parser.add_argument("--crop_radius", type=int, default=64)
    parser.add_argument("--crop_img", action='store_true')
    parser.add_argument("--use_actionmap", action='store_true')
    parser.add_argument("--use_offset", action='store_true')
    parser.add_argument("--use_layerscale", action='store_true')
    parser.add_argument("--freeze_encoder_path", type=str, default='', help='stg1 resnet encoder para path')
    parser.add_argument("--blur", type=float, default=0)
    parser.add_argument("--learn_similarity", action='store_true')
    parser.add_argument("--use_momentum", action='store_true')

    # pseudo label
    # parser.add_argument("--pseudo_path", type=str, default='')
    parser.add_argument("--train_with_pseudo", action='store_true')

    # few-shot
    parser.add_argument("--few_shot", type=int)

    # universal
    parser.add_argument('-x', "--mix_step", type=int)

    # swin,  DATR
    parser.add_argument("--swin_type", default='base')
    parser.add_argument("--down_scale", default=4, type=int)
    parser.add_argument("--fusion_depth", default=1, type=int)
    parser.add_argument("--attn_type", default='query', type=str, choices=['origin', 'query', 'query_v', 'value_qk', 'qkv', 'scale_qkv', 'qkv_share', 'all'])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # should be in front of  "import torch"

    if args.config is None:
        run_dir = os.path.join(args.run_dir, args.run_name)
        args.config = os.path.join(run_dir, 'config_train.yaml')
        if not os.path.exists(args.config):
            args.config = 'config.yaml'
    
    from train_and_test import Runner
    Runner(args).run()
