import os
import yaml
import argparse

import numpy as np

def summary(path, search=[], verbose=False, draw_loss=False, limit=None, is_clear=False, threshold=None, SDR=None):
    if SDR is None:
        SDR = [2, 2.5, 3, 4, 6, 9, 10]
    dic = {}
    no_info = []
    savename = 'summary.yaml'
    if threshold is not None:
        savename = 'summary_th{}.yaml'.format(threshold)
    for run_name in os.listdir(path):
        if is_clear:
            check_pre = os.path.join(path, run_name, 'checkpoints')
            if os.path.exists(check_pre):
                bests = []
                for f in os.listdir(check_pre):
                    if f.startswith('best'):
                        bests.append(f)
                    else:
                        os.remove(os.path.join(check_pre, f))
                for f in bests[:-3:]:
                    os.remove(os.path.join(check_pre, f))

        pre1 = os.path.join(path, run_name, 'results')
        dest = '.loss_pngs/'+os.path.basename(path)+'_'+run_name
        if os.path.isdir(pre1):
            is_eval = False
            for cur_dir in os.listdir(pre1):
                if cur_dir.startswith('eval'):
                    eval_dir = os.path.join(pre1, cur_dir)
                    if os.path.isdir(eval_dir) and savename in os.listdir(eval_dir):
                        # print(eval_dir)
                        is_eval = True
                        with open(os.path.join(eval_dir, savename)) as f:
                            d = yaml.load(f.read())
                            del d['avg']
                            key_name = '{}_{}'.format(run_name, cur_dir[5:])
                            dic[key_name] = d
            if is_eval:
                if draw_loss:
                    draw_loss_curve(dest, os.path.join(pre1, 'loss'), run_name, limit)
            else:
                no_info.append(run_name)

    print('\n[{}/{}] {}'.format(len(dic), len(dic)+len(no_info), path))
    keys = sorted(list(dic.keys()), key=lambda name: name[:name.rfind('_epoch')])
    if search!=[]:
        keys = [k for k in keys if all([s in k for s in search])]
    # keys = sorted(keys, key=lambda k: cal(dic[k]), reverse=True)

    import pandas as pd
    for k in keys:
        print('----', k)
        if 'MEAN' in dic[k]:
            print('mean:', dic[k]['mean']['MRE'])

        data_names = ['cephalometric', 'hand', 'chest','spineweb16', 'avg', 'jsrt']
        # metrics = ['MRE', 'STD', 'ID', 'TH_ID', 'ORIGIN_NUM', 'NUM', 'DET_NUM', 'SDR']
        metrics = ['MRE', 'STD', 'SDR']
        # # TODO

        metrics += SDR
        df = pd.DataFrame(dic[k])


        df = df.reindex(index=metrics)
        # df = df.reindex(columns=data_names)
        df = df.T

        for sdr in SDR:
            df[sdr] = df['SDR'].map(lambda sdr_dic: sdr_dic[sdr] if isinstance(sdr_dic, dict) else sdr_dic)

        df = df.drop(['SDR'], axis=1)
        df = df.rename(index={'cephalometric': 'ceph'}, columns={})
        df = df.rename(index={'spineweb3': 'spin3'}, columns={})
        # df.loc['avg'] = (df.loc['ceph']*150 + df.loc['hand']*300
        print(df.round(decimals=4))

    if verbose and no_info:
        print('---- No eval:')
        for i in no_info:
            print(' ', i)


def cal(all_metrics):
    metrics = all_metrics['mask']
    data_names = ['NLPR_test', 'NJU2K_test', 'test_NLPR', 'test_NJU2K485', 'STERE']
    vals = [val  for dataset, metric in metrics.items() for key, val in metric.items() if dataset in data_names  and key in ['FMAX', 'SM', 'EMAX']]
    if any(np.isnan(i) for i in vals):
        print('Error NaN', vals)
        return 0
    else:
        return sum(vals)


def draw_loss_curve(dest, src, title, limit=None):
    info = [f[:-4] for f in os.listdir(src) if f.endswith('.txt')]
    data = {}
    for f in info:
        segs = f.split('_')
        epoch, dataname, loss = segs[1], segs[2], segs[-1]
        if len(segs)==4:
            dataname = 'mix'
        epoch = int(epoch)
        loss = float(loss)
        if epoch in data:
            if dataname not in data[epoch] or data[epoch][dataname]>loss:
                data[epoch][dataname] = loss
        else:
            data[epoch] = {dataname: loss}

    xs = sorted(list(data.keys()))
    if xs:
        legends = list(data[xs[0]].keys())
        fig = plt.figure()
        if limit is not None:
            plt.ylim(0, limit)
            dest = dest + '_lim{}'.format(limit)
        for idx, leg in enumerate(legends):
            ys = [data[x][leg]/512/512 for x in xs]
            plt.plot(xs, ys)
            best_x = min(xs, key=lambda x: data[x][leg])
            best_y = data[best_x][leg]
            plt.text(best_x, best_y, '{:.4f}'.format(best_y), fontsize=10)

        n = len(legends)
        if n>1:
            legends.append('sum')
            ys = [sum(data[x].values())/512/512 for x in xs]
            plt.plot(xs, ys)
        plt.xlabel('epoch')
        plt.title(title)
        plt.ylabel('loss')
        labels = [str(x) if i%3==0 else '' for i, x in enumerate(xs)]
        plt.xticks(xs, labels, rotation=270)
        plt.grid(False)
        plt.legend(legends)
        plt.savefig(dest)
        plt.close('all')

def get_args():
    parser = argparse.ArgumentParser()
    # required
    # optional
    parser.add_argument('-r', "--run_dir", nargs='+', type=str, default=[])
    parser.add_argument('-v', "--verbose", action='store_true')
    parser.add_argument('-s', "--search", nargs='+', default=[])
    parser.add_argument('-d', '--draw_loss', action='store_true')
    parser.add_argument('-l', '--limit', type=int)
    parser.add_argument('-t', '--threshold', type=int)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--SDR', nargs='+', type=float)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    for path in args.run_dir:
        if os.path.isdir(path):
            summary(path, args.search, args.verbose, args.draw_loss, args.limit, args.clear, args.threshold, args.SDR)
