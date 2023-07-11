import numpy as np


class Evaluater(object):
    def __init__(self):
        self.RE_list = list()
        self.recall_radius = [2, 2.5, 3, 4, 6, 9, 10]

    def reset(self):
        self.RE_list.clear()

    def record(self, pred, landmark, ratios=(1, 1)):
        # n = batchsize = 1
        # pred : list[ c(y) ; c(x) ]
        # landmark: list [ (x , y) * c]
        c = pred[0].shape[0]
        diff = np.zeros([c, 2], dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][1]) * ratios[0]
            diff[i][1] = abs(pred[1][i] - landmark[i][0]) * ratios[1]
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        self.RE_list.append(Radial_Error)


    def cal_metrics(self):
        temp = np.array(self.RE_list)
        Mean_RE_channel = temp.mean(axis=0)
        mre = Mean_RE_channel.mean()
        print('MRE channel-wise', Mean_RE_channel)

        metric_data = {'MRE': mre, 'SDR':{}}
        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            ratio = shot*100/total
            metric_data['SDR'][int(radius*10)/10] = ratio
        return metric_data
