import wandb
import torch
from torch.utils.data import TensorDataset

import numpy as np
import logging
import abc
import random
import copy


class BaseModel(torch.nn.Module):
    def __init__(self,
                 input_size,
                 wandb,
                 metrics):
        super(BaseModel, self).__init__()

        self.wandb = wandb
        self.input_size = input_size
        self.metrics = metrics
        self.network = None

    def test(self, samples):
        """
        :param samples:
        元组 （X, y, ...）

        :return:
        """
        with torch.no_grad():
            y_pred = self.model_out(samples)
            metrics_result = self.metrics(samples, y_pred)
        return metrics_result

    @abc.abstractmethod
    def model_out(self, output):
        pass


class Metrics:
    """
    mtc = Metrics("metrics name")
    mtc(net_output, *instance)
    # the type of net_output and vector in instance are both numpy array
    """

    def __init__(self, *metrics_names):
        self.all_metrics = {
            'acc': self.acc,
            'dp': self.dp,
            'sdp': self.sdp,
            'eo': self.eo,
            'eo2': self.eo2,
            'dp2': self.dp2,
            'ap': self.ap,
            'ap2': self.ap2,
            'di': self.di
        }

        self.keys = list(self.all_metrics.keys())
        self.funcs = {}
        for metrics in metrics_names:
            v = self.all_metrics.get(metrics, None)
            if v:
                self.funcs[metrics] = v
            else:
                raise KeyError(f'No metrics named {metrics}, use add_metrics add your own metric.')
        print(f'metrics options: {self.keys}')

    def __call__(self, model_output, samples):
        out = {}
        for func in self.funcs:
            v = self.funcs[func](model_output, *samples)
            self.fill_dict(out, func, v)
        return out

    def add_metrics(self, m_name, fn):
        assert fn.__code__.co_argcount == 2
        self.funcs[m_name] = fn

    def fill_dict(self, d, k, v):
        if type(v) == dict:
            for k in v:
                vv = v[k]
                self.fill_dict(d, k, vv)
        else:
            d[k] = v

    def acc(self, model_output, *samples):
        y_truth = samples[1]
        y_predict = model_output
        y_predict = np.where(y_predict > 0.5, 1, 0)
        return np.mean(y_predict == y_truth)

    def dp(self, model_output, *samples):
        y_predict = model_output
        y_pred_cla = np.where(y_predict > 0.5, 1, 0)
        a_truth = samples[2]
        dps = {}
        kn = 0
        for j in range(a_truth.shape[1]):
            a_single = a_truth[:, j]
            y_pred_cla_a0 = y_pred_cla[np.where(a_single == 0)]
            y_pred_cla_a1 = y_pred_cla[np.where(a_single == 1)]
            diff_abs_dp = abs((y_pred_cla_a0.mean() - y_pred_cla_a1.mean()))
            dps[f'dp_{j}'] = diff_abs_dp
            kn += diff_abs_dp * a_truth[:, j].mean()
        dps['dp_avg'] = kn
        return dps

    def dp2(self, model_output, *samples):
        y_predict = model_output
        y_pred_cla = np.where(y_predict > 0.5, 1, 0)
        a_truth = samples[2]
        dps = []
        for j in range(a_truth.shape[1]):
            a_single = a_truth[:, j]
            y_pred_cla_a1 = y_pred_cla[np.where(a_single == 1)]
            dps.append(y_pred_cla_a1.mean())

        return {'max_diff_dp': max(dps) - min(dps)}

    def di(self, model_output, *samples):
        y_predict = model_output
        y_pred_cla = np.where(y_predict > 0.5, 1, 0)
        a_truth = samples[2]
        dps = []
        for j in range(a_truth.shape[1]):
            a_single = a_truth[:, j]
            y_pred_cla_a1 = y_pred_cla[np.where(a_single == 1)]
            dps.append(y_pred_cla_a1.mean())

        return {'DI': min(dps) / max(dps)}

    def eo(self, model_output, *samples):
        y_predict = model_output
        _, y_truth, a_truth = samples
        y_pred_cla = np.where(y_predict > 0.5, 1, 0)
        a_tags = np.argmax(a_truth, axis=1)
        eo_ = []
        for j in range(a_truth.shape[1]):
            ids = np.where(np.logical_and(a_tags == j, y_truth == 1))
            eo_.append(y_pred_cla[ids].mean())
        return {'eo_max_diff': max(eo_) - min(eo_)}

    def eo2(self, model_output, *samples):
        y_predict = model_output
        _, y_truth, a_truth = samples
        y_pred_cla = np.where(y_predict > 0.5, 1, 0)
        a_tags = np.argmax(a_truth, axis=1)
        eo_ = []
        for j in range(a_truth.shape[1]):
            ids = np.where(np.logical_and(a_tags == j, y_truth == 0))
            eo_.append(y_pred_cla[ids].mean())
        return {'eo2_max_diff': max(eo_) - min(eo_)}

    def ap(self, model_output, *samples):
        y_predict = model_output
        _, y_truth, a_truth = samples
        y_pred_cla = np.where(y_predict > 0.5, 1, 0)
        y_pred_cla = np.where(y_pred_cla == y_truth, 1, 0)
        a_tags = np.argmax(a_truth, axis=1)
        eo_ = []
        for j in range(a_truth.shape[1]):
            ids = np.where(a_tags == j)
            eo_.append(y_pred_cla[ids].mean())
        return {'accuracy_max_disparity': max(eo_) - min(eo_)}

    def ap2(self, model_output, *samples):
        y_predict = model_output
        y_pred_cla = np.where(y_predict > 0.5, 1, 0)
        _, y_truth, a_truth = samples
        kks = y_pred_cla == y_truth
        kks = np.where(kks, 1, 0)
        a_truth = samples[2]
        dps = {}
        kn = 0
        for j in range(a_truth.shape[1]):
            a_single = a_truth[:, j]
            y_pred_cla_a0 = kks[np.where(a_single == 0)]
            y_pred_cla_a1 = kks[np.where(a_single == 1)]
            diff_abs_dp = abs((y_pred_cla_a0.mean() - y_pred_cla_a1.mean()))
            dps[f'ap_{j}'] = diff_abs_dp
            kn += diff_abs_dp * a_truth[:, j].mean()
        dps['ap_avg'] = kn
        return dps

    def sdp(self, model_output, *samples):
        a_truth = samples[2]
        y_predict = model_output
        y_predict_a0 = y_predict[np.where(a_truth == 0)]
        y_predict_a1 = y_predict[np.where(a_truth == 1)]
        diff_abs_sdp = (y_predict_a0.mean() - y_predict_a1.mean()).abs()
        return diff_abs_sdp


def fun(a, b):
    return -torch.sum(a * torch.log(b) + (1 - a) * torch.log(1 - b))


class K_means():
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def distance(self, p1, p2):
        return torch.sum((p1 - p2) ** 2).sqrt()

    def generate_center(self):
        # 随机初始化聚类中心
        n = self.data.shape[0]
        rand_id = random.sample(range(n), self.k)
        center = []
        for id in rand_id:
            center.append(self.data[id])
        return center

    def converge(self, old_center, new_center):
        # 判断是否收敛
        set1 = set(old_center)
        set2 = set(new_center)
        return set1 == set2

    def forward(self):
        center = self.generate_center()
        n = self.data.shape[0]
        labels = torch.zeros(n).long()
        flag = 100
        while flag > 0:
            old_center = copy.deepcopy(center)

            for i in range(n):
                cur = self.data[i]
                min_dis = 10 * 9
                for j in range(self.k):
                    dis = self.distance(cur, center[j])
                    if dis < min_dis:
                        min_dis = dis
                        labels[i] = j

            # 更新聚类中心
            for j in range(self.k):
                center[j] = torch.mean(self.data[labels == j], dim=0)

            flag = flag - 1

        return labels, center
