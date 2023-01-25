import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import tools
import wandb
import argparse
import base_model
import numpy as np


class GradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural
    network.
    The forward part is the identity function while the backward part is the negative
    function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class CFairNet(torch.nn.Module):
    """
    Multi-layer perceptron with adversarial training for conditional fairness.
    """

    def __init__(self, input_size, mu, device):
        """

        :param configs: num_classes: 敏感类别数目
        input_dim
        hidden_layers
        """
        super(CFairNet, self).__init__()
        hidden = 32
        configs = {"num_classes": 2, "num_groups": 2,
                   "lr": 0.1,
                   "hidden_layers": [32],
                   "adversary_layers": [32],
                   "mu": 100}
        self.input_dim = input_size
        self.num_classes = 2
        self.num_hidden_layers = 1
        self.num_adversaries_layers = 1
        self.num_neurons = [self.input_dim, hidden]
        self.softmax = nn.Linear(self.num_neurons[-1], 2)
        self.num_adversaries = [self.num_neurons[-1], hidden]

        """ net arch"""
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        # Parameter of the conditional adversary classification layer.
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for
        # one class label.
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                         for i in range(self.num_adversaries_layers)])
                                          for _ in range(self.num_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.num_classes)])

        # self.opt = torch.optim.Adam(self.parameters())

        self.mu = mu
        self.device = device

    def forward(self, sample, ws):
        reweight_target_tensor, reweight_attr_tensors = ws
        xx, yy, ss = sample
        xx = xx.to(self.device)
        yy = yy.to(self.device)
        ss = ss.to(self.device)
        # reweight_target_tensor = reweight_target_tensor.to(self.device)
        # reweight_attr_tensors = reweight_attr_tensors.to(self.device)
        h_relu = xx
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        # Adversary classification component.
        c_losses = []
        h_relu = grad_reverse(h_relu)
        for j in range(self.num_classes):
            idx = yy == j
            c_h_relu = h_relu[idx]
            for hidden in self.adversaries[j]:
                c_h_relu = F.relu(hidden(c_h_relu))
            c_cls = F.log_softmax(self.sensitive_cls[j](c_h_relu), dim=1)
            c_losses.append(c_cls)
        reweight_target_tensor = reweight_target_tensor.to(self.device)
        loss = F.nll_loss(logprobs, yy.long(), weight=reweight_target_tensor)
        adv_loss = torch.mean(torch.stack([F.nll_loss(c_losses[j], ss[yy == j][:, 1].long(),
                                                      weight=reweight_attr_tensors[j].to(self.device))
                                           for j in range(self.num_classes)]))

        return loss, self.mu * adv_loss

    def fair_predict(self, inputs):
        inputs = inputs.to(self.device)
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # return F.log_softmax(self.softmax(h_relu), dim=1)
        return self.softmax(h_relu)


def get_W(samples):
    X1, y1, A1 = samples
    train_y_1 = np.mean(y1)
    train_idx = (A1[:, 0] == 0)
    train_base_0 = np.mean(y1[train_idx])
    train_base_1 = np.mean(y1[~train_idx])

    reweight_target_tensor = torch.Tensor([1.0 / (1.0 - train_y_1), 1.0 / train_y_1])

    reweight_attr_0_tensor = torch.Tensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0])
    reweight_attr_1_tensor = torch.Tensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1])

    reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]

    return reweight_target_tensor, reweight_attr_tensors


def train(m: torch.nn.Module, opt: torch.optim.Optimizer, ds: tools.DataStream, epoch: int):
    """
    base fir function... just compute the loss and next
    :param m:
    :param opt:
    :param ds:
    :param epoch:
    :return:
    """
    es = tools.LossRecoder()
    ws = get_W(list(map(lambda x: x.numpy(), ds.train_loader.dataset.tensors)))
    print(ws)
    for _ in range(epoch):
        for sample in ds.train_loader:
            if sample[0].shape[0] < 50:
                continue
            es.pack(m(sample, ws))
            loss = es.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print({**es.dict(),
               'epoch': _})


def test(m: torch.nn.Module, ds: tools.DataStream):
    mtc = base_model.Metrics('acc', 'ap2','ap', 'dp','dp2', 'eo', 'eo2', )
    with torch.no_grad():
        x2, y2, s2 = ds.test_loader.dataset.tensors
        y_pred = m.fair_predict(x2)
        y_pred = torch.softmax(y_pred, dim=1).cpu().numpy()[:, 1]
        res = mtc(model_output=y_pred, samples=(
            x2.numpy(), y2.numpy(), s2.numpy()
        ))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--data', type=str, default='compas')
    parser.add_argument('--f1', type=float, default=100)
    args = parser.parse_args()
    print(args)
    tools.seed_everything(args.seed)
    wandb.init(project="twin-fair", entity="tstk")
    "************* setting configs *************"
    config = wandb.config
    config.method = 'c-fair'
    config.batch_size = 64  # fixed
    config.zdim = 10
    config.f1 = args.f1
    config.data = args.data
    config.epoch = args.epoch
    config.seed = args.seed
    config.device = 'cuda'
    ds = tools.DataStream(config.data)
    model = CFairNet(ds.x_dim, mu=config.f1, device=config.device)
    model.to(config.device)
    opt = torch.optim.Adam(model.parameters())
    train(model, opt, ds, config.epoch)
    print('*' * 30, 'FINISH TRAIN', '*' * 30)
    res = test(model, ds)
    wandb.log(res)
    print('FINISH TEST:', res)
    wandb.watch(model)
