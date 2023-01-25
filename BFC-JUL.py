import wandb
import torch
from torch.utils.data import TensorDataset
import numpy as np
import logging
import base_model
import random
import argparse
import tools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s=>%(message)s')
logger = logging.getLogger(__name__)


class FairRep(torch.nn.Module):

    def __init__(self, input_size, config):
        super(FairRep, self).__init__()
        z_dim = config.zdim
        hidden = (32, 32)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, z_dim),
        )

        self.cla = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden[0], 2),
        )

    def distance(self, x, y):
        x_mu = torch.mean(x, dim=0)
        y_mu = torch.mean(y, dim=0)
        x_var = torch.var(x, dim=0)
        y_var = torch.var(y, dim=0)
        static_ = torch.sum((x_mu - y_mu) ** 2) + torch.sum((x_var - y_var) ** 2)
        return static_

    # def encoder(self, x):
    #     z = torch.empty(size=[x.shape[0], len(self.encoders)])
    #     for i in range(len(self.encoders)):
    #         z_i = self.encoders[i](x)
    #         z[:, [i]] = z_i
    #     return z

    def twin_loss_fun(self, z, y, s):
        twin_loss = 0
        for yyy in range(2):
            ids = torch.where(y == yyy)[0]
            z_y = z[ids]
            z_a = s[ids]
            a_tag = torch.argmax(z_a, dim=1)
            mus = []
            for j in range(s.shape[1]):
                iids = torch.where(a_tag == j)[0]
                points_j = z_y[iids]
                mus.append(points_j)

            for ii in range(len(mus) - 1):
                g1 = mus[ii]
                g2 = mus[ii + 1]
                if g1.shape[0] < 2 or g2.shape[0] < 2:
                    continue
                twin_loss += self.distance(g1, g2)
        return twin_loss

    def loss_t(self, z, y, s):
        twin_loss = 0
        for yyy in range(2):
            ids = torch.where(y == yyy)[0]
            z_y = z[ids]
            z_a = s[ids]
            a_tag = torch.argmax(z_a, dim=1)
            mus = []
            for j in range(s.shape[1]):
                iids = torch.where(a_tag == j)[0]
                points_j = z_y[iids]
                mus.append(points_j)

            for ii in range(len(mus) - 1):
                g1 = mus[ii]
                g2 = mus[ii + 1]
                if g1.shape[0] < 2 or g2.shape[0] < 2:
                    continue
                for k in range(g1.shape[1]):
                    twin_loss += self.corr(g1[:, k], g2[:, k])
        return twin_loss

    def cla_diff(self, z, y, s):
        cla_diff_loss = 0
        a_tag = torch.argmax(s, dim=1)
        mus = []
        for j in range(s.shape[1]):
            iids = torch.where(a_tag == j)[0]
            z_j = z[iids]
            y_j = y[iids]
            mus.append(torch.nn.CrossEntropyLoss(reduction='mean')(self.cla(z_j), y_j.long()))

        for ii in range(len(mus) - 1):
            g1 = mus[ii]
            g2 = mus[ii + 1]
            cla_diff_loss += abs(g1 - g2)

        return cla_diff_loss

    def corr(self, x, y):
        """
        相关系数 越低，越不相关
        """
        xm, ym = torch.mean(x), torch.mean(x)
        xvar = torch.sum((x - xm) ** 2) / x.shape[0]
        yvar = torch.sum((y - ym) ** 2) / x.shape[0]
        return torch.abs(torch.sum((x - xm) * (y - ym)) / (xvar * yvar) ** 0.5)

    def forward(self, sample):
        x, y, a = sample
        z = self.encoder(x)
        twin_loss = self.twin_loss_fun(z, y, a)
        a_tags = torch.argmax(a, dim=1).float()
        class_loss = torch.nn.CrossEntropyLoss(reduction='none')(self.cla(z), y.long())

        return torch.sum(class_loss), config.f1 * self.corr(a_tags, class_loss), 0.1 * twin_loss


def train(m, opt, epochs):
    for i in range(epochs):
        sr = tools.LossRecoder()
        for sample in ds.train_loader:
            l1, l2, l3 = m(sample)
            sr.add(l1, l2, l3)
            loss = l1 + l2 + l3
            opt.zero_grad()
            loss.backward()
            opt.step()
        print({**sr.dict(),
               'epoch': i})
        if i % 10 == 0:
            res = test(m, *ds.test_loader.dataset.tensors)
            print(res)


def test(m, x, y, s):
    with torch.no_grad():
        z = m.encoder(x)
        return mtc(model_output=torch.softmax(m.cla(z), dim=1).numpy()[:, 1], samples=(
            x.numpy(), y.numpy(), s.numpy()
        ))


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

class AF:
    def __init__(self):
        pass


if __name__ == '__main__':
    wandb.init(project="twin-fair", entity="tstk")

    config = wandb.config
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--data', type=str, default='compas')
    parser.add_argument('--f1', type=float, default=1)

    args = parser.parse_args()
    seed_everything(args.seed)
    print(args)

    "************* setting configs *************"
    config.batch_size = 64  # fixed
    config.method = 'BFA'  # fixed
    config.zdim = 10  # 10
    config.data = args.data
    config.epoch = args.epoch
    config.seed = args.seed
    config.f1 = args.f1

    "************* loading data *************"
    ds = tools.DataStream(config.data)
    print(ds.train_loader.dataset.tensors[2].mean(dim=0))
    atags = torch.argmax(ds.train_loader.dataset.tensors[2], dim=1)
    print(ds.train_loader.dataset.tensors[1][atags == 0].mean())
    print(ds.train_loader.dataset.tensors[1][atags == 1].mean())
    "************* train and test *************"
    model = FairRep(ds.x_dim, config)
    opt = torch.optim.Adam(model.parameters())
    mtc = base_model.Metrics('acc', 'dp', 'dp2', 'ap', 'ap2', 'di', 'eo', 'eo2')
    train(model, opt, config.epoch)
    res = test(model, *ds.test_loader.dataset.tensors)
    wandb.log(res)
    print('final TEST:', res)
    wandb.watch(model)
