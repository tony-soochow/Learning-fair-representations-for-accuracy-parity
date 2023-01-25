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
        self.device='cpu'

        self.z_dim=config.zdim

        self.fc1 = torch.nn.Linear(input_size, 16)
        self.fc2 = torch.nn.Linear(16, self.z_dim)
        self.fc3 = torch.nn.Linear(16, self.z_dim)
        self.fc4 = torch.nn.Linear(self.z_dim, 16)
        self.fc5 = torch.nn.Linear(16, input_size)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, z_dim),
        )


        self.axx = [0, 1]
        self.bxx = [0, 0, 0, 0, 1, 1, 1, 1]
        self.split = 2
        self.alpha = config.alpha









    def corr(self, x, y):
        """
        相关系数 越低，越不相关
        """
        xm, ym = torch.mean(x), torch.mean(x)
        xvar = torch.sum((x - xm) ** 2) / x.shape[0]
        yvar = torch.sum((y - ym) ** 2) / x.shape[0]
        return torch.abs(torch.sum((x - xm) * (y - ym)) / (xvar * yvar) ** 0.5)

    def out(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return mu, log_var, x_reconst, z

    def decode(self, z):
        "decode x from z"
        h = torch.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def reparameterize(self, mu, log_var):
        "reparameterize z from mu, log_var"
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        "get encode mu, logvar"
        h = torch.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def corr_loss_fun(self, sensitive, y, save_sensitive_ids, remove_sensitive_ids):
        """
        :param sensitive: sensitive array
        :param y: representation array
        :param save_sensitive_ids: y index that need save sensitive message
        :param remove_sensitive_ids: y index that need remove sensitive message
        :return: total CORR
        """
        ans = 0
        ii = 0

        for i in save_sensitive_ids:
            ans -= self.corr(sensitive[:, [i]], y[:, [ii]])
            ii += 1
        for i in remove_sensitive_ids:
            ans += self.corr(sensitive[:, [i]], y[:, [ii]])
            ii += 1
        return ans

    def forward(self, data):
        x, y, s = data
        x = x.to(self.device)
        y = y.to(self.device)
        s = s.to(self.device)
        mu, log_var, x_reconst, z = self.out(x)

        reconst_loss = torch.nn.MSELoss(reduction='sum')(x_reconst, x)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # loss = reconst_loss + 0.01 * kl_div

        corr_loss = self.corr_loss_fun(s, z, self.axx, self.bxx)
        # loss += self.alpha * corr_loss
        # self.opt.zero_grad()
        # loss.backward()
        # self.opt.step()
        # 其他数据集是0.01
        return reconst_loss, 0.001 * kl_div, self.alpha * corr_loss


    def representation(self, x):
        mu, log_var, x_reconst, z = self.out(x)
        return z


    def fair_representation(self, x):
        z = self.representation(x)
        # z[:, :self.split] = torch.randn_like(z[:, :self.split])
        return z[:, self.split:]




def train(m, opt, epochs):
    for i in range(epochs):
        sr = tools.LossRecoder()
        for sample in ds.train_loader:
            l1, l2, l3 = m.forward(sample)
            sr.add(l1, l2, l3)
            loss = l1 + l2 + l3
            opt.zero_grad()
            loss.backward()
            opt.step()
        print({**sr.dict(),
               'epoch': i})
        if i % 10 == 0:
            res = test(m, ds.test_loader.dataset.tensors, ds.validate_loader.dataset.tensors)
            print(res)


def test(m, d1, d2):
    x, y, s = d1
    x_v, y_v, s_v = d2
    with torch.no_grad():
        lr = LogisticRegression(max_iter=2000)
        in_ = m.fair_representation(x_v).cpu().numpy()
        print(in_.shape)
        la_ = y_v.cpu().numpy()
        lr.fit(in_, la_)
        y_pred = lr.predict_proba(m.fair_representation(x).cpu().numpy())[:, 1]

        return mtc(model_output=y_pred, samples=(
            x.numpy(), y.numpy(), s.numpy()
        ))


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)




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
    config.method = 'FRC'  # fixed
    config.zdim = 10  # 10
    config.data = args.data
    config.epoch = args.epoch
    config.seed = args.seed
    config.alpha = args.f1

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
    res = test(model, ds.test_loader.dataset.tensors, ds.validate_loader.dataset.tensors)
    wandb.log(res)
    print('final TEST:', res)
    wandb.watch(model)
