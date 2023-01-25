import torch
import wandb
import argparse
import random
import numpy as np
import base_model
import tools


class Corr(torch.nn.Module):
    def __init__(self, input_size):
        print(args)
        super(Corr, self).__init__()
        p_dropout = 0.
        hidden = (32, 32)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(hidden[0], hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(hidden[1], 2)
        )

    def corr(self, x, y):
        """
        相关系数 越低，越不相关
        """
        xm, ym = torch.mean(x), torch.mean(x)
        xvar = torch.sum((x - xm) ** 2) / x.shape[0]
        yvar = torch.sum((y - ym) ** 2) / x.shape[0]
        return torch.abs(torch.sum((x - xm) * (y - ym)) / (xvar * yvar) ** 0.5)

    def forward(self, sample):
        """
        y: [N,]
        s: [N, 1]
        :param sample:
        :return:
        """
        x, y, s = sample
        y_predict = self.network(x)
        ce_loss = torch.nn.CrossEntropyLoss()(y_predict, y.view(-1).long())
        y_predict = torch.softmax(y_predict, dim=-1)
        a_tags = torch.argmax(s, dim=1).float()
        c_loss = self.corr(y_predict[:, 1] * y, a_tags)
        return ce_loss, config.f1 * c_loss

    def fair_representation(self, x):
        return torch.softmax(self.network(x), dim=-1)


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
    for _ in range(epoch):
        for sample in ds.train_loader:
            if sample[0].shape[0] < 50:
                continue
            es.pack(m(sample))
            loss = es.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print({**es.dict(),
               'epoch': _})


def test(m: torch.nn.Module, ds: tools.DataStream):
    mtc = base_model.Metrics('acc', 'ap2', 'ap', 'dp', 'dp2', 'eo', 'eo2', )
    with torch.no_grad():
        x2, y2, s2 = ds.test_loader.dataset.tensors
        y_pred = m.fair_representation(x2)
        y_pred = torch.softmax(y_pred, dim=1).cpu().numpy()[:, 1]
        res = mtc(model_output=y_pred, samples=(
            x2.numpy(), y2.numpy(), s2.numpy()
        ))
    return res




def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    wandb.init(project="twin-fair", entity="tstk")

    config = wandb.config
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--data', type=str, default='german_ss')
    parser.add_argument('--f1', type=float, default=0.03)

    args = parser.parse_args()
    seed_everything(args.seed)
    print(args)

    "************* setting configs *************"
    config.batch_size = 64  # fixed
    config.method = 'corr'  # fixed
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
    model = Corr(ds.x_dim)
    opt = torch.optim.Adam(model.parameters())
    mtc = base_model.Metrics('acc', 'dp', 'dp2', 'ap', 'ap2', 'di', 'eo', 'eo2')
    train(model, opt, ds, config.epoch)
    res = test(model, ds)
    wandb.log(res)
    print('final TEST:', res)
    wandb.watch(model)
