import torch, os
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from sklearn.metrics import roc_auc_score


class FAD_class:
    def __init__(
            self,
            input_size,  # input size
            num_layers_z,  # no. layers in first network
            num_layers_y,  # no. layers in sensitive and output networks
            step_z,  # step in first network
            step_y,  # step in sensitive and output networks
            name="FAD",  # name of model
            save_dir=None,  # directory where weights should be saved
    ):

        self.device = torch.device("cpu")
        self.model = FAD_class.FairClass(
            input_size, num_layers_z, num_layers_y, step_z, step_y
        )
        self.model.to(self.device)
        self.path = os.path.join(
            save_dir,
            name,
        )

    class FairClass(nn.Module):
        def __init__(self, inp_size, num_layers_z, num_layers_y, step_z, step_y):
            super(FAD_class.FairClass, self).__init__()

            num_layers_A = num_layers_y
            lst_z = nn.ModuleList()
            lst_1 = nn.ModuleList()
            lst_2 = nn.ModuleList()
            out_size = inp_size

            lst_z.append(nn.Sequential(
                nn.Linear(inp_size, 8),
                nn.BatchNorm1d(num_features=8),
                nn.ReLU(),
            ))
            lst_z.append(nn.Sequential(
                nn.Linear(8, 6),
                nn.BatchNorm1d(num_features=6),
                nn.ReLU(),
            ))

            lst_1.append(nn.Sequential(
                nn.Linear(6, 3),
                nn.BatchNorm1d(num_features=3),
                nn.ReLU(),
            ))
            lst_1.append(nn.Sequential(
                nn.Linear(3, 1)
            ))

            lst_2.append(nn.Sequential(
                nn.Linear(6, 3),
                nn.BatchNorm1d(num_features=3),
                nn.ReLU(),
            ))
            lst_2.append(nn.Sequential(
                nn.Linear(3, 1)
            ))
            self.fc1 = nn.Sequential(*lst_z)
            self.fc2 = nn.Sequential(*lst_1)
            self.fc3 = nn.Sequential(*lst_2)


        def forward(self, x):
            z = self.fc1(x)
            y = torch.sigmoid(self.fc2(z))
            A = torch.sigmoid(self.fc3(z))
            return y, A

    def fit(
            self,
            dataloader,  # train dataloader
            dataloader_val,  # validation dataloader
            early_stopping_no=3,  # early stopping no.
            max_epoch=300,  # max_epochs
            alpha=1,  # hyperparameter \alpha
            log=1,  # evaluate validation loss {1 - Yes, 0 - No}
            log_epoch=1,  # no. of epoch for evaluation of validation loss
            learning_rate=0.0001,  # learning rate
    ):

        self.model.train()
        nll_criterion = F.binary_cross_entropy
        list_1 = list(self.model.fc1.parameters()) + list(self.model.fc2.parameters())
        list_2 = list(self.model.fc3.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr=learning_rate)
        optimizer_2 = torch.optim.Adam(list_2, lr=learning_rate)

        prev_loss_y = 9e10
        no_val = 0

        for e in range(max_epoch):
            print(e)
            for batch_x, batch_y, batch_A in dataloader:
                batch_x = batch_x.to(self.device, dtype=torch.float)
                batch_y = batch_y.unsqueeze(dim=1).to(self.device, dtype=torch.float)
                batch_A = batch_A.unsqueeze(dim=1).to(self.device, dtype=torch.float)

                output_1, output_2 = self.model(batch_x)
                batch_A = batch_A.argmax(dim=2).float()
                loss2 = nll_criterion(output_2, batch_A)
                optimizer_2.zero_grad()
                loss2.backward()
                optimizer_2.step()

                output_1, output_2 = self.model(batch_x)
                loss1 = nll_criterion(output_1, batch_y) - alpha * nll_criterion(
                    output_2, batch_A
                )

                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()

            if e % log_epoch == 0 and log == 1:

                self.model.eval()
                for x_val, y_val, A_val in dataloader_val:

                    x_val = x_val.to(self.device, dtype=torch.float)
                    y_val = y_val.unsqueeze(dim=1).to(self.device, dtype=torch.float)
                    A_val = A_val.unsqueeze(dim=1).to(self.device, dtype=torch.float).argmax(dim=2).float()

                    out_1_val, out_2_val = self.model(x_val)

                    loss_y_val = (
                        (
                                nll_criterion(out_1_val, y_val)
                                - alpha * nll_criterion(output_2, batch_A)
                        )
                            .data.cpu()
                            .numpy()
                    )
                    loss_A_val = nll_criterion(out_2_val, A_val).data.cpu().numpy()

                    if loss_y_val > prev_loss_y:
                        no_val += 1
                    else:
                        prev_loss_y, _ = loss_y_val, loss_A_val
                        torch.save(self.model.state_dict(), self.path)
                        print("Model saved")
                        no_val = 0

                # if no_val == early_stopping_no:
                #     break

    def predict(self, x_test):  # Inference
        self.model.eval()
        y, A = self.model(x_test)
        y = np.round(y.data)
        A = np.round(A.data)
        return y, A

    def predict_proba(self, dataloader):  # Evaluation for given dataloader
        self.model.to(self.device).eval()
        self.model.load_state_dict(torch.load(self.path))
        for batch_x, _, _ in dataloader:
            y, A = self.model(batch_x.to(self.device, dtype=torch.float))
            y, A = y.data.cpu().numpy(), A.data.cpu().numpy()
        return y, _


import tools
import base_model


def expm(config):
    ds = tools.DataStream(config.data)
    (
        num_layers_z,
        num_layers_y,
        num_layers_w,
        num_layers_A,
        step_z,
        step_y,
        step_A,
        step_w,
    ) = [2, 2, 2, 2, 1.5, 1.5, 1.5, 1.5]

    m = FAD_class(
        input_size=ds.x_dim,
        num_layers_z=num_layers_z,
        num_layers_y=num_layers_y,
        step_z=num_layers_A,
        step_y=step_A,
        name="FAD",  # name of model
        save_dir='data/Trained_models',
    )
    m.fit(ds.train_loader, ds.validate_loader, max_epoch=config.epoch, alpha=config.f1)
    mtc = base_model.Metrics('acc', 'dp', 'eo', 'eo2', 'ap', 'ap2', 'dp2')
    with torch.no_grad():
        x2, y2, s2 = ds.test_loader.dataset.tensors
        y_pred = m.predict(x2)[0]
        y_pred = np.squeeze(y_pred)
        res = mtc(model_output=y_pred, samples=(
            x2.numpy(), y2.numpy(), s2.numpy()
        ))
    return res, m


import argparse
import wandb

if __name__ == '__main__':
    alpha = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--data', type=str, default='compas')
    parser.add_argument('--f1', type=float, default=10)
    args = parser.parse_args()
    print(args)

    tools.seed_everything(args.seed)
    wandb.init(project="twin-fair", entity="tstk")
    "************* setting configs *************"
    config = wandb.config
    config.method = 'FAD'
    config.f1 = args.f1
    config.data = args.data
    config.epoch = args.epoch
    config.seed = args.seed
    config.device = 'cpu'
    res, m = expm(config)
    print('*' * 30, 'FINISH TRAIN', '*' * 30)
    wandb.log(res)
    print('FINISH TEST:', res)
    wandb.watch(m.model)
