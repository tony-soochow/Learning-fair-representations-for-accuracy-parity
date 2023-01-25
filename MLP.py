from sklearn.linear_model import LogisticRegression
import wandb
import argparse
from tools import seed_everything
import base_model
import tools

wandb.init(project="twin-fair", entity="tstk")

config = wandb.config
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--data', type=str, default='german_ss')

args = parser.parse_args()
seed_everything(args.seed)
print(args)
"************* setting configs *************"
config.batch_size = 64  # fixed
config.method = 'MLP'  # fixed
config.data = args.data
config.epoch = args.epoch
config.seed = args.seed

"************* loading data *************"
ds = tools.DataStream(config.data)
"************* train and test *************"
model = LogisticRegression()
x, y, s = ds.train_loader.dataset.tensors
kk = x, y, s
x, y, s = list(map(lambda v: v.cpu().numpy(), kk))
model.fit(x, y)
x1, y1, s1 = ds.test_loader.dataset.tensors
kk = x1, y1, s1
x1, y1, s1 = list(map(lambda v: v.cpu().numpy(), kk))
mtc = base_model.Metrics('acc', 'dp','dp2', 'ap', 'ap2', 'eo', 'eo2')
res = mtc(model_output=model.predict_proba(x1)[:, 1], samples=(x1, y1, s1))
# wandb.log(res)
print('final TEST:', res)
# wandb.watch(model)
