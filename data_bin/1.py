import torch

s = torch.Tensor([1, 0, 1]).mean()
print(torch.hstack([s, s, s]))
print((s.mean() + s.mean()).sum())
import numpy as np

ls = np.percentile(np.asarray([1, 2, 2, 4, 5]), (25, 50, 75))
print(ls)
