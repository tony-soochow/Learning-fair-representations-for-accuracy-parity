import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data = np.loadtxt('german.data-numeric')
print(data.shape)
age = np.where(data[:, 9] > 25, 0, 1)
sex = np.where(np.logical_or(data[:, 6] == 2, data[:, 6] == 5), 0, 1)
x = np.hstack([data[:, :6], data[:, 7: 9]])
x = np.hstack([x, data[:, 10: 24]])
for i in range(22):
    x[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()
y = data[:, 24]
y = np.where(y==1, 1, 0)
ls = []

for i in range(sex.shape[0]):
    if sex[i] == 0 and age[i] == 0:
        ls.append([1, 0, 0, 0])
    elif sex[i] == 1 and age[i] == 0:
        ls.append([0, 1, 0, 0])
    elif sex[i] == 0 and age[i] == 1:
        ls.append([0, 0, 1, 0])
    elif sex[i] == 1 and age[i] == 1:
        ls.append([0, 0, 0, 1])
    else:
        raise 1
ss = np.array(ls)
sex = np.vstack([1 - sex, sex]).T
age = np.vstack([1 - age, age]).T
print(x, x.shape)
print(sex)
print(ss)
print(ss.mean(axis=0))
x1, x2, y1, y2, sex1, sex2, age1, age2, ss1, ss2 = train_test_split(x, y, sex,
                                           age, ss, test_size=0.3, random_state=1, shuffle=True)
x3, x2, y3, y2, sex3, sex2, age3, age2, ss3, ss2 = train_test_split(x2, y2, sex2,
                                           age2, ss2, test_size=0.5, random_state=1, shuffle=True)
print(x1.shape, x2.shape, x3.shape)
np.savez('german_sex_train', x=x1, y=y1, s=sex1)
np.savez('german_sex_verify', x=x2, y=y2, s=sex2)
np.savez('german_sex_test', x=x3, y=y3, s=sex3)

np.savez('german_age_train', x=x1, y=y1, s=age1)
np.savez('german_age_verify', x=x2, y=y2, s=age2)
np.savez('german_age_test', x=x3, y=y3, s=age3)

np.savez('german_ss_train', x=x1, y=y1, s=ss1)
np.savez('german_ss_verify', x=x2, y=y2, s=ss2)
np.savez('german_ss_test', x=x3, y=y3, s=ss3)