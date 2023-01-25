import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def draw(dataset, metrics):
    df = pd.read_excel('exp/exps.xlsx')
    ee = 'Name 	ac_gap acc	eo2_gap	eo_gap	dp_gap	method		seed	f1	f2 data'.split()
    df.columns = ee
    df = df.dropna(how='all')
    data = dataset
    df = df[df['data'] == data]
    ylim = {
        'compas': (0.66, 0.8),
        'adult': (0.72, 0.86),
        'german': (0.71, 0.83),

    }
    xlim = {
        'compas': (0., 0.25),
        'adult': (0, 0.2),
        'german': (0.0, 0.3),

    }
    x_label = {
        'ac_gap': r'$\Delta AC$',
        'eo2_gap': r'$\Delta$ FPR',
        'eo_gap': r'$\Delta$ FNR',
        'dp_gap': r'$\Delta$ SP',
    }
    print(df)
    df = df[df['acc'] > 0.6]
    models = df.groupby('method')
    fair = metrics
    plt.figure(figsize=(10, 5))
    m = ['D', '<', 's', 'H', 'X', '>']

    for i, (key, s) in enumerate(models):
        ls = list(zip(s[fair], s['acc']))
        ls.sort(key=lambda x: x[0])
        gs = {}
        for a, b in ls:
            for ff in range(0, 400, 5):
                gg = ff / 1000
                if gg - 0.005 < a < gg:
                    gs[gg] = gs.get(gg, [])
                    gs[gg].append(b)
        bs = []
        for k in gs.keys():
            bs.append([k, *np.percentile(gs[k], (25, 50, 75))])
        bs = np.array(bs)
        plt.scatter(bs[:, 0], bs[:, 2], label=key, marker=m[i], s=100)
    font = {
        'size': 14
    }
    if data == 'german_ss':
        data = 'german'
    plt.title(f'{data}'.title(), fontdict=font)
    plt.legend()
    plt.grid(linestyle='-.')
    plt.ylabel('Accuracy', fontdict=font)
    plt.xlabel(x_label[fair], fontdict=font)
    plt.xlim(*xlim[data])
    plt.ylim(*ylim[data])
    plt.show()


draw('adult', 'ac_gap')
draw('compas', 'ac_gap')
draw('german_ss', 'ac_gap')