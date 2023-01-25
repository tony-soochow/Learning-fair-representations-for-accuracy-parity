import pandas as pd
import numpy as np

desc = """
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
""".strip().replace('\n', '').split('.')
columns = list(map(lambda x: x.strip().split(':')[0], desc))
columns[-1] = 'income'

def create(path):
    tsne = False
    data = pd.read_csv(path, names=columns)
    data['workclass'] = data['workclass'].astype('category').cat.codes
    data['education'] = data['education'].astype('category').cat.codes
    data['occupation'] = data['occupation'].astype('category').cat.codes
    data['relationship'] = data['relationship'].astype('category').cat.codes
    data['marital-status'] = data['marital-status'].astype('category').cat.codes
    data['income'] = data['income'].astype('category').cat.codes
    data['gender'] = data['sex'].astype('category').cat.codes
    data['race'] = data['race'].astype('category').cat.codes

    data = data[['education-num',
                 'hours-per-week',
                 'relationship', 'occupation',
                 'capital-gain', 'capital-loss',
                 'marital-status', 'workclass', 'education', 'race', 'income', 'sex']]

    for var in list(data.columns):
        data = data[~data[var].isnull()]

    outcome = data['income']

    if tsne:
        sensitive = pd.get_dummies(data['sex'])
    else:
        data['race_gender'] = data['race'] + '_' + data['sex']
        sensitive = pd.get_dummies(data['race_gender'])

    features = data[['education-num', 'hours-per-week', 'relationship', 'occupation',
                     'capital-gain', 'capital-loss', 'marital-status', 'workclass',
                     'education']]
    features = (features - features.mean()) / features.std()

    print(f'none sensitive attribute names: {list(features.columns)}')
    print(f'sensitive attribute names: {list(sensitive.columns)}')
    print(f'x shape: {features.shape}')
    print(f'a shape: {sensitive.shape}')
    print(f'y shape: {outcome.shape}')
    return features, outcome, sensitive


def save_npz():
    x, y, a = create('adult.data')
    np.savez('../Madult.npz',
             x=x, y=y, a=a,
             input_columns=list(x.columns),
             s_columns=list(a.columns)
             )


if __name__ == '__main__':
    x, y, a = create('adult.data')
    print(a)


