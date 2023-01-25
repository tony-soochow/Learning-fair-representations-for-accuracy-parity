import pandas as pd
import numpy as np


def adult(path):
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
    data = pd.read_csv(path, names=columns)
    data['workclass'] = data['workclass'].astype('category').cat.codes
    data['education'] = data['education'].astype('category').cat.codes
    data['occupation'] = data['occupation'].astype('category').cat.codes
    data['relationship'] = data['relationship'].astype('category').cat.codes
    data['marital-status'] = data['marital-status'].astype('category').cat.codes
    data['income'] = data['income'].astype('category').cat.codes
    data['gender'] = data['sex'].astype('category').cat.codes
    # data['srace'] = data['race'].astype('category').cat.codes
    p_data = data[['education-num',
                   'hours-per-week',
                   'relationship', 'occupation',
                   'capital-gain', 'capital-loss',
                   'marital-status', 'workclass', 'education', 'race', 'income', 'sex']]
    cl_data = p_data[(p_data['race'] == ' White') | (p_data['race'] == ' Black')]
    for var in list(cl_data.columns):
        cl_data = cl_data[~cl_data[var].isnull()]
    outcome = cl_data['income']
    sensitive_sex = pd.get_dummies(cl_data['sex'])
    sensitive_race = pd.get_dummies(cl_data['race'])
    cl_data['race_gender'] = cl_data['race'] + '_' + cl_data['sex']
    sensitive_mul = pd.get_dummies(cl_data['race_gender'])
    cl_data.drop(['sex', 'income', 'race', 'race_gender'], axis=1, inplace=True)
    cl_data = (cl_data - cl_data.mean()) / cl_data.std()
    return {
        'x': cl_data,
        'y': outcome,
        'sensitive_race': sensitive_race,
        'sensitive_sex': sensitive_sex,
        'sensitive_merge': sensitive_mul
    }


if __name__ == '__main__':
    res = adult('adult.data')
    np.savez('../adult_train', **res)
    res = adult('adult.test')
    np.savez('../adult_test', **res)
