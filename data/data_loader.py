import torch
import pandas as pd

def load_data():
    url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    df = pd.read_csv(url)
    df['variety'] = df['variety'].replace({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

    X = df.drop('variety', axis=1).values
    y = df['variety'].values

    return torch.FloatTensor(X), torch.LongTensor(y)