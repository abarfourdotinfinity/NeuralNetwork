import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(test_size):
    url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    df = pd.read_csv(url)
    df['variety'] = df['variety'].replace({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

    X = df.drop('variety', axis=1).values
    y = df['variety'].values

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Return the split data
    return X_train, X_test, y_train, y_test