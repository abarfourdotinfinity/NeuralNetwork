import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(test_size, config):
    if(config['data_type'] == 'url'):
        url = config['data_path']
        df = pd.read_csv(url)
        df['variety'] = df['variety'].replace({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

        X = df.drop('variety', axis=1).values
        y = df['variety'].values

        A = X, y
        A_train, A_test =  split_data(A, test_size)

        # Convert to torch tensors
        X_train = torch.FloatTensor(A_train[0])
        X_test = torch.FloatTensor(A_test[0])
        y_train = torch.LongTensor(A_train[1])
        y_test = torch.LongTensor(A_test[1])

        return X_train, X_test, y_train, y_test
    
    elif(config['data_type'] == 'csv'):
        # Load the data
        df = pd.read_csv(config['data_path'])
        df = df[config['columns']]
        df_y = scaler(df)

        return df_y
    

def split_data(A, test_size=0.2):
    if isinstance(A, (tuple, list)):
        # Unpack all elements using train_test_split
        splits = train_test_split(*A, test_size=test_size)
        # Even indices = train, odd indices = test
        A_train = tuple(splits[i] for i in range(0, len(splits), 2))
        A_test = tuple(splits[i] for i in range(1, len(splits), 2))
        return A_train, A_test
    else:
        # Just split A if it's a single array (e.g., only features)
        return train_test_split(A, test_size=test_size)
    
def scaler(df):
    df = df.to_numpy()
    df = df.reshape(-1,1)
    scaler = StandardScaler()
    df  = pd.DataFrame(scaler.fit_transform(df))
    
    # Save the scaler to a file so it can be loaded later
    scaler_file = 'saved_scalers/scaler.pkl'  # You can adjust this path as needed
    joblib.dump(scaler, scaler_file)  # Save the scaler
    
    return df
    
