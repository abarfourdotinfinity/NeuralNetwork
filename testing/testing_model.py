import joblib
import torch
from torch.utils.data import DataLoader

def test_model(A, device, model, hparams, config):
    if(config['model'] == 'neural_network'):
        # ===== Load Test Data =====
        _, X_test, _, y_test = A
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        # ===== Evaluate the model =====
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to track gradients during evaluation
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            correct = (predicted == y_test).sum().item()
            total = y_test.size(0)

        accuracy = correct / total * 100
        return accuracy
    
    elif(config['model'] == 'lstm'):

        results = torch.zeros(hparams['time_step'], 1).to(device)
        losses = []
        df = A
        
        train_split=int(len(df)*(1-hparams['test_size']))
        test_split = len(df) - train_split
        _train_dataloader = model.train_dataloader(df, time_step=hparams['time_step'], start_index=train_split, population=test_split, device=device)
        dataloader = DataLoader(_train_dataloader, batch_size=hparams['batch_size'], shuffle=False)

        loss_fn = torch.nn.MSELoss()

        model.eval()
        with torch.no_grad():  # No need to compute gradients
            for data, ans in dataloader:
                data = data.to(device)
                ans = ans.to(device)
                res = model(data)
                dimensions = results.shape
                results = torch.cat((results, res), dim=0)
                losses.append(loss_fn(res, ans))
                
        accuracy = 1 - sum(losses) / len(losses)
        return accuracy * 100
