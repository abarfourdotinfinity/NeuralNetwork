from torch.utils.data import DataLoader

def training_loop(model, A, optimizer, criterion, hparams, config, device):
    
    losses = []

    if(config['model'] == 'neural_network'):
        X_train, _, y_train, _ = A
        X = X_train.to(device)
        y = y_train.to(device)

        for epoch in range(hparams['epochs']):
            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    elif(config['model'] == 'lstm'):
        df_y = A
        population=int(len(df_y)*(1-hparams['test_size']))
        _train_dataloader = model.train_dataloader(df_y, time_step=hparams['time_step'], start_index=0, population=population, device=device)
        train_data = DataLoader(_train_dataloader, batch_size=hparams['batch_size'], shuffle=True)

        for epoch in range(hparams['epochs']):
            accumulative_loss = 0
            for i, data in enumerate(train_data, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                accumulative_loss += loss.item()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

            losses.append(accumulative_loss / len(train_data))

    return losses