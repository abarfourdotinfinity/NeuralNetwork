# NeuralNetwork

This repository contains a neural network implementation in Python. The project is structured into several directories and files, each serving a specific purpose in the development and execution of the neural network model.

## Project Structure

```
NeuralNetwork/
│── data/             # Contains datasets used for training and testing the neural network
│── models/           # Stores trained models
│── saved_models/     # Contains saved versions of models for future use or analysis
│── saved_scalers/    # Holds saved scaler objects used for data normalization or standardization
│── testing/          # Includes scripts and resources for testing the neural network's performance
│── training/         # Contains scripts and resources for training the neural network
│── utils/            # Provides utility functions that support various operations within the project
│── .env              # Environment configuration file
│── .gitignore        # Specifies files and directories to be ignored by Git
│── config.yaml       # Configuration file containing settings and parameters for the project
│── hparams.yaml      # Hyperparameter configuration file
│── main.py           # The main script that initializes and runs the neural network
```

## Getting Started

To get started with this project, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/abarfourdotinfinity/NeuralNetwork.git
```

### 2. Navigate to the project directory

```bash
cd NeuralNetwork
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the main script

```bash
python main.py
```

## Configuration

- **config.yaml**: Contains general configuration settings for the project.
- **hparams.yaml**: Specifies hyperparameters for the neural network, such as learning rate, batch size, and number of epochs.

Modify these files to adjust the behavior and performance of the neural network according to your requirements.

## License

This project is licensed under the [FDI License](LICENSE).
