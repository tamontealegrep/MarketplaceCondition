
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from timeit import default_timer as timer
from IPython.display import clear_output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------------------------------------------------------


class PuntualDataset(Dataset):
    """
    Custom dataset class for handling tabular data.

    This class encapsulates the functionality required to prepare tabular data for training and inference
    in PyTorch models.

    Args:
        X (array-like or DataFrame): Features data.
        y (array-like or Series): Target data.

    Attributes:
        X (Tensor): Features tensor converted to torch.float32.
        y (Tensor): Target tensor converted to torch.float32.
    """

    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = torch.tensor(X.values, dtype=torch.float32)
        
        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = torch.tensor(y.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class ModelFC(nn.Module):
    """
    Fully connected neural network model for classification.

    Args:
        input_size (int): Number of features.
        output_size (int): Number of categories.
        fc_hidden_sizes (list, optional): Number of hidden sizes per fully connected (fc) layer. Default is [1].
        fc_dropout (float, optional): Dropout probability after each fc layer. Default is 0.2.

    Attributes:
        name (str): Model name.
        input_size (int): Number of features.
        output_size (int): Number of categories.
        fc_hidden_sizes (list): Number of hidden sizes per fc layer.
        fc_dropout (float): Dropout probability after each fc layer.
        activation (torch.nn.Module): Activation function (Sigmoid).
        layers (torch.nn.Sequential): Sequential container for fc layers.

    Methods:
        forward(x): Forward pass through the model.
        train_validation(train_loader, val_loader, criterion, optimizer, num_epochs=10): Train and validate the model.
        train_model(train_loader, criterion, optimizer, num_epochs=10): Train the model.
        _train_step(train_loader, criterion, optimizer): Perform a single training step.
        _validation_step(val_loader, criterion): Perform a single validation step.
        plot_training(save_path=""): Plot training curves.

    """
    def __init__(self,
                 input_size, # Number of features
                 output_size, # Number of categories
                 fc_hidden_sizes=[1], # Number of hidden sizes per fc layer
                 fc_dropout=0.2, # Dropout probability after each fc layer
                 ):
        super(ModelFC, self).__init__()
        self.name = 'FC'
        self.device = DEVICE
        self.input_size = input_size
        self.output_size = output_size
        self.fc_hidden_sizes = fc_hidden_sizes
        self.fc_dropout = fc_dropout
        self.activation = nn.Sigmoid()

        layers = []

        fc_input_size = self.input_size
        for fc_hidden_size in self.fc_hidden_sizes:
            if fc_hidden_size <= 0:
                raise ValueError("hidden_size must be greater than 0")
            layers.append(nn.Linear(fc_input_size, fc_hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.fc_dropout))
            fc_input_size = fc_hidden_size


        layers.append(nn.Linear(fc_input_size, self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = self.activation(x)

        return x
        
    def train_validation(self, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        self.to(self.device)

        results = {"train_loss": [],"val_loss": []}

        if not hasattr(self, "results"):
            setattr(self, "results", results)

        for epoch in range(num_epochs):
            start_time = timer() 
            train_loss = self._train_step(train_loader, criterion, optimizer)
            val_loss = self._validation_step(val_loader, criterion)
            end_time = timer()
            
            self.results["train_loss"].append(train_loss)
            self.results["val_loss"].append(val_loss)

            clear_output(wait=True)
            print(f'| Epoch [{epoch+1}/{num_epochs}] | Time: {end_time-start_time:.1f} |\n'
                  f'| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} |\n')

            
            self.plot_training()

    def train_model(self, train_loader, criterion, optimizer, num_epochs=10):
        self.to(self.device)

        results = {"train_loss": [], "val_loss": []} 

        if not hasattr(self, "results"):
            setattr(self, "results", results)

        for _ in range(num_epochs):

            train_loss = self._train_step(train_loader, criterion, optimizer)
            val_loss = None

            self.results["train_loss"].append(train_loss)
            self.results["val_loss"].append(val_loss)



    def _train_step(self, train_loader, criterion, optimizer):
        self.train()  # Set the model to training mode

        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self(inputs)
            if outputs.size() != labels.size():
                outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)

        return epoch_loss

    def _validation_step(self, val_loader, criterion):
        self.eval()  # Set the model to evaluation mode

        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

        epoch_loss = running_loss / len(val_loader.dataset)

        return epoch_loss
    

    
    def plot_training(self, save_path: str = ""):
        """
        Plots training curves of a results dictionary.

        Args:
            results (dict): dictionary containing list of values, e.g.
                {"train_loss": [...],
                "val_loss": [...]}
            save_path (str): path to save the plot as PNG
        """
        results = self.results

        loss = results['train_loss']
        val_loss = results['val_loss']

        epochs = range(len(results['train_loss']))

        plt.figure(figsize=(5, 5))
        plt.plot(epochs, loss, label='train_loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

#---------------------------------------------------------------------------------------------------
            
class PyTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, criterion=nn.BCELoss(), optimizer=None, learning_rate=0.001, weight_decay=0.001, num_epochs=10):
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_class = optimizer
        self.num_epochs = num_epochs
        self.threshold = 0.5
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = optimizer
            
    def fit(self, X, y):
        dataset = PuntualDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train_model(train_loader, self.criterion, self.optimizer, self.num_epochs)
        return self

    def predict(self, X):
        device = self.model.device
        dataset = PuntualDataset(X, pd.Series([0] * len(X)))  # Dummy labels, not used during prediction
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        predictions = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().numpy())

        return (np.array(predictions) > self.threshold).astype(int)  # Binary predictions based on threshold 0.5

    def predict_proba(self, X):
        device = self.model.device
        dataset = PuntualDataset(X, pd.Series([0] * len(X)))  # Dummy labels, not used during prediction
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        probabilities = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                probabilities.extend(outputs.cpu().numpy())

        probabilities = np.array(probabilities)
        return np.hstack((1 - probabilities, probabilities))
    
#---------------------------------------------------------------------------------------------------