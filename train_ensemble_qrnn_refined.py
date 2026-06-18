# IMPORT LIBRARIES

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import grad
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# LOAD DATA, DEFINE HYPERPARAMETERS

# Hyperparameters
alpha = 0.05
quantiles = [alpha/2, 0.5, 1 - alpha/2]
N = 2000 # number of examples in training dataset
learning_rate = 0.0001 # learning rate when initializing the optimizer
num_epochs = 5 # number of training epochs
batch_size = 200 # minibatch size during training 
n_models = 2 # number of models in ensemble

# Data
X_train = np.load('./Training/X_train_refined.npy')
X_train_numpy = X_train[0:N].reshape([N,1])
X_train = torch.tensor(X_train_numpy,dtype=float)
X_val = np.load('./Validation/X_valid_refined.npy')
X_val_numpy = X_val[:1000].reshape([1000,1])
X_val = torch.tensor(X_val_numpy,dtype=float)
d_alt = X_train.shape[1]
print('d_x =', d_alt)

Y_train = np.load('./Training/Y_train_refined.npy')
Y_train = torch.tensor(np.log(Y_train[0:N]),dtype=float)
Y_val = np.load('./Validation/Y_valid_refined.npy')
Y_val = torch.tensor(np.log(Y_val[:1000]),dtype=float)

mean_y = Y_train.mean()
sigma_y = Y_train.std()
Y_train = (Y_train-mean_y)
Y_val = (Y_val-mean_y)
d = Y_train.shape[1]
print('d_y =', d)

class AltitudePressureDataset(Dataset):
    def __init__(self, altitude_maps, pressure_maps, transform=None):
        self.altitude_maps = altitude_maps
        self.pressure_maps = pressure_maps
        self.transform = transform

    def __len__(self):
        return len(self.altitude_maps)

    def __getitem__(self, idx):
        altitude = self.altitude_maps[idx]
        pressure = self.pressure_maps[idx]
        
        if self.transform:
            altitude = self.transform(altitude)
            pressure = self.transform(pressure)
        
        return altitude, pressure

altitude_data = X_train
pressure_data = Y_train
dataset = AltitudePressureDataset(altitude_data, pressure_data) # Training dataset

altitude_data2 = X_val
pressure_data2 = Y_val
valid_dataset = AltitudePressureDataset(altitude_data2, pressure_data2) # Validation dataset


# DEFINE MODEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Hopefully runs on GPU
    
class ConditionalQR(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, quantiles):
        super(ConditionalQR, self).__init__()
        
        self.output_dim = output_dim
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(hidden_dim, output_dim * self.n_quantiles)

    def forward(self, x):
        features = self.shared_net(x)
        out = self.head(features)
        out = out.view(-1, self.output_dim, self.n_quantiles)
        return out  # shape: (batch, d, n_quantiles)

    def pinball_loss(self, preds, y_true):
        loss = 0.0
        for i, tau in enumerate(self.quantiles):
            errors = y_true - preds[:, :, i]
            loss += torch.max(tau * errors, (tau - 1) * errors).mean()
        return loss / self.n_quantiles   
    


for i in range(n_models):
    
    print("#### TRAINING OF MODEL "+str(i+1)+" ####")
    folder_exp = './Saved_models/'
    os.makedirs(folder_exp, exist_ok=True)
    set_seed(i)
    model = ConditionalQR(input_dim=d_alt, hidden_dim=184, output_dim=d, quantiles=quantiles)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The model is made of " + str(n_parameters) + " parameters.")
    model = model.double()
    model.to(device)



    # TRAINING PHASE

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Create dataloader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1000, shuffle=True)

    # Training loop
    model.train()
    training_loss = []
    validation_loss = []
    
    epoch = 0
    while epoch < num_epochs:
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)  # (batch, d, 2)
            loss = model.pinball_loss(preds, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        training_loss.append(float(total_loss/len(train_loader)))
        model.eval()
        with torch.no_grad():
            x, y = next(iter(valid_loader))
            x, y = x.to(device), y.to(device)
            preds = model(x)  # (batch, d, 2)
            vloss = model.pinball_loss(preds, y)
            validation_loss.append(float(vloss))
        print("Epoch " + str(epoch+1) + " with model " + str(i+1))
        print("Training loss =", float(total_loss/len(train_loader)))
        print("Validation loss =", float(vloss))
        epoch += 1

    torch.save(model.state_dict(), folder_exp+'qrnn_refined_'+str(i)) # Save trained model

