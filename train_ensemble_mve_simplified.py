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
N = 2000 # number of examples in training dataset
learning_rate = 0.0001 # learning rate when initializing the optimizer
num_epochs = 5 # number of training epochs
batch_size = 200 # minibatch size during training 
n_models = 2 # number of models in ensemble

# Data
X_train = np.load('./Training/X_train_simplified.npy')
X_train_numpy = X_train[0:N].reshape([N,1])
X_train = torch.tensor(X_train_numpy,dtype=float)
X_val = np.load('./Validation/X_valid_simplified.npy')
X_val_numpy = X_val[:1000].reshape([1000,1])
X_val = torch.tensor(X_val_numpy,dtype=float)
d_alt = X_train.shape[1]
print('d_x =', d_alt)

Y_train = np.load('./Training/Y_train_simplified.npy')
Y_train = torch.tensor(np.log(Y_train[0:N]),dtype=float)
Y_val = np.load('./Validation/Y_valid_simplified.npy')
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
    
class ConditionalMVE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConditionalMVE, self).__init__()
        
        self.output_dim = output_dim

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            #nn.Dropout(p=0.1)
        )

        # Branched heads for mean and log variance
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        features = self.shared_net(x)
        mean = self.mean_head(features)
        log_var = self.logvar_head(features)
        return mean, log_var

    def loss(self, mean, log_var, y_true):
        var = torch.exp(log_var)
        nll = 0.5 * (log_var + ((y_true - mean) ** 2) / var)
        return nll.mean()
    
    def loss0(self, mean, y_true):
        nll = (y_true - mean) ** 2
        return nll.mean()    
    


for i in range(n_models):
    
    print("#### TRAINING OF MODEL "+str(i+1)+" ####")
    folder_exp = './Saved_models/'
    os.makedirs(folder_exp, exist_ok=True)
    set_seed(i)
    model = ConditionalMVE(input_dim=d_alt, hidden_dim=256, output_dim=d)
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
            mean, logvar = model(x)
            if epoch < -200:
                loss = model.loss0(mean, y)
            else:
                loss = model.loss(mean, logvar, y)
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
            mean, logvar = model(x)
            if epoch < 200:
                vloss = model.loss0(mean, y)
            else:
                vloss = model.loss(mean, logvar, y)
            validation_loss.append(float(vloss))
        print("Epoch " + str(epoch+1) + " with model " + str(i+1))
        print("Training loss =", float(total_loss/len(train_loader)))
        print("Validation loss =", float(vloss))
        epoch += 1
    
    torch.save(model.state_dict(), folder_exp+'mve_simplified_'+str(i)) # Save trained model

