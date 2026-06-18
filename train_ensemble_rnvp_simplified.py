# IMPORT LIBRARIES

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import grad
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
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
    
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------- Real NVP -------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------


# ---------------------------------
#  Residual MLP (shared building block)
# ---------------------------------
class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_blocks, out_dim, dropout=0.0):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_blocks)
            ]
        )
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.input(x)
        for block in self.blocks:
            h = h + block(h)  # residual connection
        return self.output(h)


# ---------------------------------
#  Conditional Gaussian base p0(y0 | x)
# ---------------------------------
class ConditionalGaussianBase(nn.Module):
    """
    Learns a diagonal Gaussian base distribution p0(y0 | x)
    with mean mu(x) and log-std log_sigma(x).

    y0 | x ~ N(mu(x), diag(exp(2*log_sigma(x)))).
    """

    def __init__(self, Dx, Dy, hidden_dim=128, num_blocks=2, log_sigma_clip=5.0):
        super().__init__()
        self.Dx = Dx
        self.Dy = Dy
        self.log_sigma_clip = log_sigma_clip

        self.net = ResidualMLP(
            in_dim=Dx,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            out_dim=2 * Dy,  # mu and log_sigma
        )

        # Initialize to standard normal: mu=0, log_sigma=0 at start
        with torch.no_grad():
            self.net.output.weight.zero_()
            self.net.output.bias.zero_()

    def forward(self, x):
        h = self.net(x)  # (B, 2*Dy)
        h = h.view(h.shape[0], self.Dy, 2)  # (B, Dy, 2)
        mu = h[..., 0]
        log_sigma = h[..., 1]

        # Optionally clip log_sigma for stability
        log_sigma = torch.clamp(
            log_sigma, min=-self.log_sigma_clip, max=self.log_sigma_clip
        )
        return mu, log_sigma

    def log_prob(self, y0, x):
        mu, log_sigma = self(x)  # (B, Dy), (B, Dy)
        var = torch.exp(2.0 * log_sigma)

        # log N(y0 | mu, var I) = -0.5 * [ ((y0-mu)^2 / var) + 2*log_sigma + log(2*pi) ]
        log_2pi = math.log(2.0 * math.pi)
        norm_term = ((y0 - mu) ** 2) / var + 2.0 * log_sigma + log_2pi
        return -0.5 * norm_term.sum(dim=-1)  # (B,)

    def sample(self, x):
        mu, log_sigma = self(x)
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sigma) * eps


# ---------------------------------
#  Conditional RealNVP coupling layer (unchanged)
# ---------------------------------
class ConditionalAffineCoupling(nn.Module):
    """
    RealNVP-style affine coupling layer, conditional on x.

    - Split y into (y_a, y_b)
    - Predict scale s and translation t for y_b using a conditioner network on [y_a, x]
    - Forward:
        y_b' = y_b * exp(log_s) + t
        log|det J| = sum(log_s)
    - Inverse:
        y_b = (y_b' - t) * exp(-log_s)
        log|det J^{-1}| = -sum(log_s)
    """

    def __init__(
        self,
        Dy,
        Dx,
        hidden_dim=128,
        num_blocks=3,
        scale_clip=2.0,
    ):
        super().__init__()
        self.Dy = Dy
        self.Dx = Dx
        self.scale_clip = scale_clip

        # Split structure
        self.Dy_a = Dy // 2
        self.Dy_b = Dy - self.Dy_a

        # Conditioner predicts [t, log_s] for y_b
        self.conditioner = ResidualMLP(
            in_dim=self.Dy_a + Dx,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            out_dim=self.Dy_b * 2,  # t and log_s
        )

        # Initialize conditioner to identity transform: t=0, log_s=0 at start
        with torch.no_grad():
            self.conditioner.output.weight.zero_()
            self.conditioner.output.bias.zero_()

        # Fixed permutation for features in this layer
        perm = torch.randperm(Dy)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(Dy)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def _permute(self, y):
        return y[..., self.perm]

    def _inv_permute(self, y):
        return y[..., self.inv_perm]

    def _get_t_log_s(self, cond_input):
        h = self.conditioner(cond_input)         # (B, 2*Dy_b)
        h = h.view(h.shape[0], self.Dy_b, 2)     # (B, Dy_b, 2)
        t_raw = h[..., 0]
        log_s_raw = h[..., 1]

        # Keep changes small and stable at the beginning
        t = 0.1 * t_raw
        log_s = 0.1 * torch.tanh(log_s_raw)      # in (-0.1, 0.1)
        log_s = torch.clamp(log_s, -self.scale_clip, self.scale_clip)
        return t, log_s

    def forward(self, y, x):
        y = self._inv_permute(y)

        y_a = y[..., :self.Dy_a]
        y_b = y[..., self.Dy_a:]

        cond_input = torch.cat([y_a, x], dim=-1)  # (B, Dy_a + Dx)
        t, log_s = self._get_t_log_s(cond_input)  # each (B, Dy_b)

        y_b_out = y_b * torch.exp(log_s) + t
        logabsdet = log_s.sum(dim=-1)             # (B,)

        y_out = torch.cat([y_a, y_b_out], dim=-1)
        y_out = self._permute(y_out)
        return y_out, logabsdet

    def inverse(self, y, x):
        y = self._inv_permute(y)

        y_a = y[..., :self.Dy_a]
        y_b = y[..., self.Dy_a:]

        cond_input = torch.cat([y_a, x], dim=-1)
        t, log_s = self._get_t_log_s(cond_input)

        y_b_in = (y_b - t) * torch.exp(-log_s)
        logabsdet = -log_s.sum(dim=-1)

        y_in = torch.cat([y_a, y_b_in], dim=-1)
        y_in = self._permute(y_in)
        return y_in, logabsdet


# ---------------------------------
#  Full conditional RealNVP with learned base
# ---------------------------------
class ConditionalRealNVP(nn.Module):
    def __init__(
        self,
        Dy,
        Dx,
        num_layers=6,
        hidden_dim=128,
        num_blocks=2,
        scale_clip=2.0,
        base_hidden_dim=None,
        base_num_blocks=2,
        base_log_sigma_clip=5.0,
    ):
        """
        Dy: dimensionality of y
        Dx: dimensionality of x

        num_layers:      number of RealNVP coupling layers
        hidden_dim:      hidden dimension for each coupling conditioner
        num_blocks:      residual blocks per coupling conditioner
        scale_clip:      clipping for log_s in affine couplings

        base_hidden_dim: hidden dimension for base network (if None, use hidden_dim)
        base_num_blocks: residual blocks in base network
        base_log_sigma_clip: clipping for base log_sigma
        """
        super().__init__()
        self.Dy = Dy
        self.Dx = Dx

        if base_hidden_dim is None:
            base_hidden_dim = hidden_dim

        # Learned conditional base p0(y0 | x)
        self.base = ConditionalGaussianBase(
            Dx=Dx,
            Dy=Dy,
            hidden_dim=base_hidden_dim,
            num_blocks=base_num_blocks,
            log_sigma_clip=base_log_sigma_clip,
        )

        # RealNVP coupling layers
        self.layers = nn.ModuleList(
            [
                ConditionalAffineCoupling(
                    Dy=Dy,
                    Dx=Dx,
                    hidden_dim=hidden_dim,
                    num_blocks=num_blocks,
                    scale_clip=scale_clip,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, y0, x):
        logabsdet_total = torch.zeros(y0.shape[0], device=y0.device)
        y = y0
        for layer in self.layers:
            y, lad = layer(y, x)
            logabsdet_total += lad
        return y, logabsdet_total

    def inverse(self, y, x):
        logabsdet_total = torch.zeros(y.shape[0], device=y.device)
        y_curr = y
        for layer in reversed(self.layers):
            y_curr, lad = layer.inverse(y_curr, x)
            logabsdet_total += lad
        return y_curr, logabsdet_total

    def log_prob(self, y, x):
        y0, logabsdet = self.inverse(y, x)   # y0 = f^{-1}(y; x)
        log_p0 = self.base.log_prob(y0, x)   # conditional Gaussian log-density
        return log_p0 + logabsdet

    def sample(self, x, n_samples=None):
        if n_samples is not None:
            B, Dx = x.shape
            x = x.unsqueeze(1).expand(B, n_samples, Dx).reshape(-1, Dx)

        # Sample from conditional base
        y0 = self.base.sample(x)  # (B, Dy)
        y, _ = self.forward(y0, x)

        if n_samples is not None:
            y = y.view(-1, n_samples, self.Dy)
        return y


for i in range(n_models):
    
    print("#### TRAINING OF MODEL "+str(i+1)+" ####")
    folder_exp = './Saved_models/'
    os.makedirs(folder_exp, exist_ok=True)
    set_seed(i)
    model = ConditionalRealNVP(d, d_alt, num_layers=2,
            hidden_dim=96,
            num_blocks=3,
            scale_clip=10.0,
            base_hidden_dim=96,
            base_num_blocks=3,
            base_log_sigma_clip=10.0)
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

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            #x, y = x.unsqueeze(1), y.unsqueeze(1)
            # Forward pass through the model
            log_prob = model.log_prob(y, x)
            loss = -log_prob.mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            total_loss += loss.item()
        training_loss.append(float(total_loss/len(train_loader)))
        
        model.eval()
        with torch.no_grad():
            x, y = next(iter(valid_loader))
            x, y = x.to(device), y.to(device)
            #x, y = x.unsqueeze(1), y.unsqueeze(1)
            log_prob = model.log_prob(y, x)
            vloss = -log_prob.mean() # Calculate validation loss
            validation_loss.append(float(vloss))

        print("Epoch " + str(epoch+1) + " with model " + str(i+1))
        print("Training loss =", float(total_loss/len(train_loader)))
        print("Validation loss =", float(vloss))
    
    torch.save(model.state_dict(), folder_exp+'rnvp_simplified_'+str(i)) # Save trained model

