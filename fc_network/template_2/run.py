'''Load and filter data based on threshold eigenvalue.
Create simple datasets and dataloaders.
Create basic MLP model that uses spectral composition
but predict sin and cos of angles instead of angles
by using tanh instead of sigmoid.
MSE loss and AdamW optimizer.
Option to resume training.
Epoch loop with basic train and validation steps.
Save best checkpoint and plot losses.'''


import os
import time
import random
import logging
from functools import wraps
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Callable, TypeVar

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


@dataclass
class Hyperparameters:
    
    seed: int
    
    hidden_sizes: list[int]
    output_size: int
    threshold_eigval: float
    
    lr: float
    num_epochs: int
    batch_size: int
    train_percent: float

    simulation_data_experiment_path: str


@dataclass
class Paths:
    experiments_dir = os.path.join('fc_network', 'template_2', 'experiments')
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    simulation_data_experiment_path = ''

    @property
    def experiment_path(self):
        return os.path.join(self.experiments_dir, self.experiment_name)
    
    @property
    def checkpoint_file(self): 
        return os.path.join(self.experiment_path, 'checkpoint.pt')
    
    @property
    def log_file(self):
        return os.path.join(self.experiment_path, 'log.txt')
    
    @property
    def losses_plot_file(self):
        return os.path.join(self.experiment_path, 'losses.png')
    
    @property
    def d_tensors_file(self):
        return os.path.join(self.simulation_data_experiment_path, 'd_tensors.pt')
    
    @property
    def noisy_signals_file(self):
        return os.path.join(self.simulation_data_experiment_path, 'noisy_signals.pt')


class DiffusionDataset(Dataset):
    
    def __init__(self, d_tensors: torch.Tensor, noisy_signals: torch.Tensor):
        super().__init__()
        self.d_tensors = d_tensors
        self.noisy_signals = noisy_signals

    def __len__(self):
        return self.d_tensors.shape[0]

    def __getitem__(self, idx):
        return self.d_tensors[idx], self.noisy_signals[idx]


class SpectralDiffusionNet(torch.nn.Module):
    
    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            threshold_eigval: float
        ):
        super().__init__()

        self.threshold_eigval = threshold_eigval

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(torch.nn.ReLU())

        self.net = torch.nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        output = self.net(input)
        
        # Split the tensor into individual elements
        x_sins, x_cosines, y_sins, y_cosines, z_sins, z_cosines, eig_val_1, eig_val_2_over_1, eig_val_3_over_2 = torch.split(output, split_size_or_sections=1, dim=1)

        # Activate the eigenvalues
        eig_val_1 = torch.sigmoid(eig_val_1).squeeze()
        eig_val_2_over_1 = torch.sigmoid(eig_val_2_over_1).squeeze()
        eig_val_3_over_2 = torch.sigmoid(eig_val_3_over_2).squeeze()

        # Activate sins and cosines
        x_sins = torch.tanh(x_sins).squeeze()
        x_cosines = torch.tanh(x_cosines).squeeze()
        y_sins = torch.tanh(y_sins).squeeze()
        y_cosines = torch.tanh(y_cosines).squeeze()
        z_sins = torch.tanh(z_sins).squeeze()
        z_cosines = torch.tanh(z_cosines).squeeze()
        
        # Create the roation matrices around the x axis.
        R_x = torch.zeros((x_sins.shape[0], 3, 3))
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = x_cosines
        R_x[:, 1, 2] = -x_sins
        R_x[:, 2, 1] = x_sins
        R_x[:, 2, 2] = x_cosines

        # Create the roation matrices around the y axis.
        R_y = torch.zeros((y_sins.shape[0], 3, 3))
        R_y[:, 0, 0] = y_cosines
        R_y[:, 0, 2] = y_sins
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -y_sins
        R_y[:, 2, 2] = y_cosines

        # Create the roation matrices around the z axis.
        R_z = torch.zeros((z_sins.shape[0], 3, 3))
        R_z[:, 0, 0] = z_cosines
        R_z[:, 0, 1] = -z_sins
        R_z[:, 1, 0] = z_sins
        R_z[:, 1, 1] = z_cosines
        R_z[:, 2, 2] = 1

        # Calculate the rotation matrices.
        R = torch.bmm(R_z, torch.bmm(R_y, R_x))

        # Compute eigenvalues
        eig_val_1 = eig_val_1 * self.threshold_eigval
        eig_val_2 = eig_val_1 * eig_val_2_over_1
        eig_val_3 = eig_val_2 * eig_val_3_over_2

        # Calculate the diagonal matrix of eigenvalues.
        eig_vals = torch.zeros((eig_val_1.shape[0], 3, 3))
        eig_vals[:, 0, 0] = eig_val_1
        eig_vals[:, 1, 1] = eig_val_2
        eig_vals[:, 2, 2] = eig_val_3

        # Reconstruct the diffusion tensors.
        D = torch.bmm(R, torch.bmm(eig_vals, R.transpose(1, 2)))

        return D


T = TypeVar('T')

def timer(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> tuple[T, float]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


@timer
def train(train_loader, model, loss_fn, optimizer, device) -> float:

    epoch_loss = 0.0
    model.train()
    
    for batch_d_tensors, batch_noisy_signals in train_loader:
        
        batch_d_tensors = batch_d_tensors.to(device)
        batch_noisy_signals = batch_noisy_signals.to(device)

        pred_d_tensors = model(batch_noisy_signals)
        
        batch_loss = loss_fn(pred_d_tensors, batch_d_tensors)
    
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        
    epoch_loss /= len(train_loader)

    return epoch_loss


@timer
def validate(val_loader, model, loss_fn, device) -> float:

    epoch_loss = 0.0
    model.eval()
    
    with torch.no_grad():
        
        for batch_d_tensors, batch_noisy_signals in val_loader:

            batch_d_tensors = batch_d_tensors.to(device)
            batch_noisy_signals = batch_noisy_signals.to(device)

            pred_d_tensors = model(batch_noisy_signals)

            batch_loss = loss_fn(pred_d_tensors, batch_d_tensors)

            epoch_loss += batch_loss.item()
        
    epoch_loss /= len(val_loader)

    return epoch_loss


def main() -> None:

    
    paths = Paths()
    

    ## HYPERPARAMETERS

    resume_parser = ArgumentParser()    
    resume_parser.add_argument('--resume_experiment_name', type=str, required=False, default=None)
    resume_args, other_args = resume_parser.parse_known_args()

    if resume_args.resume_experiment_name:

        paths.experiment_name = resume_args.resume_experiment_name

        checkpoint = torch.load(paths.checkpoint_file)
        
        HP = Hyperparameters(**checkpoint['hyperparameters'])

        secondary_parser = ArgumentParser()
        secondary_parser.add_argument('--num_epochs', type=int, required=True)
        secondary_parser.add_argument('--lr', type=float, required=True)
        secondary_args = secondary_parser.parse_args(other_args)

        HP.num_epochs = secondary_args.num_epochs
        HP.lr = secondary_args.lr

        logging_messege = 'Resuming training with hyperparameters:'


    else:

        main_parser = ArgumentParser()

        main_parser.add_argument('--seed', type=int, required=True)

        main_parser.add_argument('--hidden_sizes', nargs='+', type=int, required=True)
        main_parser.add_argument('--output_size', type=int, required=True)
        main_parser.add_argument('--threshold_eigval', type=float, required=True)

        main_parser.add_argument('--lr', type=float, required=True)
        main_parser.add_argument('--num_epochs', type=int, required=True)
        main_parser.add_argument('--batch_size', type=int, required=True)
        main_parser.add_argument('--train_percent', type=float, required=True)
        
        main_parser.add_argument('--simulation_data_experiment_path', type=str, required=True)

        main_args = main_parser.parse_args(other_args)

        os.makedirs(paths.experiment_path)
        
        HP = Hyperparameters(**vars(main_args))

        logging_messege = 'Starting training with hyperparameters:'
    

    paths.simulation_data_experiment_path = HP.simulation_data_experiment_path


    ## LOGGING

    logging.basicConfig(
        filename=paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(logging_messege)

    for key, value in vars(HP).items():
        logging.info(f'{key}: {value}')


    ## DEVICE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f'Using {device} device')

    
    ## REPRODUCIBILITY

    # Set seed for Python's random module
    random.seed(HP.seed)

    # Set seed for NumPy's random module
    np.random.seed(HP.seed)

    # Set seed for PyTorch's CPU RNG
    torch.manual_seed(HP.seed)

    # If CUDA is available, set seed for CUDA RNGs and enable deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed(HP.seed)
        torch.cuda.manual_seed_all(HP.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    ## FILTER DATA

    d_tensors = torch.load(paths.d_tensors_file)
    noisy_signals = torch.load(paths.noisy_signals_file)

    eigenvalues = torch.linalg.eigvalsh(d_tensors)
    filter_indices = torch.where(eigenvalues[:,-1] <= HP.threshold_eigval)[0]
    
    d_tensors = d_tensors[filter_indices]
    noisy_signals = noisy_signals[filter_indices]


    ## SHUFFLE AND SPLIT INDICES

    num_data = len(d_tensors)
    num_train = int(HP.train_percent * num_data)

    generator = torch.Generator().manual_seed(HP.seed)
    indices = torch.randperm(num_data, generator=generator)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]


    ## DATASETS AND DATALOADERS

    train_dataset = DiffusionDataset(d_tensors[train_indices], noisy_signals[train_indices])
    val_dataset = DiffusionDataset(d_tensors[val_indices], noisy_signals[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=HP.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=HP.batch_size, shuffle=False, drop_last=True)


    ## INITIALIZE MODEL, LOSS FUNCTION AND OPTIMIZER

    spectral_dnet = SpectralDiffusionNet(
        input_size=noisy_signals.shape[1], 
        hidden_sizes=HP.hidden_sizes,
        output_size=HP.output_size,
        threshold_eigval=HP.threshold_eigval
    ).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(spectral_dnet.parameters(), lr=HP.lr)

    train_losses = []
    val_losses = []
    epochs = []


    ## RESUME TRAINING

    if resume_args.resume_experiment_name:

        spectral_dnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        epochs = checkpoint['epochs']
        
        logging.info(f'Resuming training from epoch {epochs[-1] + 1}')
    

    ## TRAIN MODEL

    last_epoch = epochs[-1] if epochs else 0

    for epoch in tqdm(range(last_epoch, last_epoch + HP.num_epochs)):
            
        epoch_train_loss, train_time = train(train_loader, spectral_dnet, loss_fn, optimizer, device)
        epoch_val_loss, eval_time = validate(val_loader, spectral_dnet, loss_fn, device)

        logging.info(f'Epoch: {epoch + 1}/{last_epoch + HP.num_epochs} - ' + 
                     f'Train Loss: {epoch_train_loss:.8f} - ' +
                     f'Eval Loss: {epoch_val_loss:.8f} - ' +
                     f'Train Time: {train_time:.2f} - ' +
                     f'Eval Time: {eval_time:.2f}')

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        epochs.append(epoch + 1)

        if epoch_val_loss == min(val_losses):
            
            torch.save({
                'hyperparameters': vars(HP),
                'model_state_dict': spectral_dnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs': epochs
            }, paths.checkpoint_file)
            
            logging.info('Checkpoint saved')

    
    # save losses plot
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(paths.losses_plot_file)


    ## FINISHED TRAINING
    logging.info('-' * 50)


if __name__ == '__main__':
    main()
