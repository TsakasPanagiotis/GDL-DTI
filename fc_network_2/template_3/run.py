'''Load simulation b-values and b-vectors.
Load simulation train and validation data
of bi-exponential signals.
Create simple datasets and dataloaders.
Create a fully connected network for
approximating irreps for D and D* tensors,
f-values and S0_correction.
MSE loss and AdamW optimizer.
Option to resume training from a checkpoint.
Epoch loop with basic train and validation steps.
Save best and last checkpoints and losses plot.'''


import os
import time
import random
import pickle
import logging
from functools import wraps
from datetime import datetime
from typing import Protocol, Any
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Callable, TypeVar

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class SimulationDataPaths(Protocol):
    new_b_values_file: str
    new_b_vectors_file: str
    train_sim_data_file: str
    eval_sim_data_file: str


@dataclass
class FCNetworkHyperparameters:
    seed: int
    hidden_sizes: list[int]
    num_epochs: int
    learning_rate: float
    batch_size: int
    alpha: float

    b_values_to_select: list[float]
    simulation_data_paths_pkl: str


class FCNetworkPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('fc_network_2', 'template_3', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')

        self.losses_plot_file = os.path.join(self.experiment_path, 'losses.png')
        self.last_checkpoint_file = os.path.join(self.experiment_path, 'last_checkpoint.pkl')
        self.best_checkpoint_file = os.path.join(self.experiment_path, 'best_checkpoint.pkl')
        self.best_model_file = os.path.join(self.experiment_path, 'best_model.pth')


@dataclass
class Checkpoint:
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    train_losses: list[float]
    val_losses: list[float]
    epochs: list[int]


def get_selection_mask(b_values_to_select_list: list[float], b_values: np.ndarray) -> np.ndarray:

    b_values_to_select = set(b_values_to_select_list)
    unique_b_values = set(b_values)

    if len(b_values_to_select) == 0:
        b_values_to_select = unique_b_values
        logging.warning(f'b_values_to_select is empty. Using all b-values: {b_values_to_select}')
        logging.warning('')
    
    if 0.0 not in b_values_to_select:
        logging.error('b_values_to_select must contain 0.0')
        raise ValueError('b_values_to_select must contain 0.0')

    if not b_values_to_select.issubset(unique_b_values):
        invalid_values = b_values_to_select.difference(unique_b_values)
        invalid_messege = f'Invalid b_values_to_select: {invalid_values}. Valid values are: {unique_b_values}'
        logging.error(invalid_messege)
        raise ValueError(invalid_messege)

    selection_mask = np.isin(b_values, list(b_values_to_select))

    return selection_mask


class DiffusionDataset(Dataset):
    
    def __init__(self, signals: torch.Tensor):
        super().__init__()
        self.signals = signals

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        return self.signals[idx]


class DiffusionNet(torch.nn.Module):
    
    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int
        ):
        super().__init__()

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(torch.nn.ReLU())

        self.net = torch.nn.Sequential(*layers)

        self.basis_irrep_0_2 = self.prepare_basis_irrep_0_2()
    
    def prepare_basis_irrep_0_2(self) -> torch.Tensor:

        basis_irrep_0 = torch.tensor([[[1,0,0],[0,1,0],[0,0,1]]],dtype=torch.float) / np.sqrt(3)

        basis_irrep_2 = torch.tensor([ [[ 0.,  0.,  1.],
                                        [ 0.,  0.,  0.],
                                        [ 1.,  0.,  0.]],

                                       [[ 0.,  1.,  0.],
                                        [ 1.,  0.,  0.],
                                        [ 0.,  0.,  0.]],

                                       [[-1.,  0.,  0.],
                                        [ 0.,  2.,  0.],
                                        [ 0.,  0., -1.]],

                                       [[ 0.,  0.,  0.],
                                        [ 0.,  0.,  1.],
                                        [ 0.,  1.,  0.]],

                                       [[-1.,  0.,  0.],
                                        [ 0.,  0.,  0.],
                                        [ 0.,  0.,  1.]] ])
        
        basis_irrep_2 = basis_irrep_2 / torch.linalg.norm(basis_irrep_2.reshape(basis_irrep_2.shape[0],-1),dim=-1)[:,None,None]

        basis_irrep_0_2 = torch.cat([basis_irrep_0, basis_irrep_2])

        return basis_irrep_0_2

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        output = self.net(input)
        
        d_trace, d_traceless, d_scaling, d_star_trace, d_star_traceless, d_star_scaling, f, S0 = torch.split(output, [1, 5, 1, 1, 5, 1, 1, 1], dim=1)
        
        d_trace = torch.relu(d_trace) + 1e-6 # bounded in (0, inf)
        d_scaling = torch.sigmoid(d_scaling) # bounded in (0, 1)
        d_coeffs = torch.cat([d_trace, d_scaling * d_traceless], dim=1)
        D = torch.einsum('kc,cmn->kmn', d_coeffs, self.basis_irrep_0_2)

        d_star_trace = torch.relu(d_star_trace) + 1e-6 # bounded in (0, inf)
        d_star_scaling = torch.sigmoid(d_star_scaling) # bounded in (0, 1)
        d_star_coeffs = torch.cat([d_star_trace, d_star_scaling * d_star_traceless], dim=1)
        D_star = torch.einsum('kc,cmn->kmn', d_star_coeffs, self.basis_irrep_0_2)

        f = 0.95 * torch.sigmoid(f) + 0.05 # bounded in [0.05, 1]

        S0 = torch.tanh(S0) + 1.0 # bounded in [0, 2]

        return D, D_star, f, S0


def reconstruct(S0, bvals, bvecs, D, D_star, f) -> torch.Tensor:
    '''
    Parameters:
        S0: torch.Tensor of shape (batch_size, 1)
        bvals: torch.Tensor of shape (channels,)
        bvecs: torch.Tensor of shape (channels, 3)
        D: torch.Tensor of shape (batch_size, 3, 3)
        D_star: torch.Tensor of shape (batch_size, 3, 3)
        f: torch.Tensor of shape (batch_size, 1)
    
    Returns:
        reconstructed signals: torch.Tensor of shape (batch_size, channels)
    '''
    
    d_exp = torch.exp(- torch.einsum('c, ci, bij, cj -> bc', bvals, bvecs, D, bvecs))
    d_star_exp = torch.exp(- torch.einsum('c, ci, bij, cj -> bc', bvals, bvecs, D_star, bvecs))
    
    return S0 * (f * d_star_exp + (1 - f) * d_exp)


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
def train(train_loader, model, loss_fn, optimizer, device, bvals, bvecs, alpha):

    model.train()
    epoch_loss = 0.0
    
    for signals in train_loader:
        
        signals = signals.to(device)

        D, D_star, f, S0 = model(signals)
        
        recon_signals = reconstruct(S0, bvals, bvecs, D, D_star, f)

        D_norm = torch.norm(D, dim=(1, 2))
        D_star_norm = torch.norm(D_star, dim=(1, 2))
        reg_term = torch.relu(D_norm - D_star_norm).mean()

        batch_loss = loss_fn(recon_signals, signals) + alpha * reg_term
    
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

    epoch_loss /= len(train_loader)

    return epoch_loss


@timer
def validate(val_loader, model, loss_fn, device, bvals, bvecs):

    model.eval()
    epoch_loss = 0.0
    
    with torch.no_grad():
        
        for signals in val_loader:

            signals = signals.to(device)

            D, D_star, f, S0 = model(signals)

            recon_signals = reconstruct(S0, bvals, bvecs, D, D_star, f)

            batch_loss = loss_fn(recon_signals, signals)

            epoch_loss += batch_loss.item()
        
    epoch_loss /= len(val_loader)

    return epoch_loss


def main():

    ## CHECK FOR RESUMING EXPERIMENT

    resume_parser = ArgumentParser()    
    resume_parser.add_argument('--resume_experiment_paths_pkl', type=str, required=False, default=None)
    resume_args, other_args = resume_parser.parse_known_args()

    
    if resume_args.resume_experiment_paths_pkl: # Resuming experiment

        ## FC NETWORK PATHS
        
        with open(resume_args.resume_experiment_paths_pkl, 'rb') as f:
            fc_network_paths: FCNetworkPaths = pickle.load(f)
        
        
        ## LOGGING

        logging.basicConfig(
            filename=fc_network_paths.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s')
        
        
        ## FC NETWORK HYPERPARAMETERS
        
        with open(fc_network_paths.hyperparameters_file, 'rb') as f:
            fc_network_hparams: FCNetworkHyperparameters = pickle.load(f)

        secondary_parser = ArgumentParser()
        secondary_parser.add_argument('--num_epochs', type=int, required=True)
        secondary_parser.add_argument('--learning_rate', type=float, required=True)
        secondary_args = secondary_parser.parse_args(other_args)

        fc_network_hparams.num_epochs = secondary_args.num_epochs
        fc_network_hparams.learning_rate = secondary_args.learning_rate

        with open(fc_network_paths.hyperparameters_file, 'wb') as f:
            pickle.dump(fc_network_hparams, f)
        
        logging.info('New FC network hyperparameters:')
        logging.info(f'num_epochs: {fc_network_hparams.num_epochs}')
        logging.info(f'learning_rate: {fc_network_hparams.learning_rate}')
        logging.info('')

    
    else: # New experiment

        ## FC NETWORK PATHS

        fc_network_paths = FCNetworkPaths()

        print('Experiment path:', fc_network_paths.experiment_path)

        os.makedirs(fc_network_paths.experiment_path)

        with open(fc_network_paths.paths_file, 'wb') as f:
            pickle.dump(fc_network_paths, f)
        
        
        ## LOGGING

        logging.basicConfig(
            filename=fc_network_paths.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info('FC network experiment:')
        logging.info(fc_network_paths.experiment_path)
        logging.info('')


        ## FC NETWORK HYPERPARAMETERS

        main_parser = ArgumentParser()
        
        main_parser.add_argument('--seed', type=int, default=0)
        main_parser.add_argument('--hidden_sizes', nargs='*', type=int, required=True)
        main_parser.add_argument('--num_epochs', type=int, required=True)
        main_parser.add_argument('--learning_rate', type=float, required=True)
        main_parser.add_argument('--batch_size', type=int, required=True)        
        main_parser.add_argument('--b_values_to_select', type=float, nargs='*', default=[])
        main_parser.add_argument('--simulation_data_paths_pkl', type=str, required=True)
        main_parser.add_argument('--alpha', type=float, required=True)

        main_args = main_parser.parse_args(other_args)

        fc_network_hparams = FCNetworkHyperparameters(**vars(main_args))

        with open(fc_network_paths.hyperparameters_file, 'wb') as f:
            pickle.dump(fc_network_hparams, f)
        
        logging.info('FC network hyperparameters:')
        for key, value in vars(fc_network_hparams).items():
            logging.info(f'{key}: {value}')
        logging.info('')
    

    ## DEVICE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'Using {device} device')
    logging.info('')


    ## REPRODUCIBILITY

    # Set seed for Python's random module
    random.seed(fc_network_hparams.seed)

    # Set seed for NumPy's random module
    np.random.seed(fc_network_hparams.seed)

    # Set seed for PyTorch's CPU RNG
    torch.manual_seed(fc_network_hparams.seed)

    # If CUDA is available, set seed for CUDA RNGs and enable deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fc_network_hparams.seed)
        torch.cuda.manual_seed_all(fc_network_hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    generator = torch.Generator().manual_seed(fc_network_hparams.seed)
    

    ## SIMULATION DATA PATHS

    with open(fc_network_hparams.simulation_data_paths_pkl, 'rb') as f:
        sim_data_paths: SimulationDataPaths = pickle.load(f)
    

    ## NUMPY DATA

    b_values: np.ndarray = np.load(sim_data_paths.new_b_values_file)
    b_vectors: np.ndarray = np.load(sim_data_paths.new_b_vectors_file)
    
    train_data: np.ndarray = np.load(sim_data_paths.train_sim_data_file)
    eval_data: np.ndarray = np.load(sim_data_paths.eval_sim_data_file)

    selection_mask = get_selection_mask(fc_network_hparams.b_values_to_select, b_values)

    
    ## TORCH CONVERSION

    bvals = torch.from_numpy(b_values[selection_mask]).float().to(device)
    bvecs = torch.from_numpy(b_vectors[selection_mask]).float().to(device)
    
    train_signals = torch.from_numpy(train_data[:, selection_mask]).float()
    eval_signals = torch.from_numpy(eval_data[:, selection_mask]).float()


    ## DATASETS AND DATALOADERS

    train_dataset = DiffusionDataset(train_signals)
    val_dataset = DiffusionDataset(eval_signals)

    print('Train dataset:', len(train_dataset))
    print('Validation dataset:', len(val_dataset))

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=fc_network_hparams.batch_size, 
        shuffle=True, 
        generator=generator, 
        drop_last=True)
    
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=fc_network_hparams.batch_size, 
        shuffle=False, 
        drop_last=True)
    

    ## INITIALIZE MODEL, LOSS FUNCTION AND OPTIMIZER

    diffusion_net = DiffusionNet(
        input_size=train_signals.shape[1],
        hidden_sizes=fc_network_hparams.hidden_sizes,
        output_size=1 + 5 + 1 + 1 + 5 + 1 + 1 + 1 
        # D_trace, D_traceless, D_scaling, 
        # D_star_trace, D_star_traceless, D_scaling, 
        # f, S0
    ).to(device)

    logging.info(diffusion_net)
    logging.info('')

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(params=diffusion_net.parameters(), lr=fc_network_hparams.learning_rate)

    train_losses = []
    val_losses = []
    epochs = []


    ## RESUME TRAINING

    if resume_args.resume_experiment_paths_pkl:

        with open(fc_network_paths.last_checkpoint_file, 'rb') as f:
            checkpoint: Checkpoint = pickle.load(f)

        diffusion_net.load_state_dict(checkpoint.model_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        train_losses = checkpoint.train_losses
        val_losses = checkpoint.val_losses
        epochs = checkpoint.epochs

        logging.info(f'Resuming training from epoch {len(epochs)}')
        logging.info('')
    

    ## TRAIN MODEL

    last_epoch = epochs[-1] + 1 if epochs else 1

    for epoch in tqdm(range(last_epoch, last_epoch + fc_network_hparams.num_epochs)):

        train_epoch_loss, train_time = train(train_loader, diffusion_net, loss_fn, optimizer, device, bvals, bvecs, alpha=fc_network_hparams.alpha)
        val_epoch_loss, val_time = validate(val_loader, diffusion_net, loss_fn, device, bvals, bvecs)

        logging.info(f'Epoch: {epoch:2}/{last_epoch + fc_network_hparams.num_epochs - 1:2} - ' + 
                     f'Train Loss: {train_epoch_loss:.8f} - ' +
                     f'Eval Loss: {val_epoch_loss:.8f} - ' +
                     f'Train Time: {train_time:.2f} - ' +
                     f'Eval Time: {val_time:.2f}')

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        epochs.append(epoch)
    
        if val_epoch_loss == min(val_losses):
            
            # save best checkpoint
            checkpoint = Checkpoint(
                model_state_dict=diffusion_net.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                train_losses=train_losses,
                val_losses=val_losses,
                epochs=epochs)
            
            with open(fc_network_paths.best_checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)

            with open(fc_network_paths.best_model_file, 'wb') as f:
                torch.save(diffusion_net, f)
            
            logging.info('Best checkpoint saved')
    
    # save last checkpoint
    checkpoint = Checkpoint(
        model_state_dict=diffusion_net.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        train_losses=train_losses,
        val_losses=val_losses,
        epochs=epochs)
    
    with open(fc_network_paths.last_checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logging.info('Last checkpoint saved')
    logging.info('')
    
    # save losses plot
    plt.semilogy(epochs, train_losses, label='Train Loss')
    plt.semilogy(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fc_network_paths.losses_plot_file)

    logging.info('-' * 50)
    logging.info('')

if __name__ == '__main__':
    main()
