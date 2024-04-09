'''Load simulation data.
Load b-values and b-vectors from processed data.
Create simple datasets and dataloaders.
Create a fully connected network 
based on ground truth template 5.2:
angle and eigenvalue parameters with sigmoid
and S0 correction with tanh.
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
from typing import Any, Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Callable, TypeVar

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class ProcessedDataPaths(Protocol):
    b_values_file: str
    b_vectors_file: str


class GroundTruthHyperparameters(Protocol):
    processed_data_paths_pkl: str


class GroundTruthPaths(Protocol):
    hyperparameters_file: str


class SimulationDataHyperparameters(Protocol):
    ground_truth_paths_pkl: str


class SimulationDataPaths(Protocol):
    hyperparameters_file: str
    simulation_data_file: str


@dataclass
class FCNetworkHyperparameters:
    seed: int
    hidden_sizes: list[int]
    num_epochs: int
    learning_rate: float
    batch_size: int
    train_percent: float

    threshold_eigval: float
    b_values_to_select: list[float]
    simulation_data_paths_pkl: str


class FCNetworkPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('fc_network', 'template_1.2', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')

        self.losses_plot_file = os.path.join(self.experiment_path, 'losses.png')
        self.last_checkpoint_file = os.path.join(self.experiment_path, 'last_checkpoint.pkl')
        self.best_checkpoint_file = os.path.join(self.experiment_path, 'best_checkpoint.pkl')


@dataclass
class Checkpoint:
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    train_losses: list[float]
    val_losses: list[float]
    epochs: list[int]


def create_masks(b_values_to_select_list: list[float], b_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    b_values_to_select = set(b_values_to_select_list)
    unique_nonzero_b_values = set(b_values) - {0.0}

    if len(b_values_to_select) == 0:
        b_values_to_select = unique_nonzero_b_values
        logging.warning(f'b_values_to_select is empty. Using all nonzero b-values: {b_values_to_select}')
        logging.warning('')
    
    if 0.0 in b_values_to_select:
        logging.error('b_values_to_select must not contain 0.0')
        raise ValueError('b_values_to_select must not contain 0.0')

    if not b_values_to_select.issubset(unique_nonzero_b_values):
        logging.error(f'Invalid b_values_to_select: {b_values_to_select.difference(unique_nonzero_b_values)}. ' \
                      + f'Valid values are: {unique_nonzero_b_values}')
        raise ValueError(f'Invalid b_values_to_select: {b_values_to_select.difference(unique_nonzero_b_values)}. ' \
                         + f'Valid values are: {unique_nonzero_b_values}')

    selection_mask = np.isin(b_values, list(b_values_to_select))
    zero_mask = b_values == 0.0

    return selection_mask, zero_mask


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

    def forward(self, input: torch.Tensor) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        output = self.net(input)

        # Split output into diffusion tensor parameters and S0 corrections
        d_params, S0 = output[:, :-1], output[:, -1]

        d_params = torch.sigmoid(d_params)
        
        # Split the diffusion tensor parameters into individual elements
        x_angles, y_angles, z_angles, \
        eig_val_1, eig_val_2_over_1, eig_val_3_over_2, \
        = torch.split(d_params, split_size_or_sections=1, dim=1)
        
        # Compute angles
        x_angles = x_angles.squeeze() * 2 * torch.pi
        y_angles = y_angles.squeeze() * torch.pi
        z_angles = z_angles.squeeze() * 2 * torch.pi
        
        # Compute eigenvalues
        eig_val_1 = eig_val_1.squeeze() * self.threshold_eigval
        eig_val_2 = eig_val_1 * eig_val_2_over_1.squeeze()
        eig_val_3 = eig_val_2 * eig_val_3_over_2.squeeze()

        # Create the roation matrices around the x axis
        R_x = torch.zeros((x_angles.shape[0], 3, 3))
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = torch.cos(x_angles)
        R_x[:, 1, 2] = -torch.sin(x_angles)
        R_x[:, 2, 1] = torch.sin(x_angles)
        R_x[:, 2, 2] = torch.cos(x_angles)

        # Create the roation matrices around the y axis
        R_y = torch.zeros((y_angles.shape[0], 3, 3))
        R_y[:, 0, 0] = torch.cos(y_angles)
        R_y[:, 0, 2] = torch.sin(y_angles)
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -torch.sin(y_angles)
        R_y[:, 2, 2] = torch.cos(y_angles)

        # Create the roation matrices around the z axis
        R_z = torch.zeros((z_angles.shape[0], 3, 3))
        R_z[:, 0, 0] = torch.cos(z_angles)
        R_z[:, 0, 1] = -torch.sin(z_angles)
        R_z[:, 1, 0] = torch.sin(z_angles)
        R_z[:, 1, 1] = torch.cos(z_angles)
        R_z[:, 2, 2] = 1

        # Calculate the rotation matrices.
        R = torch.bmm(R_z, torch.bmm(R_y, R_x))

        # Calculate the diagonal matrix of eigenvalues
        E = torch.zeros((eig_val_1.shape[0], 3, 3))
        E[:, 0, 0] = eig_val_1
        E[:, 1, 1] = eig_val_2
        E[:, 2, 2] = eig_val_3

        # Reconstruct the diffusion tensors
        D = torch.bmm(R, torch.bmm(E, R.transpose(1, 2)))

        # S0 corrections between 0 and 2
        S0 = torch.tanh(S0).squeeze() + 1

        return R, E, D, S0


def reconstruct(S0, bvals, bvecs, D) -> torch.Tensor:
    '''
    Parameters:
        S0: torch.Tensor of shape (batch_size,)
        bvals: torch.Tensor of shape (channels,)
        bvecs: torch.Tensor of shape (channels, 3)
        D: torch.Tensor of shape (batch_size, 3, 3)
    
    Returns:
        reconstructed signals: torch.Tensor of shape (batch_size, channels)
    '''
    return torch.einsum('b, bc -> bc', S0, torch.exp(- torch.einsum('c, ci, bij, cj -> bc', bvals, bvecs, D, bvecs)))


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
def train(train_loader, model, loss_fn, optimizer, device, bvals, bvecs):

    model.train()
    epoch_loss = 0.0
    invalid_count = 0
    
    for signals in train_loader:
        
        signals = signals.to(device)

        R, E, D, S0 = model(signals)
        
        recon_signals = reconstruct(S0, bvals, bvecs, D)
        
        batch_loss = loss_fn(recon_signals, signals)
    
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

        symmetric_mask = torch.isclose(D, D.transpose(1,2), atol=1e-8).all(dim=(1,2))
        positive_mask = (torch.linalg.eigvals(D.to(torch.complex64)).real > 0).all(dim=1)
        threshold_mask = (torch.linalg.eigvals(D.to(torch.complex64)).real <= model.threshold_eigval).all(dim=1)

        mask = symmetric_mask & positive_mask & threshold_mask
        invalid_count += (~mask).sum().item()
        
    epoch_loss /= len(train_loader)

    return epoch_loss, invalid_count


@timer
def validate(val_loader, model, loss_fn, device, bvals, bvecs):

    model.eval()
    epoch_loss = 0.0
    invalid_count = 0
    
    with torch.no_grad():
        
        for signals in val_loader:

            signals = signals.to(device)

            R, E, D, S0 = model(signals)

            recon_signals = reconstruct(S0, bvals, bvecs, D)

            batch_loss = loss_fn(recon_signals, signals)

            epoch_loss += batch_loss.item()

            symmetric_mask = torch.isclose(D, D.transpose(1,2), atol=1e-8).all(dim=(1,2))
            positive_mask = (torch.linalg.eigvals(D.to(torch.complex64)).real > 0).all(dim=1)
            threshold_mask = (torch.linalg.eigvals(D.to(torch.complex64)).real <= model.threshold_eigval).all(dim=1)

            mask = symmetric_mask & positive_mask & threshold_mask
            invalid_count += (~mask).sum().item()
        
    epoch_loss /= len(val_loader)

    return epoch_loss, invalid_count


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
        
        main_parser.add_argument('--seed', type=int, required=True)
        main_parser.add_argument('--hidden_sizes', nargs='+', type=int, required=True)
        main_parser.add_argument('--num_epochs', type=int, required=True)
        main_parser.add_argument('--learning_rate', type=float, required=True)
        main_parser.add_argument('--batch_size', type=int, required=True)
        main_parser.add_argument('--train_percent', type=float, required=True)
        
        main_parser.add_argument('--threshold_eigval', type=float, required=True)
        main_parser.add_argument('--b_values_to_select', type=float, nargs='*')
        main_parser.add_argument('--simulation_data_paths_pkl', type=str, required=True)

        main_args = main_parser.parse_args(other_args)

        fc_network_hparams = FCNetworkHyperparameters(**vars(main_args))

        with open(fc_network_paths.hyperparameters_file, 'wb') as f:
            pickle.dump(fc_network_hparams, f)
        
        logging.info('FC network hyperparameters:')
        for key, value in vars(fc_network_hparams).items():
            logging.info(f'{key}: {value}')
        logging.info('')
    

    ## DEVICE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    

    ## SIMULATION DATA HYPERPARAMETERS

    with open(sim_data_paths.hyperparameters_file, 'rb') as f:
        sim_data_hparams: SimulationDataHyperparameters = pickle.load(f)
    

    ## GROUND TRUTH PATHS

    with open(sim_data_hparams.ground_truth_paths_pkl, 'rb') as f:
        ground_truth_paths: GroundTruthPaths = pickle.load(f)
    

    ## GROUND TRUTH HYPERPARAMETERS

    with open(ground_truth_paths.hyperparameters_file, 'rb') as f:
        ground_truth_hparams: GroundTruthHyperparameters = pickle.load(f)
    

    ## PROCESSED DATA PATHS

    with open(ground_truth_hparams.processed_data_paths_pkl, 'rb') as f:
        proc_data_paths: ProcessedDataPaths = pickle.load(f)
    

    ## NUMPY DATA

    b_values: np.ndarray = np.load(proc_data_paths.b_values_file)
    b_vectors: np.ndarray = np.load(proc_data_paths.b_vectors_file)
    sim_data: np.ndarray = np.load(sim_data_paths.simulation_data_file)

    selection_mask, zero_mask = create_masks(fc_network_hparams.b_values_to_select, b_values)

    ## TORCH CONVERSION

    bvals = torch.from_numpy(b_values[selection_mask]).float().to(device)
    bvecs = torch.from_numpy(b_vectors[selection_mask]).float().to(device)
    
    signals = torch.from_numpy(sim_data[:, selection_mask]).float()

    
    ## SHUFFLE AND SPLIT INDICES

    num_data = sim_data.shape[0]
    num_train = int(fc_network_hparams.train_percent * num_data)

    indices = torch.randperm(num_data, generator=generator)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]


    ## DATASETS AND DATALOADERS

    train_dataset = DiffusionDataset(signals[train_indices])
    val_dataset = DiffusionDataset(signals[val_indices])

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
        input_size=signals.shape[1],
        hidden_sizes=fc_network_hparams.hidden_sizes,
        output_size=7,
        threshold_eigval=fc_network_hparams.threshold_eigval
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

        (train_epoch_loss, train_invalid_count), train_time = train(train_loader, diffusion_net, loss_fn, optimizer, device, bvals, bvecs)
        (val_epoch_loss, val_invalid_count), val_time = validate(val_loader, diffusion_net, loss_fn, device, bvals, bvecs)

        logging.info(f'Epoch: {epoch:2}/{last_epoch + fc_network_hparams.num_epochs - 1:2} - ' + 
                     f'Train Loss: {train_epoch_loss:.8f} - ' +
                     f'Eval Loss: {val_epoch_loss:.8f} - ' +
                     f'Train Invalid: {train_invalid_count:4} - ' +
                     f'Eval Invalid: {val_invalid_count:4} - ' +
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
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fc_network_paths.losses_plot_file)

    logging.info('-' * 50)
    logging.info('')

if __name__ == '__main__':
    main()
