'''Load simulation train and validation data.
Load b-values and b-vectors from processed data.
Create simple datasets and dataloaders.
Create an equivariant network for
approximating three eigenvalues and two eigenvectors
(the third eigenvector is the cross product of the first two)
using S0 correction.
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

import e3nn
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
    simulation_train_data_file: str
    simulation_eval_data_file: str


@dataclass
class EquivariantNNHyperparameters:
    seed: int
    
    lmax: int
    num_hidden_channels: int
    num_hidden_layers: int
    
    num_epochs: int
    learning_rate: float
    batch_size: int

    b_values_to_select: list[float]
    simulation_data_paths_pkl: str


class EquivariantNNPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('equivariant_nn', 'template_1.1', 'experiments', 
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


def get_selection_masks(b_values_to_select_list: list[float], b_values: np.ndarray) -> dict[float, torch.Tensor]:

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

    selection_masks = {}
    
    for b_value in list(b_values_to_select):
        selection_masks[b_value] = torch.isin(torch.from_numpy(b_values), b_value)

    return selection_masks


class SignalToIrreps(torch.nn.Module):
    
    def __init__(self, lmax: int):
        super().__init__()
        self.lmax = lmax
        self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(self.lmax)
    
    def forward(self, signals: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
        signals: torch.Tensor of shape (batch size x number of directions)
        grids: torch.Tensor of shape (batch size x number of directions x 3)

        Intermidiate:
        Y: torch.Tensor of shape (batch size x number of directions x number of spherical harmonics)
        
        Returns:
        coeffs: torch.Tensor of shape (batch size x number of spherical harmonics)
        '''
        
        assert signals.ndim == 2
        assert grids.ndim == 3
        assert signals.shape[0] == grids.shape[0]
        assert signals.shape[1] == grids.shape[1]
        assert grids.shape[2] == 3

        Y = e3nn.o3.spherical_harmonics(self.irreps_sh, grids, normalize=False, normalization='component') / torch.sqrt(torch.tensor(signals.shape[1]))
        coeffs = torch.linalg.lstsq(Y, signals).solution
        
        return coeffs


class SignalsDataset(Dataset):
    
    def __init__(self, signals: torch.Tensor):
        super().__init__()
        self.signals = signals

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        return self.signals[idx]


class DiffusionDataset(Dataset):
    
    def __init__(self, signals: torch.Tensor, coeffs: torch.Tensor):
        super().__init__()
        self.signals = signals
        self.coeffs = coeffs

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        return self.signals[idx], self.coeffs[idx]


class GatedLinear(torch.nn.Module):

    def __init__(self, in_irrep, out_irrep) -> None:
        super().__init__()

        self.in_irrep = in_irrep
        self.out_irrep = out_irrep

        irreps_scalars = e3nn.o3.Irreps(str(self.out_irrep[0]))
        irreps_gates = e3nn.o3.Irreps("{}x0e".format(self.out_irrep.num_irreps - irreps_scalars.num_irreps))
        irreps_gated = e3nn.o3.Irreps(str(self.out_irrep[1:]))

        pre_gate_irreps = irreps_scalars + irreps_gates + irreps_gated

        self.linear = e3nn.o3.Linear(self.in_irrep, pre_gate_irreps, internal_weights=True, shared_weights=True)
        self.gating = e3nn.nn.Gate(irreps_scalars, [torch.nn.SiLU()], irreps_gates, [torch.nn.Sigmoid()], irreps_gated)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gating(self.linear(x))


class EquivariantNet(torch.nn.Module):
    
    def __init__(
            self,
            input_irrep: e3nn.o3.Irreps,
            num_hidden_channels: int,
            num_hidden_layers: int
        ):
        super().__init__()

        hidden_irrep = (input_irrep * num_hidden_channels).sort().irreps.simplify()
        output_irrep = e3nn.o3.Irreps("4x0e + 2x1o")

        layer_irreps = [input_irrep] + [hidden_irrep] * num_hidden_layers + [output_irrep]

        self.net = torch.nn.Sequential()
        for i in range(len(layer_irreps) - 1):
            self.net.append(GatedLinear(layer_irreps[i], layer_irreps[i + 1]))

    def forward(self, sh_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        output = self.net(sh_coeffs)

        S0, eigval_1, eigval_2, eigval_3, eigvec_1, eigvec_2 = output.split([1, 1, 1, 1, 3, 3], dim=1)

        S0 = S0.squeeze()

        eigvals = torch.zeros((eigval_1.shape[0], 3, 3), device=sh_coeffs.device)
        eigvals[:, 0, 0] = eigval_1.squeeze()
        eigvals[:, 1, 1] = eigval_2.squeeze()
        eigvals[:, 2, 2] = eigval_3.squeeze()
        
        # #? We make each eigenvector in eigvec_2 orthogonal 
        # #? to the corresponding eigenvector in eigvec_1
        # #? by subtracting the projection of eigvec_2 onto eigvec_1.
        # ? eigvec_2 = eigvec_2 - ((eigvec_2 * eigvec_1).sum(dim=1, keepdim=True) / (eigvec_1 * eigvec_1).sum(dim=1, keepdim=True)) * eigvec_1

        # We normalize the eigenvectors
        eigvec_1 = eigvec_1 / eigvec_1.norm(dim=1, keepdim=True)
        eigvec_2 = eigvec_2 / eigvec_2.norm(dim=1, keepdim=True)

        # We create the third eigenvector by taking 
        # the cross product of the first two eigenvectors
        eigvec_3 = torch.cross(eigvec_1, eigvec_2, dim=1)

        # We stack the eigenvectors to create the eigenvector matrix
        # with shape (batch_size, 3, 3)
        # The eigenvectors are the columns of the matrix
        # The eigenvectors are orthogonal and normalized
        eigvecs = torch.stack([eigvec_1, eigvec_2, eigvec_3], dim=2)

        # We create the diffusion tensor by multiplying the eigenvector matrix of shape (batch_size, 3, 3)
        # with the eigenvalue matrix of shape (batch_size, 3, 3) and the transpose of the eigenvector matrix
        D = torch.bmm(eigvecs, torch.bmm(eigvals, eigvecs.transpose(1, 2)))

        return D, S0


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
    
    for signals, coeffs in train_loader:
        
        signals = signals.to(device)
        coeffs = coeffs.to(device)

        D, S0 = model(coeffs)
        
        recon_signals = reconstruct(S0, bvals, bvecs, D)
        
        batch_loss = loss_fn(recon_signals, signals)
    
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
        
        for signals, coeffs in val_loader:

            signals = signals.to(device)
            coeffs = coeffs.to(device)

            D, S0 = model(coeffs)

            recon_signals = reconstruct(S0, bvals, bvecs, D)

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

        ## EQUIVARIANT NN PATHS
        
        with open(resume_args.resume_experiment_paths_pkl, 'rb') as f:
            equivariant_nn_paths: EquivariantNNPaths = pickle.load(f)
        
        
        ## LOGGING

        logging.basicConfig(
            filename=equivariant_nn_paths.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s')
        
        
        ## EQUIVARIANT NN HYPERPARAMETERS
        
        with open(equivariant_nn_paths.hyperparameters_file, 'rb') as f:
            equivariant_nn_hparams: EquivariantNNHyperparameters = pickle.load(f)

        secondary_parser = ArgumentParser()
        secondary_parser.add_argument('--num_epochs', type=int, required=True)
        secondary_parser.add_argument('--learning_rate', type=float, required=True)
        secondary_args = secondary_parser.parse_args(other_args)

        equivariant_nn_hparams.num_epochs = secondary_args.num_epochs
        equivariant_nn_hparams.learning_rate = secondary_args.learning_rate

        with open(equivariant_nn_paths.hyperparameters_file, 'wb') as f:
            pickle.dump(equivariant_nn_hparams, f)
        
        logging.info('New Equivariant NN hyperparameters:')
        logging.info(f'num_epochs: {equivariant_nn_hparams.num_epochs}')
        logging.info(f'learning_rate: {equivariant_nn_hparams.learning_rate}')
        logging.info('')

    
    else: # New experiment

        ## EQUIVARIANT NN PATHS

        equivariant_nn_paths = EquivariantNNPaths()

        print('Experiment path:', equivariant_nn_paths.experiment_path)

        os.makedirs(equivariant_nn_paths.experiment_path)

        with open(equivariant_nn_paths.paths_file, 'wb') as f:
            pickle.dump(equivariant_nn_paths, f)
        
        
        ## LOGGING

        logging.basicConfig(
            filename=equivariant_nn_paths.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info('Equivariant NN experiment:')
        logging.info(equivariant_nn_paths.experiment_path)
        logging.info('')


        ## FC NETWORK HYPERPARAMETERS

        main_parser = ArgumentParser()
        
        main_parser.add_argument('--seed', type=int, default=0)

        main_parser.add_argument('--lmax', type=int, required=True)
        main_parser.add_argument('--num_hidden_channels', type=int, required=True)
        main_parser.add_argument('--num_hidden_layers', type=int, required=True)
        
        main_parser.add_argument('--num_epochs', type=int, required=True)
        main_parser.add_argument('--learning_rate', type=float, required=True)
        main_parser.add_argument('--batch_size', type=int, required=True)

        main_parser.add_argument('--b_values_to_select', type=float, nargs='*')
        main_parser.add_argument('--simulation_data_paths_pkl', type=str, required=True)

        main_args = main_parser.parse_args(other_args)

        equivariant_nn_hparams = EquivariantNNHyperparameters(**vars(main_args))

        with open(equivariant_nn_paths.hyperparameters_file, 'wb') as f:
            pickle.dump(equivariant_nn_hparams, f)
        
        logging.info('Equicariant NN hyperparameters:')
        for key, value in vars(equivariant_nn_hparams).items():
            logging.info(f'{key}: {value}')
        logging.info('')
    

    ## DEVICE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f'Using {device} device')
    logging.info('')


    ## REPRODUCIBILITY

    # Set seed for Python's random module
    random.seed(equivariant_nn_hparams.seed)

    # Set seed for NumPy's random module
    np.random.seed(equivariant_nn_hparams.seed)

    # Set seed for PyTorch's CPU RNG
    torch.manual_seed(equivariant_nn_hparams.seed)

    # If CUDA is available, set seed for CUDA RNGs and enable deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed(equivariant_nn_hparams.seed)
        torch.cuda.manual_seed_all(equivariant_nn_hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    generator = torch.Generator().manual_seed(equivariant_nn_hparams.seed)
    

    ## SIMULATION DATA PATHS

    with open(equivariant_nn_hparams.simulation_data_paths_pkl, 'rb') as f:
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

    b_values = np.load(proc_data_paths.b_values_file)
    b_vectors = np.load(proc_data_paths.b_vectors_file)
    
    train_data = np.load(sim_data_paths.simulation_train_data_file)
    eval_data = np.load(sim_data_paths.simulation_eval_data_file)

    selection_masks = get_selection_masks(equivariant_nn_hparams.b_values_to_select, b_values)


    ## TORCH CONVERSION

    b_values = torch.from_numpy(b_values).float()
    b_vectors = torch.from_numpy(b_vectors).float()

    train_data = torch.from_numpy(train_data).float()
    eval_data = torch.from_numpy(eval_data).float()

    
    # SIGNAL TO IRREPS IN BATCHES

    signal_to_irreps = SignalToIrreps(equivariant_nn_hparams.lmax)

    bvals = []
    bvecs = []
    
    train_signals = []
    eval_signals = []

    train_coeffs = []
    eval_coeffs = []

    for b_value, selection_mask in selection_masks.items():

        print(f'Computing train and eval coeffs for b-value: {b_value}')

        selected_bvals = b_values[selection_mask]
        selected_bvecs = b_vectors[selection_mask]

        selected_train_signals = train_data[:, selection_mask]
        selected_eval_signals = eval_data[:, selection_mask]

        selected_train_dataset = SignalsDataset(selected_train_signals)
        selected_eval_dataset = SignalsDataset(selected_eval_signals)

        selected_train_loader = DataLoader(
            dataset=selected_train_dataset, 
            batch_size=equivariant_nn_hparams.batch_size, 
            shuffle=False,
            drop_last=False)
        
        selected_eval_loader = DataLoader(
            dataset=selected_eval_dataset, 
            batch_size=equivariant_nn_hparams.batch_size, 
            shuffle=False,
            drop_last=False)
        
        selected_train_coeffs = []
        for signals in tqdm(selected_train_loader):
            selected_train_coeffs.append(
                signal_to_irreps(
                    signals, 
                    selected_bvecs.unsqueeze(0).repeat(signals.shape[0], 1, 1)))
        
        selected_eval_coeffs = []
        for signals in tqdm(selected_eval_loader):
            selected_eval_coeffs.append(
                signal_to_irreps(
                    signals, 
                    selected_bvecs.unsqueeze(0).repeat(signals.shape[0], 1, 1)))
        
        bvals.append(selected_bvals)
        bvecs.append(selected_bvecs)
        
        train_signals.append(selected_train_signals)
        eval_signals.append(selected_eval_signals)

        train_coeffs.append(torch.cat(selected_train_coeffs, dim=0))
        eval_coeffs.append(torch.cat(selected_eval_coeffs, dim=0))

    bvals = torch.cat(bvals, dim=0).to(device)
    bvecs = torch.cat(bvecs, dim=0).to(device)
    
    train_signals = torch.cat(train_signals, dim=1)
    eval_signals = torch.cat(eval_signals, dim=1)

    train_coeffs = torch.cat(train_coeffs, dim=1)
    eval_coeffs = torch.cat(eval_coeffs, dim=1)

    
    ## DATASETS AND DATALOADERS

    train_dataset = DiffusionDataset(train_signals, train_coeffs)
    val_dataset = DiffusionDataset(eval_signals, eval_coeffs)

    logging.info(f'Train dataset: {len(train_dataset)}')
    logging.info(f'Validation dataset: {len(val_dataset)}')
    logging.info('')

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=equivariant_nn_hparams.batch_size, 
        shuffle=True, 
        generator=generator, 
        drop_last=True)
    
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=equivariant_nn_hparams.batch_size, 
        shuffle=False, 
        drop_last=True)
    

    ## INITIALIZE MODEL, LOSS FUNCTION AND OPTIMIZER

    equivariant_net = EquivariantNet(
        input_irrep=signal_to_irreps.irreps_sh * len(selection_masks),
        num_hidden_channels=equivariant_nn_hparams.num_hidden_channels,
        num_hidden_layers=equivariant_nn_hparams.num_hidden_layers
    ).to(device)

    logging.info(equivariant_net)
    logging.info('')

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(params=equivariant_net.parameters(), lr=equivariant_nn_hparams.learning_rate)

    train_losses = []
    val_losses = []
    epochs = []


    ## RESUME TRAINING

    if resume_args.resume_experiment_paths_pkl:

        with open(equivariant_nn_paths.last_checkpoint_file, 'rb') as f:
            checkpoint: Checkpoint = pickle.load(f)

        equivariant_net.load_state_dict(checkpoint.model_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        train_losses = checkpoint.train_losses
        val_losses = checkpoint.val_losses
        epochs = checkpoint.epochs

        logging.info(f'Resuming training from epoch {len(epochs)}')
        logging.info('')
    

    ## TRAIN MODEL

    last_epoch = epochs[-1] + 1 if epochs else 1

    for epoch in tqdm(range(last_epoch, last_epoch + equivariant_nn_hparams.num_epochs)):

        train_epoch_loss, train_time = train(train_loader, equivariant_net, loss_fn, optimizer, device, bvals, bvecs)
        val_epoch_loss, val_time = validate(val_loader, equivariant_net, loss_fn, device, bvals, bvecs)

        logging.info(f'Epoch: {epoch:2}/{last_epoch + equivariant_nn_hparams.num_epochs - 1:2} - ' + 
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
                model_state_dict=equivariant_net.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                train_losses=train_losses,
                val_losses=val_losses,
                epochs=epochs)
            
            with open(equivariant_nn_paths.best_checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            with open(equivariant_nn_paths.best_model_file, 'wb') as f:
                torch.save(equivariant_net, f)

            logging.info('Best checkpoint saved')
    
    # save last checkpoint
    checkpoint = Checkpoint(
        model_state_dict=equivariant_net.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        train_losses=train_losses,
        val_losses=val_losses,
        epochs=epochs)
    
    with open(equivariant_nn_paths.last_checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logging.info('Last checkpoint saved')
    logging.info('')
    
    # save losses plot
    plt.semilogy(epochs, train_losses, label='Train Loss')
    plt.semilogy(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(equivariant_nn_paths.losses_plot_file)
    plt.clf()

    logging.info('-' * 50)
    logging.info('')

if __name__ == '__main__':
    main()
