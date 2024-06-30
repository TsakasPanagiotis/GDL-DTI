'''Connected to fc_network template 5.
Load simulated test data.
Load processed b-values and b-vectors.
Use signals from selected b-values that include zero.
Pass through the best model.
Save diffusion tensors and S0_corrections.
Measure reconstruction errors.'''


import os
import sys
sys.path.append(os.getcwd())
import random
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from fc_network.template_5.run import (
    DiffusionNet,
    DiffusionDataset,
    get_selection_mask
)


class ProcessedDataPaths(Protocol):
    b_values_file: str
    b_vectors_file: str


class GroundTruthHyperparameters(Protocol):
    processed_data_paths_pkl: str


class GroundTruthPaths(Protocol):
    hyperparameters_file: str
    d_tensors_file: str


class SimulationDataHyperparameters(Protocol):
    ground_truth_paths_pkl: str
    test_snrs: list[float]


class SimulationDataPaths(Protocol):
    hyperparameters_file: str
    test_data_template: str


class FCNetworkHyperparameters(Protocol):
    b_values_to_select: list[float]
    simulation_data_paths_pkl: str


class FCNetworkPaths(Protocol):
    hyperparameters_file: str
    best_model_file: str


@dataclass
class FCNetworkAnalysisHyperparameters:
    seed: int
    snr: float
    batch_size: int
    fc_network_paths_pkl: str


class FCNetworkAnalysisPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('fc_network_analysis', 'template_4', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')

        self.pred_d_tensors_file = os.path.join(self.experiment_path, 'pred_d_tensors.npy')


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


def main():

    ## FC NETWORK ANALYSIS PATHS

    fc_network_analysis_paths = FCNetworkAnalysisPaths()

    print(f'Experiment path: {fc_network_analysis_paths.experiment_path}')

    os.makedirs(fc_network_analysis_paths.experiment_path)

    with open(fc_network_analysis_paths.paths_file, 'wb') as f:
        pickle.dump(fc_network_analysis_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=fc_network_analysis_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f'FC network analysis experiment:')
    logging.info(fc_network_analysis_paths.experiment_path)
    logging.info('')


    ## FC NETWORK ANALYSIS HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--fc_network_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    fc_network_analysis_hparams = FCNetworkAnalysisHyperparameters(**vars(args))

    with open(fc_network_analysis_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(fc_network_analysis_hparams, f)
    
    logging.info('FC network analysis hyperparameters:')
    for key, value in vars(fc_network_analysis_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## DEVICE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f'Using {device} device')
    logging.info('')


    ## REPRODUCIBILITY

    # Set seed for Python's random module
    random.seed(fc_network_analysis_hparams.seed)

    # Set seed for NumPy's random module
    np.random.seed(fc_network_analysis_hparams.seed)

    # Set seed for PyTorch's CPU RNG
    torch.manual_seed(fc_network_analysis_hparams.seed)

    # If CUDA is available, set seed for CUDA RNGs and enable deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fc_network_analysis_hparams.seed)
        torch.cuda.manual_seed_all(fc_network_analysis_hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    generator = torch.Generator().manual_seed(fc_network_analysis_hparams.seed)
    

    ## FC NETWORK PATHS

    with open(fc_network_analysis_hparams.fc_network_paths_pkl, 'rb') as f:
        fc_network_paths: FCNetworkPaths = pickle.load(f)
    

    ## FC NETWORK HYPERPARAMETERS

    with open(fc_network_paths.hyperparameters_file, 'rb') as f:
        fc_network_hparams: FCNetworkHyperparameters = pickle.load(f)
    

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

    assert fc_network_analysis_hparams.snr in sim_data_hparams.test_snrs, \
        f'Invalid snr: {fc_network_analysis_hparams.snr}. Valid values are: {sim_data_hparams.test_snrs}'


    b_values: np.ndarray = np.load(proc_data_paths.b_values_file)
    b_vectors: np.ndarray = np.load(proc_data_paths.b_vectors_file)
    
    test_data = np.load(sim_data_paths.test_data_template.format(fc_network_analysis_hparams.snr))

    selection_mask = get_selection_mask(fc_network_hparams.b_values_to_select, b_values)

    
    ## TORCH CONVERSION

    bvals = torch.from_numpy(b_values[selection_mask]).float().to(device)
    bvecs = torch.from_numpy(b_vectors[selection_mask]).float().to(device)

    test_signals = torch.from_numpy(test_data[:, selection_mask]).float()


    ## DATASET AND DATALOADER

    test_dataset = DiffusionDataset(test_signals)

    logging.info(f'Test dataset: {len(test_dataset)}')
    logging.info('')
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=fc_network_analysis_hparams.batch_size, 
        shuffle=False, 
        drop_last=False)
    

    ## LOAD MODEL

    model: DiffusionNet = torch.load(fc_network_paths.best_model_file).to(device)

    logging.info('Model loaded')
    logging.info(model)
    logging.info('')

    model.eval()


    ## EVALUATION

    errors = []
    pred_d_tensors = []

    with torch.no_grad():
        for signals in test_loader:
            signals = signals.to(device)
            D, S0 = model(signals)
            recon_signals = reconstruct(S0, bvals, bvecs, D)
            errors.append(recon_signals - signals)
            pred_d_tensors.append(D)
    
    pred_d_tensors = torch.cat(pred_d_tensors, dim=0)
    np.save(fc_network_analysis_paths.pred_d_tensors_file, pred_d_tensors.cpu().numpy())
    logging.info(f'Saved {pred_d_tensors.shape[0]} predicted diffusion tensors.')
    logging.info('')
    
    all_errors = torch.cat(errors, dim=0).cpu().numpy()    
    all_errors_mean = np.mean(all_errors)
    all_errors_std = np.std(all_errors)
    logging.info(f'Errors: {all_errors_mean} +- {all_errors_std}')

    mean_error_per_voxel = np.mean(all_errors, axis=1)
    mean_error_per_voxel_mean = np.mean(mean_error_per_voxel)
    mean_error_per_voxel_std = np.std(mean_error_per_voxel)
    logging.info(f'Mean error per voxel: {mean_error_per_voxel_mean} +- {mean_error_per_voxel_std}')

    mean_absolute_error_per_voxel = np.mean(np.abs(all_errors), axis=1)
    mean_absolute_error_per_voxel_mean = np.mean(mean_absolute_error_per_voxel)
    mean_absolute_error_per_voxel_std = np.std(mean_absolute_error_per_voxel)
    logging.info(f'Mean absolute error per voxel: {mean_absolute_error_per_voxel_mean} +- {mean_absolute_error_per_voxel_std}')

    root_mean_squared_error_per_voxel = np.sqrt(np.mean(all_errors**2, axis=1))
    root_mean_squared_error_per_voxel_mean = np.mean(root_mean_squared_error_per_voxel)
    root_mean_squared_error_per_voxel_std = np.std(root_mean_squared_error_per_voxel)
    logging.info(f'Root mean squared error per voxel: {root_mean_squared_error_per_voxel_mean} +- {root_mean_squared_error_per_voxel_std}')


if __name__ == '__main__':
    main()
