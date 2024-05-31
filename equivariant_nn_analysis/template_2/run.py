'''Connected to equivariant_nn template 1.1.
Load simulated test data.
Load processed b-values and b-vectors.
Create noisy test signals for given snr.
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

import e3nn
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from equivariant_nn.template_1_1.run import (
    SignalToIrreps,
    SignalsDataset,
    DiffusionDataset,
    EquivariantNet,
    GatedLinear,
    get_selection_masks
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


class EquivariantNNHyperparameters(Protocol):
    lmax: int
    b_values_to_select: list[float]
    simulation_data_paths_pkl: str


class EquivariantNNPaths(Protocol):
    hyperparameters_file: str
    best_model_file: str


@dataclass
class EquivariantNNAnalysisHyperparameters:
    seed: int
    snr: float
    batch_size: int
    equivariant_nn_paths_pkl: str


class EquivariantNNAnalysisPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('equivariant_nn_analysis', 'template_2', 'experiments', 
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

    ## EQUIVARIANT NN ANALYSIS PATHS

    equivariant_nn_analysis_paths = EquivariantNNAnalysisPaths()

    print(f'Experiment path: {equivariant_nn_analysis_paths.experiment_path}')

    os.makedirs(equivariant_nn_analysis_paths.experiment_path)

    with open(equivariant_nn_analysis_paths.paths_file, 'wb') as f:
        pickle.dump(equivariant_nn_analysis_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=equivariant_nn_analysis_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f'Equivariant nn analysis experiment:')
    logging.info(equivariant_nn_analysis_paths.experiment_path)
    logging.info('')


    ## EQUIVARIANT NN ANALYSIS HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)    
    parser.add_argument('--snr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--equivariant_nn_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    equivariant_nn_analysis_hparams = EquivariantNNAnalysisHyperparameters(**vars(args))

    with open(equivariant_nn_analysis_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(equivariant_nn_analysis_hparams, f)
    
    logging.info('Equivariant nn analysis hyperparameters:')
    for key, value in vars(equivariant_nn_analysis_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## DEVICE

    device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f'Using {device} device')
    logging.info('')


    ## REPRODUCIBILITY

    # Set seed for Python's random module
    random.seed(equivariant_nn_analysis_hparams.seed)

    # Set seed for NumPy's random module
    np.random.seed(equivariant_nn_analysis_hparams.seed)

    # Set seed for PyTorch's CPU RNG
    torch.manual_seed(equivariant_nn_analysis_hparams.seed)

    # If CUDA is available, set seed for CUDA RNGs and enable deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed(equivariant_nn_analysis_hparams.seed)
        torch.cuda.manual_seed_all(equivariant_nn_analysis_hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    generator = torch.Generator().manual_seed(equivariant_nn_analysis_hparams.seed)
    

    ## FC NETWORK PATHS

    with open(equivariant_nn_analysis_hparams.equivariant_nn_paths_pkl, 'rb') as f:
        equivariant_nn_paths: EquivariantNNPaths = pickle.load(f)
    

    ## FC NETWORK HYPERPARAMETERS

    with open(equivariant_nn_paths.hyperparameters_file, 'rb') as f:
        equivariant_nn_hparams: EquivariantNNHyperparameters = pickle.load(f)
    

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

    assert equivariant_nn_analysis_hparams.snr in sim_data_hparams.test_snrs, \
        f'Invalid snr: {equivariant_nn_analysis_hparams.snr}. Valid values are: {sim_data_hparams.test_snrs}'

    b_values: np.ndarray = np.load(proc_data_paths.b_values_file)
    b_vectors: np.ndarray = np.load(proc_data_paths.b_vectors_file)

    test_data = np.load(sim_data_paths.test_data_template.format(equivariant_nn_analysis_hparams.snr))

    selection_masks = get_selection_masks(equivariant_nn_hparams.b_values_to_select, b_values)
    
    
    ## TORCH CONVERSION

    b_values = torch.from_numpy(b_values).float()
    b_vectors = torch.from_numpy(b_vectors).float()

    test_data = torch.from_numpy(test_data).float()


    # SIGNAL TO IRREPS IN BATCHES

    signal_to_irreps = SignalToIrreps(equivariant_nn_hparams.lmax)

    bvals = []
    bvecs = []
    
    test_signals = []

    test_coeffs = []

    for b_value, selection_mask in selection_masks.items():

        print(f'Computing test coeffs for b-value: {b_value}')

        selected_bvals = b_values[selection_mask]
        selected_bvecs = b_vectors[selection_mask]

        selected_test_signals = test_data[:, selection_mask]

        selected_test_dataset = SignalsDataset(selected_test_signals)

        selected_test_loader = DataLoader(
            dataset=selected_test_dataset, 
            batch_size=equivariant_nn_analysis_hparams.batch_size, 
            shuffle=False,
            drop_last=False)
        
        selected_test_coeffs = []
        for signals in tqdm(selected_test_loader):
            selected_test_coeffs.append(
                signal_to_irreps(
                    signals, 
                    selected_bvecs.unsqueeze(0).repeat(signals.shape[0], 1, 1)))
        
        bvals.append(selected_bvals)
        bvecs.append(selected_bvecs)
        
        test_signals.append(selected_test_signals)

        test_coeffs.append(torch.cat(selected_test_coeffs, dim=0))

    bvals = torch.cat(bvals, dim=0).to(device)
    bvecs = torch.cat(bvecs, dim=0).to(device)
    
    test_signals = torch.cat(test_signals, dim=1)

    test_coeffs = torch.cat(test_coeffs, dim=1)


    ## DATASET AND DATALOADER

    test_dataset = DiffusionDataset(test_signals, test_coeffs)

    logging.info(f'Test dataset: {len(test_dataset)}')
    logging.info('')
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=equivariant_nn_analysis_hparams.batch_size, 
        shuffle=False, 
        drop_last=False)
    

    ## LOAD MODEL

    model: EquivariantNet = torch.load(equivariant_nn_paths.best_model_file).to(device)

    logging.info('Model loaded')
    logging.info(model)
    logging.info('')

    model.eval()


    ## EVALUATION

    errors = []
    pred_d_tensors = []

    with torch.no_grad():
        for signals, coeffs in test_loader:
            signals = signals.to(device)
            coeffs = coeffs.to(device)
            D, S0 = model(coeffs)
            recon_signals = reconstruct(S0, bvals, bvecs, D)
            errors.append(recon_signals - signals)
            pred_d_tensors.append(D)
    
    pred_d_tensors = torch.cat(pred_d_tensors, dim=0)
    np.save(equivariant_nn_analysis_paths.pred_d_tensors_file, pred_d_tensors.cpu().numpy())
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
