'''Based on ground truth template 10.
Load simulated data.
Load processed b-values and b-vectors.
Use signals from selected b-values.
Get scipy nonlinear linear least squares results
by approximating matrix A that will be used to create 
the diffusion tensor as A.T @ A.
Option for bounds.
Use S0_correction.
Only keep symmetric positive definite tensors 
with eigenvalues below a threshold.
Save diffusion tensors, S0_corrections and valid indices.'''


import os
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from scipy.optimize import least_squares


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
class BaselineHyperparameters:
    threshold_eigval: float
    b_values_to_select: list[float]
    simulation_data_paths_pkl: str
    bounds: list[float]


class BaselinePaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('baseline', 'template_8', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')
        
        self.d_tensors_file = os.path.join(self.experiment_path, 'd_tensors.npy')
        self.S0_corrections_file = os.path.join(self.experiment_path, 'S0_corrections.npy')
        self.valid_indices_file = os.path.join(self.experiment_path, 'valid.npy')


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


def reconstruct(params: np.ndarray) -> tuple[np.ndarray, float]:

    A = params[:9].reshape(3, 3)

    D = A.T @ A

    S0_correction = params[9]
        
    return D, S0_correction


def loss(params, S_norm, g, b):
    D, S0_correction = reconstruct(params)
    S_norm_reconstructed = S0_correction * np.exp(- b * np.einsum('bi,ij,bj->b', g, D, g))
    error = S_norm_reconstructed - S_norm
    return error


def main():

    ## BASELINE PATHS

    baseline_paths = BaselinePaths()

    print(f'Experiment path: {baseline_paths.experiment_path}')

    os.makedirs(baseline_paths.experiment_path)

    with open(baseline_paths.paths_file, 'wb') as f:
        pickle.dump(baseline_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=baseline_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info('Baseline experiment:')
    logging.info(baseline_paths.experiment_path)
    logging.info('')


    ## BASELINE HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--threshold_eigval', type=float, required=True)
    parser.add_argument('--b_values_to_select', type=float, nargs='*')
    parser.add_argument('--simulation_data_paths_pkl', type=str, required=True)
    parser.add_argument('--bounds', type=float, nargs=2, default=[-10,10])
    args = parser.parse_args()

    baseline_hparams = BaselineHyperparameters(**vars(args))

    with open(baseline_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(baseline_hparams, f)
    
    logging.info('Baseline hyperparameters:')
    for key, value in vars(baseline_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## SIMULATION DATA PATHS

    with open(baseline_hparams.simulation_data_paths_pkl, 'rb') as f:
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
    

    ## DATA
        
    b_values: np.ndarray = np.load(proc_data_paths.b_values_file)
    b_vectors: np.ndarray = np.load(proc_data_paths.b_vectors_file)
    sim_data: np.ndarray = np.load(sim_data_paths.simulation_data_file)

    selection_mask, zero_mask = create_masks(baseline_hparams.b_values_to_select, b_values)


    ## NON LINEAR LEAST SQUARES
    
    min_val, max_val = baseline_hparams.bounds
    
    min_bounds = [min_val] * 9
    max_bounds = [max_val] * 9

    # S0_correction bounds
    min_bounds.append(0)
    max_bounds.append(2)

    d_tensors = []
    S0_corrections = []
    valid_indices = []
    invalid_count = 0
    bad_count = 0

    for row_index in tqdm(range(sim_data.shape[0])):

        signal = sim_data[row_index, :]

        S = signal[selection_mask]
        S0 = signal[zero_mask].mean()
        g = b_vectors[selection_mask, :]
        b = b_values[selection_mask]

        if S0 == 0.0:
            bad_count += 1
            continue
        
        S_norm = S / S0

        params = np.random.rand(9)
        params = np.append(params, 1.0) # S0_correction initial value

        result = least_squares(
            loss, 
            params, 
            args=(S_norm, g, b),
            bounds=(min_bounds, max_bounds)
        )

        D, S0_correction = reconstruct(result.x)
        
        # D should be symmetric positive definite
        try:
            L = np.linalg.cholesky(D)
        except:
            invalid_count += 1
            continue

        # the maximum eigenvalue of D should be lower than the threshold
        if np.max(np.linalg.eigvalsh(D)) > baseline_hparams.threshold_eigval:
            invalid_count += 1
            continue
        
        d_tensors.append(D)
        S0_corrections.append(S0_correction)
        valid_indices.append(row_index)

    d_tensors = np.stack(d_tensors, axis=0)
    S0_corrections = np.array(S0_corrections)
    valid_indices = np.array(valid_indices)

    np.save(baseline_paths.d_tensors_file, d_tensors)
    np.save(baseline_paths.S0_corrections_file, S0_corrections)
    np.save(baseline_paths.valid_indices_file, valid_indices)

    logging.info(f'Total brain voxels = {sim_data.shape[0]}')
    logging.info(f'Valid approximated d-tensors = {d_tensors.shape[0]}')
    logging.info(f'Invalid approximated d-tensors = {invalid_count}')
    logging.info(f'Bad signals = {bad_count}')


if __name__ == '__main__':
    main()
