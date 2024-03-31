'''Load processed b-values, b-vectors and data.
Use signals from selected b-values.
Get scipy nonlinear linear least squares results
by approximating the lower triangular matrix 
of the Cholesky decomposition of the diffusion tensor.
Option for bounds and S0_correction.
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
    b_vectors_file: str
    b_values_file: str
    processed_data_file: str


@dataclass
class GroundTruthHyperparameters:
    threshold_eigval: float
    b_values_to_select: list[float]
    processed_data_paths_pkl: str
    bounds: list[float]
    S0_correction: bool


class GroundTruthPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('ground_truth', 'template_8', 'experiments', 
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

    lower_triangular = np.zeros((3,3))
    lower_triangular[0,0] = params[0]
    lower_triangular[1,0] = params[1]
    lower_triangular[1,1] = params[2]
    lower_triangular[2,0] = params[3]
    lower_triangular[2,1] = params[4]
    lower_triangular[2,2] = params[5]
    D = lower_triangular @ lower_triangular.T

    S0_correction = 1.0

    if len(params) == 7:
        S0_correction = params[6]
        
    return D, S0_correction


def loss(params, S_norm, g, b):
    D, S0_correction = reconstruct(params)
    S_norm_reconstructed = S0_correction * np.exp(- b * np.einsum('bi,ij,bj->b', g, D, g))
    error = S_norm_reconstructed - S_norm
    return error


def main():

    ## GROUND TRUTH PATHS

    ground_truth_paths = GroundTruthPaths()

    print(f'Experiment path: {ground_truth_paths.experiment_path}')

    os.makedirs(ground_truth_paths.experiment_path)

    with open(ground_truth_paths.paths_file, 'wb') as f:
        pickle.dump(ground_truth_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=ground_truth_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info('Ground truth experiment:')
    logging.info(ground_truth_paths.experiment_path)
    logging.info('')


    ## GROUND TRUTH HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--threshold_eigval', type=float, required=True)
    parser.add_argument('--b_values_to_select', type=float, nargs='*')
    parser.add_argument('--processed_data_paths_pkl', type=str, required=True)
    parser.add_argument('--bounds', type=float, nargs=2, default=[-10,10])
    parser.add_argument('--S0_correction', action='store_true')
    args = parser.parse_args()

    ground_truth_hparams = GroundTruthHyperparameters(**vars(args))

    with open(ground_truth_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(ground_truth_hparams, f)    
    
    logging.info('Ground truth hyperparameters:')
    for key, value in vars(ground_truth_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## PROCESSED DATA PATHS

    with open(ground_truth_hparams.processed_data_paths_pkl, 'rb') as f:
        proc_data_paths: ProcessedDataPaths = pickle.load(f)


    ## DATA
    
    b_values: np.ndarray = np.load(proc_data_paths.b_values_file)
    b_vectors: np.ndarray = np.load(proc_data_paths.b_vectors_file)
    proc_data: np.ndarray = np.load(proc_data_paths.processed_data_file)

    selection_mask, zero_mask = create_masks(ground_truth_hparams.b_values_to_select, b_values)


    ## NON LINEAR LEAST SQUARES
    
    min_val, max_val = ground_truth_hparams.bounds
    
    # diagonal elements of lower triangular matrix must be positive
    #              [0,0] -  [1,0] -  [1,1] -  [2,0] -  [2,1] -  [2,2]
    min_bounds = [      0, min_val,       0, min_val, min_val,       0]
    max_bounds = [max_val, max_val, max_val, max_val, max_val, max_val]

    if ground_truth_hparams.S0_correction:
        min_bounds.append(0)
        max_bounds.append(2)

    d_tensors = []
    S0_corrections = []
    valid_indices = []
    invalid_count = 0
    bad_count = 0

    for row_index in tqdm(range(proc_data.shape[0])):

        signal = proc_data[row_index, :]

        S = signal[selection_mask]
        S0 = signal[zero_mask].mean()
        g = b_vectors[selection_mask, :]
        b = b_values[selection_mask]

        if S0 == 0.0:
            bad_count += 1
            continue
        
        S_norm = S / S0

        params = np.random.rand(6)

        if ground_truth_hparams.S0_correction:
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
        if np.max(np.linalg.eigvalsh(D)) > ground_truth_hparams.threshold_eigval:
            invalid_count += 1
            continue
        
        d_tensors.append(D)
        S0_corrections.append(S0_correction)
        valid_indices.append(row_index)

    d_tensors = np.stack(d_tensors, axis=0)
    S0_corrections = np.array(S0_corrections)
    valid_indices = np.array(valid_indices)

    np.save(ground_truth_paths.d_tensors_file, d_tensors)
    np.save(ground_truth_paths.S0_corrections_file, S0_corrections)
    np.save(ground_truth_paths.valid_indices_file, valid_indices)

    logging.info(f'Total brain voxels = {proc_data.shape[0]}')
    logging.info(f'Valid approximated d-tensors = {d_tensors.shape[0]}')
    logging.info(f'Invalid approximated d-tensors = {invalid_count}')
    logging.info(f'Bad signals = {bad_count}')


if __name__ == '__main__':
    main()
