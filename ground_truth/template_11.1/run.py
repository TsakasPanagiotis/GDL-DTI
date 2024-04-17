'''Load processed b-values, b-vectors and data.
Use signals from selected b-values that include zero.
Normalize signals by mean S0 signal.
Get scipy nonlinear least squares results
by approximating the spectral composition parameters:
rotation matrix through quaternion representation,
eigval_1, eigval_2_over_1, eigval_3_over_2.
Quaternion q = a + bi + cj + dk is unit length.
Use sigmoid to constraint the eigenvalue parameters.
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
    b_vectors_file: str
    b_values_file: str
    processed_data_file: str


@dataclass
class GroundTruthHyperparameters:
    seed: int
    threshold_eigval: float
    b_values_to_select: list[float]
    processed_data_paths_pkl: str


class GroundTruthPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('ground_truth', 'template_11.1', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')
        
        self.d_tensors_file = os.path.join(self.experiment_path, 'd_tensors.npy')
        self.S0_corrections_file = os.path.join(self.experiment_path, 'S0_corrections.npy')
        self.valid_indices_file = os.path.join(self.experiment_path, 'valid.npy')


def get_selection_mask(b_values_to_select_list: list[float], b_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))


def reconstruct(params: np.ndarray, threshold_eigval: float) -> tuple[np.ndarray, float]:

    d_params, S0 = params[:-1], params[-1]

    a, b, c, d, eigval_1, eigval_2_over_1, eigval_3_over_2 = d_params[:7]

    norm = np.sqrt(a**2 + b**2 + c**2 + d**2)
    a = a / norm
    b = b / norm
    c = c / norm
    d = d / norm

    eigval_1 = sigmoid(eigval_1)
    eigval_1 = eigval_1 * threshold_eigval

    eigval_2_over_1 = sigmoid(eigval_2_over_1)
    eigval_2 = eigval_2_over_1 * eigval_1

    eigval_3_over_2 = sigmoid(eigval_3_over_2)
    eigval_3 = eigval_3_over_2 * eigval_2

    # Create the rotation matrix from the quaternion q = a + bi + cj + dk
    R = np.array([[1 - 2*(c**2 + d**2), 2*(b*c - a*d), 2*(a*c + b*d)],
                  [2*(b*c + a*d), 1 - 2*(b**2 + d**2), 2*(c*d - a*b)],
                  [2*(b*d - a*c), 2*(a*b + c*d), 1 - 2*(b**2 + c**2)]])

    # Create the diagonal matrix of eigenvalues
    E = np.diag([eigval_1, eigval_2, eigval_3])

    # Reconstruct the diffusion tensor
    D = R @ E @ R.T

    return D, S0


def loss(params, S_norm, g, b, threshold_eigval):
    D, S0_correction = reconstruct(params, threshold_eigval)
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threshold_eigval', type=float, required=True)
    parser.add_argument('--b_values_to_select', type=float, nargs='*')
    parser.add_argument('--processed_data_paths_pkl', type=str, required=True)
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

    selection_mask = get_selection_mask(ground_truth_hparams.b_values_to_select, b_values)
    zero_mask = b_values == 0.0


    ## REPRODUCIBILITY

    np.random.seed(ground_truth_hparams.seed)


    ## NON LINEAR LEAST SQUARES

    num_d_params = 7

    # d_params bounds
    min_bounds = [-np.inf] * num_d_params
    max_bounds = [ np.inf] * num_d_params

    # S0_correction bounds
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

        # d_params initial values
        params = np.random.rand(num_d_params)

        # S0_correction initial value
        params = np.append(params, 1.0)

        result = least_squares(
            loss, 
            params, 
            args=(S_norm, g, b, ground_truth_hparams.threshold_eigval),
            bounds=(min_bounds, max_bounds)
        )

        D, S0_correction = reconstruct(result.x, ground_truth_hparams.threshold_eigval)

        symmetric_mask = np.allclose(D, D.T)
        eigenvalues = np.linalg.eigvals(D).real
        positive_mask = np.all(eigenvalues > 0)
        theshold_mask = np.all(eigenvalues < ground_truth_hparams.threshold_eigval)

        if not (symmetric_mask & positive_mask & theshold_mask):
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
