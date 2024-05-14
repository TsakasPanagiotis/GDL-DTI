'''Load processed b-values and b-vectors.
Load simulated test data.
Use signals from selected b-values that include zero.
Get scipy nonlinear least squares results
by approximating matrix A that will be used 
to create the diffusion tensor as A.T @ A.
Option for bounds. Use S0_correction.
Save diffusion tensors and S0_corrections.
Measure reconstruction errors.'''


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
    d_tensors_file: str


class SimulationDataHyperparameters(Protocol):
    ground_truth_paths_pkl: str
    test_snrs: list[float]


class SimulationDataPaths(Protocol):
    hyperparameters_file: str
    test_data_template: str


@dataclass
class BaselineHyperparameters:
    seed: int
    snr: float
    b_values_to_select: list[float]
    simulation_data_paths_pkl: str
    bounds: list[float]


class BaselinePaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('baseline', 'template_8.2', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')
        
        self.pred_d_tensors_file = os.path.join(self.experiment_path, 'pred_d_tensors.npy')
        self.S0_corrections_file = os.path.join(self.experiment_path, 'S0_corrections.npy')


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


def reconstruct(params: np.ndarray) -> tuple[np.ndarray, float]:
    d_params, S0 = params[:-1], params[-1]
    A = d_params.reshape(3, 3)
    D = A.T @ A        
    return D, S0


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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snr', type=float, required=True)
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

    assert baseline_hparams.snr in sim_data_hparams.test_snrs, \
        f'Baseline SNR {baseline_hparams.snr} not in test SNRs {sim_data_hparams.test_snrs}'

    b_values: np.ndarray = np.load(proc_data_paths.b_values_file)
    b_vectors: np.ndarray = np.load(proc_data_paths.b_vectors_file)
    
    d_tensors: np.ndarray = np.load(ground_truth_paths.d_tensors_file)

    selection_mask = get_selection_mask(baseline_hparams.b_values_to_select, b_values)

    logging.info(f'Ground truth diffusion tensors shape: {d_tensors.shape}')
    logging.info('')


    ## REPRODUCIBILITY

    np.random.seed(baseline_hparams.seed)


    ## TEST DATA

    logging.info(f'Test SNR: {baseline_hparams.snr}')
    logging.info('')

    test_data = np.load(sim_data_paths.test_data_template.format(baseline_hparams.snr))

    logging.info(f'Test data shape: {test_data.shape}')
    logging.info('')


    ## NON LINEAR LEAST SQUARES

    num_d_params = 9

    min_val, max_val = baseline_hparams.bounds

    # d_params bounds
    min_bounds = [min_val] * num_d_params
    max_bounds = [max_val] * num_d_params

    # S0_correction bounds
    min_bounds.append(0)
    max_bounds.append(2)

    pred_d_tensors = []
    S0_corrections = []
    errors = []

    for row_index in tqdm(range(test_data.shape[0])):

        signal = test_data[row_index, :]

        S_norm = signal[selection_mask]
        g = b_vectors[selection_mask, :]
        b = b_values[selection_mask]

        # d_params initial values
        params = np.random.rand(num_d_params)

        # S0_correction initial value
        params = np.append(params, 1.0)

        result = least_squares(
            loss, 
            params, 
            args=(S_norm, g, b),
            bounds=(min_bounds, max_bounds)
        )

        D, S0_correction = reconstruct(result.x)

        S_norm_reconstructed = S0_correction * np.exp(- b * np.einsum('bi,ij,bj->b', g, D, g))
        
        pred_d_tensors.append(D)
        S0_corrections.append(S0_correction)
        errors.append(S_norm_reconstructed - S_norm)

    pred_d_tensors = np.stack(pred_d_tensors, axis=0)
    S0_corrections = np.array(S0_corrections)

    np.save(baseline_paths.pred_d_tensors_file, pred_d_tensors)
    np.save(baseline_paths.S0_corrections_file, S0_corrections)

    logging.info(f'Predicted diffusion tensors shape: {pred_d_tensors.shape}')
    logging.info('')

    all_errors = np.stack(errors, axis=0)
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
