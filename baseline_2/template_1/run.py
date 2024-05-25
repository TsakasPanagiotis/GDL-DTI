'''Load simulated b-values and b-vectors.
Load simulated test data for given snr.
Use signals from selected b-values that include zero.
Get scipy nonlinear least squares results
by approximating D = A.T @ A and D* = A*.T @ A*.
Save D, D*, f-values and S0_corrections.
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


class SimulationDataHyperparameters(Protocol):
    test_snrs: list[float]


class SimulationDataPaths(Protocol):
    hyperparameters_file: str
    new_b_values_file: str
    new_b_vectors_file: str
    test_data_template: str


@dataclass
class BaselineHyperparameters:
    seed: int
    snr: float
    b_values_to_select: list[float]
    simulation_data_paths_pkl: str


class BaselinePaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('baseline_2', 'template_1', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')
        
        self.pred_d_tensors_file = os.path.join(self.experiment_path, 'pred_d_tensors.npy')
        self.pred_d_star_tensors_file = os.path.join(self.experiment_path, 'pred_d_star_tensors.npy')
        self.pred_f_values_file = os.path.join(self.experiment_path, 'pred_f_values.npy')
        self.pred_S0_corrections_file = os.path.join(self.experiment_path, 'pred_S0_corrections.npy')


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


def reconstruct(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    d_params, d_star_params, f, S0 = params[0:9], params[9:18], params[18], params[19]
    
    A = d_params.reshape(3, 3)
    D = A.T @ A
    
    A_star = d_star_params.reshape(3, 3)
    D_star = A_star.T @ A_star
    
    return D, D_star, f, S0


def loss(params, S_norm, g, b):
    D, D_star, f, S0 = reconstruct(params)
    S_norm_reconstructed = S0 * ( 
        f * np.exp(- b * np.einsum('bi,ij,bj->b', g, D_star, g)) \
        + (1-f) * np.exp(- b * np.einsum('bi,ij,bj->b', g, D, g)) )
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
    parser.add_argument('--b_values_to_select', type=float, nargs='*', default=[])
    parser.add_argument('--simulation_data_paths_pkl', type=str, required=True)
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


    ## DATA

    assert baseline_hparams.snr in sim_data_hparams.test_snrs, \
        f'SNR {baseline_hparams.snr} not in test SNRs: {sim_data_hparams.test_snrs}'

    b_values: np.ndarray = np.load(sim_data_paths.new_b_values_file)
    b_vectors: np.ndarray = np.load(sim_data_paths.new_b_vectors_file)

    selection_mask = get_selection_mask(baseline_hparams.b_values_to_select, b_values)


    ## TEST DATA

    test_data: np.ndarray = np.load(sim_data_paths.test_data_template.format(int(baseline_hparams.snr)))

    logging.info(f'Test data shape: {test_data.shape}')
    logging.info('')


    ## REPRODUCIBILITY

    np.random.seed(baseline_hparams.seed)


    ## NON LINEAR LEAST SQUARES

    num_d_params = 9
    num_d_star_params = 9
    
    # d_params bounds
    min_bounds = [-5.0] * num_d_params
    max_bounds = [10.0] * num_d_params

    # d_star_params bounds
    min_bounds += [-50.0] * num_d_star_params
    max_bounds += [100.0] * num_d_star_params

    # f bounds
    min_bounds.append(0.05)
    max_bounds.append(0.7) # 0.3

    # S0_correction bounds
    min_bounds.append(0)
    max_bounds.append(2)

    logging.info('Bounds:')
    logging.info(f'min: {min_bounds}')
    logging.info(f'max: {max_bounds}')
    logging.info('')

    pred_d_tensors = []
    pred_d_star_tensors = []
    pred_f_values = []
    pred_S0_corrections = []
    errors = []

    for row_index in tqdm(range(test_data.shape[0])):

        signal = test_data[row_index, :]

        S_norm = signal[selection_mask]
        g = b_vectors[selection_mask, :]
        b = b_values[selection_mask]

        # d_params initial values for D and D*
        params = np.random.rand(num_d_params + num_d_star_params)

        # f initial value
        params = np.append(params, 0.15)

        # S0_correction initial value
        params = np.append(params, 1.0)

        result = least_squares(
            loss, 
            params, 
            args=(S_norm, g, b),
            bounds=(min_bounds, max_bounds)
        )

        D, D_star, f, S0_correction = reconstruct(result.x)
        
        S_norm_reconstructed = S0_correction * (
            f * np.exp(- b * np.einsum('bi, ij, bj -> b', g, D_star, g)) \
            + (1-f) * np.exp(- b * np.einsum('bi, ij, bj -> b', g, D, g)) )
        
        pred_d_tensors.append(D)
        pred_d_star_tensors.append(D_star)
        pred_f_values.append(f)
        pred_S0_corrections.append(S0_correction)
        errors.append(S_norm_reconstructed - S_norm)

    pred_d_tensors = np.stack(pred_d_tensors, axis=0)
    pred_d_star_tensors = np.stack(pred_d_star_tensors, axis=0)
    pred_f_values = np.array(pred_f_values)
    pred_S0_corrections = np.array(pred_S0_corrections)

    np.save(baseline_paths.pred_d_tensors_file, pred_d_tensors)
    np.save(baseline_paths.pred_d_star_tensors_file, pred_d_star_tensors)
    np.save(baseline_paths.pred_f_values_file, pred_f_values)
    np.save(baseline_paths.pred_S0_corrections_file, pred_S0_corrections)

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
