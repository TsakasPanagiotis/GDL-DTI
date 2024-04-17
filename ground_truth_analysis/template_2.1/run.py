'''Load processed b-values, b-vectors and data.
Load ground truth d-tensors, S0 corrections and valid indices.
Use signals from selected b-values that include zero.
Calculate mean and standard deviation for errors, 
mean errors per voxel, mean absolute errors per voxel, 
and root mean squared errors per voxel.'''


import os
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm


class ProcessedDataPaths(Protocol):
    b_vectors_file: str
    b_values_file: str
    processed_data_file: str


class GroundTruthHyperparameters(Protocol):
    b_values_to_select: list[float]
    processed_data_paths_pkl: str


class GroundTruthPaths(Protocol):
    hyperparameters_file: str
    d_tensors_file: str
    S0_corrections_file: str
    valid_indices_file: str


@dataclass
class GroundTruthAnalysisHyperparameters:
    ground_truth_paths_pkl: str


class GroundTruthAnalysisPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('ground_truth_analysis', 'template_2.1', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')


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


def main():
    
    ## GROUND TRUTH ANALYSIS PATHS

    ground_truth_analysis_paths = GroundTruthAnalysisPaths()

    print(f'Experiment path: {ground_truth_analysis_paths.experiment_path}')

    os.makedirs(ground_truth_analysis_paths.experiment_path)

    with open(ground_truth_analysis_paths.paths_file, 'wb') as f:
        pickle.dump(ground_truth_analysis_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=ground_truth_analysis_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f'Ground truth analysis experiment:')
    logging.info(ground_truth_analysis_paths.experiment_path)
    logging.info('')
    

    ## GROUND TRUTH ANALYSIS HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--ground_truth_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    ground_truth_analysis_hparams = GroundTruthAnalysisHyperparameters(**vars(args))

    with open(ground_truth_analysis_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(ground_truth_analysis_hparams, f)    
    
    logging.info('Ground truth analysis hyperparameters:')
    for key, value in vars(ground_truth_analysis_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## GROUND TRUTH PATHS

    with open(ground_truth_analysis_hparams.ground_truth_paths_pkl, 'rb') as f:
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
    proc_data: np.ndarray = np.load(proc_data_paths.processed_data_file)

    d_tensors: np.ndarray = np.load(ground_truth_paths.d_tensors_file)
    S0_corrections: np.ndarray = np.load(ground_truth_paths.S0_corrections_file)
    valid_indices: np.ndarray = np.load(ground_truth_paths.valid_indices_file)

    selection_mask = get_selection_mask(ground_truth_hparams.b_values_to_select, b_values)
    zero_mask = b_values == 0.0


    ## ANALYSIS

    valid_data = proc_data[valid_indices]

    errors = []

    for row_index in tqdm(range(valid_data.shape[0])):
        
        signal = valid_data[row_index]

        S = signal[selection_mask]
        S0 = signal[zero_mask].mean()
        g = b_vectors[selection_mask, :]
        b = b_values[selection_mask]

        S_norm = S / S0

        D = d_tensors[row_index]
        S0_correction = S0_corrections[row_index]

        S_norm_reconstructed = S0_correction * np.exp(- b * np.einsum('bi,ij,bj->b', g, D, g))

        error = S_norm_reconstructed - S_norm

        errors.append(error)
    
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
