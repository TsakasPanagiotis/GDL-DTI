'''Load raw data.
Load processed b-values, b-vectors, and median_otsu mask.
Use signals from selected b-values.
Get scipy nonlinear linear least squares results
by approximating the symmetric diffusion tensor.
Only keep symmetric positive definite tensors 
with eigenvalues below a threshold.
Save diffusion tensors and errors.'''


import os
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import nibabel as nib
from scipy.optimize import least_squares


class RawDataPaths(Protocol):
    raw_data_file: str


class ProcessedDataHyperparameters(Protocol):
    raw_data_paths_pkl: str


class ProcessedDataPaths(Protocol):
    hyperparameters_file: str
    b_vectors_file: str
    b_values_file: str
    mask_file: str


@dataclass
class GroundTruthHyperparameters:
    threshold_eigval: float
    b_values_to_select: list[float]
    processed_data_paths_pkl: str
    bounds: list[float]


class GroundTruthPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('ground_truth', 'template_2.2', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.results_file = os.path.join(self.experiment_path, 'results.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.errors_file = os.path.join(self.experiment_path, 'errors.pkl')


def create_masks(b_values_to_select_list: list[float], b_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    b_values_to_select = set(b_values_to_select_list)
    unique_nonzero_b_values = set(b_values) - {0.0}

    if len(b_values_to_select) == 0:
        b_values_to_select = unique_nonzero_b_values
        logging.warning(f'b_values_to_select is empty. Using all nonzero b-values: {b_values_to_select}')
    
    if 0.0 in b_values_to_select:
        logging.error('b_values_to_select must not contain 0.0')
        raise ValueError('b_values_to_select must not contain 0.0')

    if not b_values_to_select.issubset(unique_nonzero_b_values):
        logging.error(f'Invalid b_values_to_select: {b_values_to_select.difference(unique_nonzero_b_values)}. Valid values are: {unique_nonzero_b_values}')
        raise ValueError(f'Invalid b_values_to_select: {b_values_to_select.difference(unique_nonzero_b_values)}. Valid values are: {unique_nonzero_b_values}')

    selection_mask = np.isin(b_values, list(b_values_to_select))
    zero_mask = b_values == 0.0

    return selection_mask, zero_mask


def reconstruct(params: np.ndarray) -> np.ndarray:
    return np.array([[params[0], params[1], params[2]],
                     [params[1], params[3], params[4]],
                     [params[2], params[4], params[5]]])


def loss(params, S, S0, g, b):
    D = reconstruct(params)
    error = S0 * np.exp(- b * np.einsum('bi,ij,bj->b', g, D, g)) - S
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
    

    ## GROUND TRUTH HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--threshold_eigval', type=float, required=True)
    parser.add_argument('--b_values_to_select', type=float, nargs='+', required=True)
    parser.add_argument('--processed_data_paths_pkl', type=str, required=True)
    parser.add_argument('--bounds', type=float, nargs=2, default=[-np.inf,np.inf])
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
    

    ## PROCESSED DATA HYPERPARAMETERS

    with open(proc_data_paths.hyperparameters_file, 'rb') as f:
        proc_data_hparams: ProcessedDataHyperparameters = pickle.load(f)
    

    ## RAW DATA PATHS

    with open(proc_data_hparams.raw_data_paths_pkl, 'rb') as f:
        raw_data_paths: RawDataPaths = pickle.load(f)


    ## DATA
        
    raw_data  = nib.load(raw_data_paths.raw_data_file).get_fdata() # type: ignore
    b_values = np.load(proc_data_paths.b_values_file)
    b_vectors = np.load(proc_data_paths.b_vectors_file)
    mask = np.load(proc_data_paths.mask_file)


    ## NON LINEAR LEAST SQUARES

    selection_mask, zero_mask = create_masks(ground_truth_hparams.b_values_to_select, b_values)

    brain_voxels: int = mask.sum()
    pbar = tqdm(total=brain_voxels)

    invalid_count = 0
    results: dict[tuple[int,int,int], np.ndarray] = {}
    errors: dict[tuple[int,int,int], np.ndarray] = {}

    # loop over unmasked voxels
    for i in range(raw_data.shape[0]):
        for j in range(raw_data.shape[1]):
            for k in range(raw_data.shape[2]):
                if mask[i,j,k]:

                    S = raw_data[i, j, k, selection_mask]
                    S0 = raw_data[i, j, k, zero_mask].mean()
                    g = b_vectors[selection_mask, :]
                    b = b_values[selection_mask]
                    params = np.random.rand(6)

                    result = least_squares(
                        loss, 
                        params, 
                        args=(S, S0, g, b),
                        bounds=([ground_truth_hparams.bounds[0]]*len(params),
                                [ground_truth_hparams.bounds[1]]*len(params))
                    )

                    D = reconstruct(result.x)
                    
                    # D should be symmetric positive definite
                    try:
                        L = np.linalg.cholesky(D)
                    except:
                        invalid_count += 1
                        pbar.update()
                        continue

                    # the maximum eigenvalue of D should be lower than the threshold
                    if np.max(np.linalg.eigvalsh(D)) > ground_truth_hparams.threshold_eigval:
                        invalid_count += 1
                        pbar.update()
                        continue
                    
                    results[(i,j,k)] = D
                    
                    error = S0 * np.exp(- b * np.einsum('bi,ij,bj->b', g, D, g)) - S
                    errors[(i,j,k)] = error

                    pbar.update()

    pbar.close()

    with open(ground_truth_paths.results_file, 'wb') as f:
        pickle.dump(results, f)

    with open(ground_truth_paths.errors_file, 'wb') as f:
        pickle.dump(errors, f)

    logging.info(f'Total brain voxels = {brain_voxels}')
    logging.info(f'Valid approximated d-tensors = {len(results)}')
    logging.info(f'Invalid approximated d-tensors = {invalid_count}')
    logging.info('')


if __name__ == '__main__':
    main()
