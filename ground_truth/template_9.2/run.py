'''Load raw data.
Load processed b-values, b-vectors, and median_otsu mask.
Use signals from selected b-values.
Get scipy nonlinear linear least squares results
by approximating the spectral composition parameters:
x_angle, y_angle, z_angle,
eigval_1, eigval_2_over_1, eigval_3_over_2.
Use cos and sin functions for cosines and sines.
Use sigmoid to constraint the eigenvalue parameters.
Only keep symmetric positive definite tensors 
with eigenvalues below a threshold.
Save eigenvectors, eigenvalues, diffusion tensors and errors.'''


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


class GroundTruthPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('ground_truth', 'template_9.2', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.d_tensors_file = os.path.join(self.experiment_path, 'd_tensors.pkl')
        self.eig_pairs_file = os.path.join(self.experiment_path, 'eig_pairs.pkl')
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


def reconstruct(params: np.ndarray, threshold_eigval: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    x_angle, y_angle, z_angle, eigval_1, eigval_2_over_1, eigval_3_over_2 = params

    x_cos = np.cos(x_angle)
    x_sin = np.sin(x_angle)
    
    y_cos = np.cos(y_angle)
    y_sin = np.sin(y_angle)
    
    # y angle should cover pi radians
    # angles in [-pi/2, pi/2] have cos(y) > 0
    y_cos = np.abs(y_cos)
    
    z_cos = np.cos(z_angle)
    z_sin = np.sin(z_angle)

    eigval_1 = 1/(1 + np.exp(-eigval_1))
    eigval_1 = eigval_1 * threshold_eigval

    eigval_2_over_1 = 1/(1 + np.exp(-eigval_2_over_1))
    eigval_2 = eigval_2_over_1 * eigval_1

    eigval_3_over_2 = 1/(1 + np.exp(-eigval_3_over_2))
    eigval_3 = eigval_3_over_2 * eigval_2

    # Create the roation matrices around the x axis.
    R_x = np.zeros((3, 3))
    R_x[0, 0] = 1
    R_x[1, 1] = x_cos
    R_x[1, 2] = -x_sin
    R_x[2, 1] = x_sin
    R_x[2, 2] = x_cos

    # Create the roation matrices around the y axis.
    R_y = np.zeros((3, 3))
    R_y[0, 0] = y_cos
    R_y[0, 2] = y_sin
    R_y[1, 1] = 1
    R_y[2, 0] = -y_sin
    R_y[2, 2] = y_cos

    # Create the roation matrices around the z axis.
    R_z = np.zeros((3, 3))
    R_z[0, 0] = z_cos
    R_z[0, 1] = -z_sin
    R_z[1, 0] = z_sin
    R_z[1, 1] = z_cos
    R_z[2, 2] = 1

    # Create the rotation matrix
    R = R_z @ R_y @ R_x

    # Create the diagonal matrix of eigenvalues
    E = np.diag([eigval_1, eigval_2, eigval_3])

    # Reconstruct the diffusion tensor
    D = R @ E @ R.T

    return R, E, D


def loss(params, S, S0, g, b, threshold_eigval):
    R, E, D = reconstruct(params, threshold_eigval)
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
    
    logging.info('Ground truth experiment:')
    logging.info(ground_truth_paths.experiment_path)
    logging.info('')


    ## GROUND TRUTH HYPERPARAMETERS

    parser = ArgumentParser()
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
    d_tensors: dict[tuple[int,int,int], np.ndarray] = {}
    eig_pairs: dict[tuple[int,int,int], tuple[np.ndarray,np.ndarray]] = {}
    errors: dict[tuple[int,int,int], np.ndarray] = {}

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
                        args=(S, S0, g, b, ground_truth_hparams.threshold_eigval)
                    )

                    R, E, D = reconstruct(result.x, ground_truth_hparams.threshold_eigval)
                    
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
                    
                    d_tensors[(i,j,k)] = D
                    eig_pairs[(i,j,k)] = (R, E)
                    
                    error = S0 * np.exp(- b * np.einsum('bi,ij,bj->b', g, D, g)) - S
                    errors[(i,j,k)] = error

                    pbar.update()

    pbar.close()
                    
    with open(ground_truth_paths.d_tensors_file, 'wb') as f:
        pickle.dump(d_tensors, f)

    with open(ground_truth_paths.eig_pairs_file, 'wb') as f:
        pickle.dump(eig_pairs, f)

    with open(ground_truth_paths.errors_file, 'wb') as f:
        pickle.dump(errors, f)

    logging.info(f'Total brain voxels = {brain_voxels}')
    logging.info(f'Valid approximated d-tensors = {len(d_tensors)}')
    logging.info(f'Invalid approximated d-tensors = {invalid_count}')


if __name__ == '__main__':
    main()
