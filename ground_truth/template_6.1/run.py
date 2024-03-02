'''Load raw data (nii, b-values, direction vectors).
Get masked data (brain voxels) with median_otsu filter.
Use signals from selected b-values.
Get scipy nonlinear linear least squares results using spectral composition: 
x_angle, y_angle, z_angle, eigval_1, eigval_2_over_1, eigval_3_over_2
and bounds to restrict the parameters.
Only keep symmetric positive definite tensors with eigenvalues below a threshold.
Measure the mean absolute reconstruction error.'''


import os
import pickle
import logging
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import nibabel as nib
from scipy.optimize import least_squares
from dipy.segment.mask import median_otsu


@dataclass
class Hyperparameters:

    threshold_eigval: float
    b_values_to_select: list[float]


@dataclass
class Paths:
    experiments_dir = os.path.join('ground_truth', 'template_6.1', 'experiments')
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    nii_gz_path = 'C:/Users/panag/Desktop/Test/mgh_1001/diff/preproc/mri/diff_preproc.nii.gz'
    nii_numpy_file = 'nii_data.npy'
    
    direction_vectors_path = 'C:/Users/panag/Desktop/Test/mgh_1001/diff/preproc/bvecs_moco_norm.txt'
    b_values_path = 'C:/Users/panag/Desktop/Test/mgh_1001/diff/preproc/bvals.txt'

    @property
    def experiment_path(self):
        return os.path.join(self.experiments_dir, self.experiment_name)
    
    @property
    def log_file(self):
        return os.path.join(self.experiment_path, 'log.txt')
    
    @property
    def lstsq_results_file(self):
        return os.path.join(self.experiment_path, 'lstsq_results.pkl')
    
    @property
    def hyperparameters_file(self):
        return os.path.join(self.experiment_path, 'hyperparameters.pkl')


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


def reconstruct_tensor(params: np.ndarray) -> np.ndarray:

    # Split parameters after applying sigmoid
    x_angle, y_angle, z_angle, eigval_1, eigval_2_over_1, eigval_3_over_2 = params

    # Compute eigenvalues
    eigval_1 = eigval_1
    eigval_2 = eigval_2_over_1 * eigval_1
    eigval_3 = eigval_3_over_2 * eigval_2

    # Create the roation matrices around the x axis.
    R_x = np.zeros((3, 3))
    R_x[0, 0] = 1
    R_x[1, 1] = np.cos(x_angle)
    R_x[1, 2] = -np.sin(x_angle)
    R_x[2, 1] = np.sin(x_angle)
    R_x[2, 2] = np.cos(x_angle)

    # Create the roation matrices around the y axis.
    R_y = np.zeros((3, 3))
    R_y[0, 0] = np.cos(y_angle)
    R_y[0, 2] = np.sin(y_angle)
    R_y[1, 1] = 1
    R_y[2, 0] = -np.sin(y_angle)
    R_y[2, 2] = np.cos(y_angle)

    # Create the roation matrices around the z axis.
    R_z = np.zeros((3, 3))
    R_z[0, 0] = np.cos(z_angle)
    R_z[0, 1] = -np.sin(z_angle)
    R_z[1, 0] = np.sin(z_angle)
    R_z[1, 1] = np.cos(z_angle)
    R_z[2, 2] = 1

    # Create the rotation matrix
    R = R_x @ R_y @ R_z

    # Create the diagonal matrix of eigenvalues
    D = np.diag([eigval_1, eigval_2, eigval_3])

    # Reconstruct the diffusion tensor
    D = R @ D @ R.T

    return D


def main():

    paths = Paths()

    os.makedirs(paths.experiment_path)
    
    
    ## HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--threshold_eigval', type=float, required=True)
    parser.add_argument('--b_values_to_select', type=float, nargs='+', required=True)
    args = parser.parse_args()

    HP = Hyperparameters(**vars(args))

    with open(paths.hyperparameters_file, 'wb') as f:
        pickle.dump(HP, f)


    ## LOGGING


    logging.basicConfig(
        filename=paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info('Hyperparameters:')
    for key, value in vars(HP).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## NII DATA

    if os.path.exists(paths.nii_numpy_file):        
        nii_data = np.load(paths.nii_numpy_file)

    else:
        nii_data  = nib.load(paths.nii_gz_path).get_fdata() # type: ignore
        np.save(paths.nii_numpy_file, nii_data)


    ## DIRECTION VECTORS

    direction_vectors = np.genfromtxt(paths.direction_vectors_path)


    ## B-VALUES

    b_values = np.genfromtxt(paths.b_values_path)
    b_values[(b_values > 9_900) & (b_values < 10_100)] = 10_000
    b_values /= 1_000.0


    ## NII MASKED DATA

    # use b-value = 0.0 for stronger brain signal to get the mask
    nii_data_masked, voxel_mask = median_otsu(nii_data, vol_idx=np.where(b_values == 0.0)[0])


    ## NON LINEAR LEAST SQUARES

    def loss(params, S, S0, g, b_value, threshold_eigval):
        D = reconstruct_tensor(params)
        return S0 * np.exp(- b_value * np.einsum('bi,ij,bj->b', g, D, g)) - S

    lstsq_results = {}
    mean_absolute_errors = []

    selection_mask, zero_mask = create_masks(HP.b_values_to_select, b_values)

    # create a progress bar
    pbar = tqdm(total=voxel_mask.sum())

    # loop over unmasked voxels
    for i in range(nii_data.shape[0]):
        for j in range(nii_data.shape[1]):
            for k in range(nii_data.shape[2]):
                if voxel_mask[i,j,k]:
                    
                    # calculate y
                    S = nii_data[i, j, k, selection_mask]                    
                    S0 = nii_data[i, j, k, zero_mask].mean()

                    # calculate X
                    g = direction_vectors[selection_mask, :]
                    
                    # randomly initialize params
                    params = np.random.rand(6)

                    # minimize the loss function
                    result = least_squares(
                        loss, 
                        params, 
                        args=(S, S0, g, b_values[selection_mask], HP.threshold_eigval),
                        bounds=(
                            [0,       0,       0,       0,                   0, 0], 
                            [2*np.pi, 2*np.pi, 2*np.pi, HP.threshold_eigval, 1, 1])
                    )

                    # reconstruct tensor
                    D = reconstruct_tensor(result.x)
                    
                    # D should be symmetric positive definite
                    try:
                        L = np.linalg.cholesky(D)
                    except:
                        pbar.update()
                        continue

                    # the maximum eigenvalue of D should be lower than the threshold
                    if np.max(np.linalg.eigvalsh(D)) > HP.threshold_eigval:
                        pbar.update()
                        continue
                    
                    lstsq_results[(i,j,k)] = D
                    
                    error = S0 * np.exp(- b_values[selection_mask] * np.einsum('bi,ij,bj->b', g, D, g)) - S
                    mean_absolute_errors.append(np.mean(np.abs(error)))
                    
                    pbar.update()

    pbar.close()

    with open(paths.lstsq_results_file, 'wb') as f:
        pickle.dump(lstsq_results, f)

    logging.info(f'Total brain voxels = {np.sum(voxel_mask)}')
    logging.info(f'Valid approximated d-tensors = {len(lstsq_results)}')
    logging.info(f'Invalid approximated d-tensors = {voxel_mask.sum() - len(lstsq_results)}')
    logging.info('')
    logging.info(f'Mean Absolute reconstruction Error median = {np.median(mean_absolute_errors)}')
    logging.info(f'Mean Absolute reconstruction Error mean = {np.mean(mean_absolute_errors)}')
    logging.info(f'Mean Absolute reconstruction Error std = {np.std(mean_absolute_errors)}')
    logging.info(f'Mean Absolute reconstruction Error max = {np.max(mean_absolute_errors)}')
    logging.info(f'Mean Absolute reconstruction Error min = {np.min(mean_absolute_errors)}')


if __name__ == '__main__':
    main()
