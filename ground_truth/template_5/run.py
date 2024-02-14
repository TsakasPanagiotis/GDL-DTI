import os
import pickle
import logging
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from dipy.segment.mask import median_otsu


@dataclass
class Hyperparameters:
    threshold_eigval: float


@dataclass
class Paths:
    experiments_dir = os.path.join('ground_truth', 'template_5', 'experiments')
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
    def masked_plot_file(self):
        return os.path.join(self.experiment_path, 'masked.png')
    
    @property
    def lstsq_results_file(self):
        return os.path.join(self.experiment_path, 'lstsq_results.pkl')


def reconstruct_tensor(params: np.ndarray, threshold_eigval: float) -> np.ndarray:

    # Split parameters after applying sigmoid
    x_angle, y_angle, z_angle, eigval_1, eigval_2_over_1, eigval_3_over_2 = 1/(1 + np.exp(-params))

    # Compute angles
    x_angle = x_angle * 2 * np.pi
    y_angle = y_angle * 2 * np.pi
    z_angle = z_angle * 2 * np.pi

    # Compute eigenvalues
    eigval_1 = eigval_1 * threshold_eigval
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

    ## HYPERPARAMETERS

    parser = ArgumentParser()    
    parser.add_argument('--threshold_eigval', type=float, required=True)
    args = parser.parse_args()

    HP = Hyperparameters(**vars(args))


    ## LOGGING

    os.makedirs(paths.experiment_path)

    logging.basicConfig(
        filename=paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')


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

    b_value_to_indices = {b_value: np.where(b_values == b_value)[0] for b_value in np.unique(b_values)}


    ## NII MASKED DATA

    # use b-value = 0.0 for stronger brain signal to get the mask
    nii_data_masked, mask = median_otsu(nii_data, vol_idx=b_value_to_indices[0.0])

    # plot random slice
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(nii_data[:, :, 50, b_value_to_indices[1.0][0]], cmap='gray')
    ax[1].imshow(nii_data_masked[:, :, 50, b_value_to_indices[1.0][0]], cmap='gray')
    plt.savefig(paths.masked_plot_file)


    ## NON LINEAR LEAST SQUARES

    def loss(params, S, S0, g, b_value, threshold_eigval):
        D = reconstruct_tensor(params, threshold_eigval)
        return S0 * np.exp(- b_value * np.einsum('bi,ij,bj->b', g, D, g)) - S

    b_value = 1.0

    non_linear_lstsq_results = {}
    non_linear_errors = []

    # create a progress bar
    pbar = tqdm(total=mask.sum())

    # loop over unmasked voxels
    for i in range(nii_data.shape[0]):
        for j in range(nii_data.shape[1]):
            for k in range(nii_data.shape[2]):
                if mask[i,j,k]:
                    
                    # calculate y
                    S = nii_data[i, j, k, b_value_to_indices[b_value]]                    
                    S0 = nii_data[i, j, k, b_value_to_indices[0.0]].mean()

                    # calculate X
                    g = direction_vectors[b_value_to_indices[b_value], :]
                    
                    # randomly initialize params in [-1, 1]
                    params = np.random.rand(6) * 2 - 1

                    # minimize the loss function
                    result = least_squares(loss, params, args=(S, S0, g, b_value, HP.threshold_eigval))

                    # reconstruct tensor
                    D = reconstruct_tensor(result.x, HP.threshold_eigval)
                    
                    # try Cholesky decomposition
                    try:
                        _ = np.linalg.cholesky(D)
                    except:
                        pbar.update()
                        continue
                    
                    non_linear_lstsq_results[(i,j,k)] = D
                    
                    non_linear_errors.append( 
                        np.mean(
                            (S0 * np.exp(- b_value * np.einsum('bi,ij,bj->b', g, D, g)) - S)**2
                        )
                    )
                    
                    pbar.update()

    pbar.close()

    with open(paths.lstsq_results_file, 'wb') as f:
        pickle.dump(non_linear_lstsq_results, f)

    logging.info(f'Mean squared reconstruction error median = {np.median(non_linear_errors)}')
    logging.info(f'Mean squared reconstruction error mean = {np.mean(non_linear_errors)}')
    logging.info(f'Mean squared reconstruction error std = {np.std(non_linear_errors)}')
    logging.info(f'Mean squared reconstruction error max = {np.max(non_linear_errors)}')
    logging.info(f'Mean squared reconstruction error min = {np.min(non_linear_errors)}')

    logging.info(f'Total brain voxels = {np.sum(mask)}')
    logging.info(f'Valid approximated d-tensors = {len(non_linear_lstsq_results)}')
    logging.info(f'Invalid approximated d-tensors = {mask.sum() - len(non_linear_lstsq_results)}')


if __name__ == '__main__':
    main()
