'''Go from raw data (nii, b-values, direction vectors)
to masked data (median_otsu brain voxels)
to numpy linear least squares results.'''


import os
import pickle
import logging
from datetime import datetime

import numpy as np
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.segment.mask import median_otsu
from dataclasses import dataclass


@dataclass
class Paths:
    experiments_dir = os.path.join('ground_truth', 'template_1', 'experiments')
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


def main():

    paths = Paths()


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

    
    ## LINEAR LEAST SQUARES

    b_value = 1.0
    epsilon = 1e-8

    linear_lstsq_results = {}
    linear_errors = []

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

                    # avoid division by zero or negative values
                    if abs(S0) < epsilon or np.any(S / S0 < epsilon):
                        pbar.update()
                        continue
                    
                    y = np.log(S / S0)
                    
                    # calculate X
                    g = direction_vectors[b_value_to_indices[b_value], :]
                    X = - b_value * np.array([g[:,0]**2, 2*g[:,0]*g[:,1], 2*g[:,0]*g[:,2], 
                                            g[:,1]**2, 2*g[:,1]*g[:,2], g[:,2]**2]).T
                    
                    # calculate lstsq
                    params = np.linalg.lstsq(X, y, rcond=None)[0]
                    
                    # reconstruct tensor
                    D = np.array([[params[0], params[1], params[2]],
                                [params[1], params[3], params[4]],
                                [params[2], params[4], params[5]]])
                    
                    # try Cholesky decomposition
                    try:
                        _ = np.linalg.cholesky(D)
                    except:
                        pbar.update()
                        continue

                    linear_lstsq_results[(i,j,k)] = D                    
                    linear_errors.append( 
                        np.mean(
                            (S0 * np.exp(X @ params) - S)**2
                        )
                    )                    
                    pbar.update()

    pbar.close()

    with open(paths.lstsq_results_file, 'wb') as f:
        pickle.dump(linear_lstsq_results, f)

    logging.info(f'Mean squared reconstruction error median = {np.median(linear_errors)}')
    logging.info(f'Mean squared reconstruction error mean = {np.mean(linear_errors)}')
    logging.info(f'Mean squared reconstruction error std = {np.std(linear_errors)}')
    logging.info(f'Mean squared reconstruction error max = {np.max(linear_errors)}')
    logging.info(f'Mean squared reconstruction error min = {np.min(linear_errors)}')

    logging.info(f'Total brain voxels = {np.sum(mask)}')
    logging.info(f'Valid approximated d-tensors = {len(linear_lstsq_results)}')
    logging.info(f'Invalid approximated d-tensors = {mask.sum() - len(linear_lstsq_results)}')


if __name__ == '__main__':
    main()
