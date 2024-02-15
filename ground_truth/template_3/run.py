'''Go from raw data (nii, b-values, direction vectors)
to masked data (median_otsu brain voxels)
to masked signal range analysis per b-value.'''


import os
import logging
from datetime import datetime

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from argparse import ArgumentParser
from dipy.segment.mask import median_otsu


@dataclass
class Hyperparameters:
    
    b_zero_mean: bool


@dataclass
class Paths:
    experiments_dir = os.path.join('ground_truth', 'template_3', 'experiments')
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    nii_gz_path = 'C:/Users/panag/Desktop/Test/mgh_1001/diff/preproc/mri/diff_preproc.nii.gz'
    nii_numpy_file = 'nii_data.npy'
    
    b_values_path = 'C:/Users/panag/Desktop/Test/mgh_1001/diff/preproc/bvals.txt'

    @property
    def experiment_path(self):
        return os.path.join(self.experiments_dir, self.experiment_name)
    
    @property
    def log_file(self):
        return os.path.join(self.experiment_path, 'log.txt')
    
    @property
    def ranges_plot_file(self):
        return os.path.join(
            self.experiment_path,
            'masked signal intensity histograms per b-value.png')


def main():

    paths = Paths()


    ## HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--b_zero_mean', action='store_true')
    args = parser.parse_args()

    HP = Hyperparameters(**vars(args))


    ## LOGGING

    os.makedirs(paths.experiment_path)
    
    logging.basicConfig(
        filename=paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    for key, value in vars(HP).items():
        logging.info(f'{key}: {value}')
    
    
    ## NII DATA

    if os.path.exists(paths.nii_numpy_file):        
        nii_data = np.load(paths.nii_numpy_file)
    
    else:
        nii_data  = nib.load(paths.nii_gz_path).get_fdata() # type: ignore
        np.save(paths.nii_numpy_file, nii_data)

    
    ## B-VALUES
    
    b_values = np.genfromtxt(paths.b_values_path)
    b_values[(b_values > 9_900) & (b_values < 10_100)] = 10_000
    b_values /= 1_000.0

    b_value_to_indices = {b_value: np.where(b_values == b_value)[0] for b_value in np.unique(b_values)}
    
    
    ## NII MASKED DATA

    # use b-value = 0.0 for stronger brain signal to get the mask
    nii_data_masked, mask = median_otsu(nii_data, vol_idx=b_value_to_indices[0.0])

    
    ## MASKED SIGNAL RANGE ANALYSIS

    fig, ax = plt.subplots(1 + len(b_value_to_indices), 1, figsize=(10, 6), sharex=True)
    fig.suptitle('Masked Signal Intensity Histograms per b-value')

    for i, (b_value, indices) in enumerate(b_value_to_indices.items()):

        print(f'Processing b-value: {b_value}')
        
        S = nii_data_masked[:, :, :, indices]

        if b_value == 0.0 and HP.b_zero_mean:
            S = S.mean(axis=-1)
            b_value = '0.0 (mean)'
        
        min_val, max_val = np.min(S), np.max(S)    
        logging.info(f'b-value: {b_value} \t min: {min_val} \t max: {max_val}')    
        
        ax[i].hist(S[S != 0.0].flatten(), bins=1000, histtype='step', label=f'b-value: {b_value}')
        ax[i].legend()
        ax[i].yaxis.set_visible(False)
        ax[i].axvline(min_val, color='r', linestyle='--')
        ax[i].axvline(max_val, color='g', linestyle='--')


    min_val, max_val = np.min(nii_data_masked), np.max(nii_data_masked)
    logging.info(f'b-value: ALL \t min: {min_val} \t max: {max_val}')

    ax[-1].hist(nii_data_masked[nii_data_masked != 0.0].flatten(), bins=1000, histtype='step', label=f'b-value: ALL')
    ax[-1].legend()
    ax[-1].yaxis.set_visible(False)
    ax[-1].axvline(min_val, color='r', linestyle='--')
    ax[-1].axvline(max_val, color='g', linestyle='--')


    ax[-1].set_xlabel('Masked Signal Intensity')
    plt.savefig(paths.ranges_plot_file)


if __name__ == '__main__':
    main()
