'''Basic processing of raw data.
Store b-values, b-vectors, and median_otsu mask.'''


import os
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu


class RawDataPaths(Protocol):
    b_values_file: str
    b_vectors_file: str
    raw_data_file: str


@dataclass
class ProcessedDataHyperparameters:
    raw_data_paths_pkl: str


class ProcessedDataPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('processed_data', 'template_1', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')
        self.b_values_file = os.path.join(self.experiment_path, 'bvals.npy')
        self.b_vectors_file = os.path.join(self.experiment_path, 'bvecs.npy')
        self.mask_file = os.path.join(self.experiment_path, 'mask.npy')


def main():

    ## PROCESSED DATA PATHS
    
    proc_data_paths = ProcessedDataPaths()

    print(f'Experiment path: {proc_data_paths.experiment_path}')

    os.makedirs(proc_data_paths.experiment_path)
    
    with open(proc_data_paths.paths_file, 'wb') as f:
        pickle.dump(proc_data_paths, f)


    ## LOGGING

    logging.basicConfig(
        filename=proc_data_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')


    ## PROCESSED DATA HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--raw_data_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    proc_data_hparams = ProcessedDataHyperparameters(**vars(args))

    with open(proc_data_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(proc_data_hparams, f)
    
    logging.info('Processed data hyperparameters:')
    for key, value in vars(proc_data_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## RAW DATA PATHS

    with open(proc_data_hparams.raw_data_paths_pkl, 'rb') as f:
        raw_data_paths: RawDataPaths = pickle.load(f)


    ## B-VECTORS

    b_vectors = np.genfromtxt(raw_data_paths.b_vectors_file)

    np.save(proc_data_paths.b_vectors_file, b_vectors)


    ## B-VALUES

    b_values = np.genfromtxt(raw_data_paths.b_values_file)
    
    b_values[(b_values > 9_900) & (b_values < 10_100)] = 10_000
    b_values /= 1_000.0

    np.save(proc_data_paths.b_values_file, b_values)


    ## DIFF DATA
    
    raw_data  = nib.load(raw_data_paths.raw_data_file).get_fdata() # type: ignore
    
    # use b-value = 0.0 for stronger brain signal to get the mask
    masked_data, mask = median_otsu(raw_data, vol_idx=np.where(b_values == 0.0)[0])

    np.save(proc_data_paths.mask_file, mask)


if __name__ == '__main__':
    main()
