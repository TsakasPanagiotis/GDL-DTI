'''Load raw data.
Load processed b-values and median_otsu mask.
Analyze masked data range per b-value in histograms.
Option to zero-mean b=0.0 images.
Save histograms.'''


import os
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


class RawDataPaths(Protocol):
    raw_data_file: str


class ProcessedDataHyperparameters(Protocol):
    raw_data_paths_pkl: str


class ProcessedDataPaths(Protocol):
    hyperparameters_file: str
    b_values_file: str
    mask_file: str


@dataclass
class DataAnalysisHyperparameters:  
    b_zero_mean: bool
    processed_data_paths_pkl: str


class DataAnalysisPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('raw_data_analysis', 'template_1.1', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')    
        self.histograms_file = os.path.join(self.experiment_path, 'histograms.png')


def main():

    ## DATA ANALYSIS PATHS

    data_analysis_paths = DataAnalysisPaths()

    print(f'Experiment path: {data_analysis_paths.experiment_path}')

    os.makedirs(data_analysis_paths.experiment_path)
    
    with open(data_analysis_paths.paths_file, 'wb') as f:
        pickle.dump(data_analysis_paths, f)


    ## LOGGING
    
    logging.basicConfig(
        filename=data_analysis_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')


    ## HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--b_zero_mean', action='store_true')
    parser.add_argument('--processed_data_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    data_analysis_hparams = DataAnalysisHyperparameters(**vars(args))

    with open(data_analysis_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(data_analysis_hparams, f)
    
    logging.info('Data analysis hyperparameters:')
    for key, value in vars(data_analysis_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## PROCESSED DATA PATHS

    with open(data_analysis_hparams.processed_data_paths_pkl, 'rb') as f:
        proc_data_paths: ProcessedDataPaths = pickle.load(f)


    ## PROCESSED DATA HYPERPARAMETERS

    with open(proc_data_paths.hyperparameters_file, 'rb') as f:
        proc_data_hparams: ProcessedDataHyperparameters = pickle.load(f)


    ## RAW DATA PATHS

    with open(proc_data_hparams.raw_data_paths_pkl, 'rb') as f:
        raw_data_paths: RawDataPaths = pickle.load(f)
    
    
    ## RAW DATA
        
    raw_data  = nib.load(raw_data_paths.raw_data_file).get_fdata() # type: ignore

    
    ## B-VALUES
    
    b_values = np.load(proc_data_paths.b_values_file)

    b_value_to_indices = {b_value: np.where(b_values == b_value)[0] for b_value in np.unique(b_values)}
    
    
    ## MASKED DATA

    mask = np.load(proc_data_paths.mask_file)

    masked_data = raw_data * mask[..., None]

    
    ## MASKED SIGNAL RANGE ANALYSIS

    fig, ax = plt.subplots(1 + len(b_value_to_indices), 1, figsize=(10, 6), sharex=True)
    fig.suptitle('Masked Signal Intensity Histograms per b-value')

    for i, (b_value, indices) in enumerate(b_value_to_indices.items()):

        print(f'Processing b-value: {b_value}')
        
        S = masked_data[:, :, :, indices]

        if b_value == 0.0 and data_analysis_hparams.b_zero_mean:
            S = S.mean(axis=-1)
            b_value = '0.0 (mean)'
        
        min_val, max_val = np.min(S), np.max(S)    
        logging.info(f'b-value: {b_value} \t min: {min_val} \t max: {max_val}')    
        
        ax[i].hist(S[S != 0.0].flatten(), bins=1000, histtype='step', label=f'b-value: {b_value}')
        ax[i].legend()
        ax[i].yaxis.set_visible(False)
        ax[i].axvline(min_val, color='r', linestyle='--')
        ax[i].axvline(max_val, color='g', linestyle='--')

    min_val, max_val = np.min(masked_data), np.max(masked_data)
    logging.info(f'b-value: ALL \t min: {min_val} \t max: {max_val}')

    ax[-1].hist(masked_data[masked_data != 0.0].flatten(), bins=1000, histtype='step', label=f'b-value: ALL')
    ax[-1].legend()
    ax[-1].yaxis.set_visible(False)
    ax[-1].axvline(min_val, color='r', linestyle='--')
    ax[-1].axvline(max_val, color='g', linestyle='--')

    ax[-1].set_xlabel('Masked Signal Intensity')
    plt.savefig(data_analysis_paths.histograms_file)


if __name__ == '__main__':
    main()
