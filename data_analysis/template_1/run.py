'''Load processed b-values and data.
Analyze data per b-value and for all b-values combined:
min, max, num_zeros, num_neg, total.
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
import matplotlib.pyplot as plt


class ProcessedDataPaths(Protocol):
    b_values_file: str
    processed_data_file: str


@dataclass
class DataAnalysisHyperparameters:  
    b_zero_mean: bool
    processed_data_paths_pkl: str


class DataAnalysisPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('data_analysis', 'template_1', 'experiments', 
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

    
    ## B-VALUES
    
    b_values: np.ndarray = np.load(proc_data_paths.b_values_file)

    b_value_to_indices: dict[float, np.ndarray] = {b_value: np.where(b_values == b_value)[0] for b_value in np.unique(b_values)}
    
    
    ## PROCESSED DATA

    processed_data: np.ndarray = np.load(proc_data_paths.processed_data_file)

    
    ## MASKED SIGNAL RANGE ANALYSIS

    fig, ax = plt.subplots(1 + len(b_value_to_indices), 1, figsize=(10, 6), sharex=True)
    fig.suptitle('Signal Intensity Histograms per b-value')

    for i, (b_value, indices) in enumerate(b_value_to_indices.items()):

        print(f'Processing b-value: {b_value}')
        
        S = processed_data[:, indices]

        if b_value == 0.0 and data_analysis_hparams.b_zero_mean:
            S = S.mean(axis=1)
            b_value = '0.0 (mean)'
        
        min_val, max_val, num_zeros, num_neg = np.min(S), np.max(S), np.sum(S == 0.0), np.sum(S < 0.0)
        logging.info(f'b-value: {b_value:<10} min: {min_val:<10.1f} max: {max_val:<10.1f} num_zeros: {num_zeros:<10} num_neg: {num_neg:<10} total: {S.size:<10}')
        
        ax[i].hist(S.flatten(), bins=1000, histtype='step', label=f'b-value: {b_value}')
        ax[i].legend()
        ax[i].yaxis.set_visible(False)
        ax[i].axvline(min_val, color='r', linestyle='--')
        ax[i].axvline(max_val, color='g', linestyle='--')

    min_val, max_val, num_zeros, num_neg = np.min(processed_data), np.max(processed_data), np.sum(processed_data == 0.0), np.sum(processed_data < 0.0)
    logging.info(f'b-value: {"ALL":<10} min: {min_val:<10.1f} max: {max_val:<10.1f} num_zeros: {num_zeros:<10} num_neg: {num_neg:<10} total: {processed_data.size:<10}')

    ax[-1].hist(processed_data.flatten(), bins=1000, histtype='step', label=f'b-value: ALL')
    ax[-1].legend()
    ax[-1].yaxis.set_visible(False)
    ax[-1].axvline(min_val, color='r', linestyle='--')
    ax[-1].axvline(max_val, color='g', linestyle='--')

    ax[-1].set_xlabel('Signal Intensity')
    plt.savefig(data_analysis_paths.histograms_file)


if __name__ == '__main__':
    main()
