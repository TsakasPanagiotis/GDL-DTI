'''Load errors from ground truth.
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


class GroundTruthPaths(Protocol):
    errors_file: str


@dataclass
class GroundTruthAnalysisHyperparameters:
    ground_truth_paths_pkl: str


class GroundTruthAnalysisPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('ground_truth_analysis', 'template_2', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')


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
    

    ## GROUND TRUTH ERRORS
        
    with open(ground_truth_paths.errors_file, 'rb') as f:
        ground_truth_errors: dict[tuple[int,int,int], np.ndarray] = pickle.load(f)


    ## ANALYSIS

    all_errors = np.vstack(list(ground_truth_errors.values()))

    all_errors_mean = np.mean(all_errors)
    all_errors_std = np.std(all_errors)
    logging.info(f'Errors: {all_errors_mean:.2f} +- {all_errors_std:.2f}')

    mean_error_per_voxel = np.mean(all_errors, axis=1)
    mean_error_per_voxel_mean = np.mean(mean_error_per_voxel)
    mean_error_per_voxel_std = np.std(mean_error_per_voxel)
    logging.info(f'Mean error per voxel: {mean_error_per_voxel_mean:.2f} +- {mean_error_per_voxel_std:.2f}')

    mean_absolute_error_per_voxel = np.mean(np.abs(all_errors), axis=1)
    mean_absolute_error_per_voxel_mean = np.mean(mean_absolute_error_per_voxel)
    mean_absolute_error_per_voxel_std = np.std(mean_absolute_error_per_voxel)
    logging.info(f'Mean absolute error per voxel: {mean_absolute_error_per_voxel_mean:.2f} +- {mean_absolute_error_per_voxel_std:.2f}')

    root_mean_squared_error_per_voxel = np.sqrt(np.mean(all_errors**2, axis=1))
    root_mean_squared_error_per_voxel_mean = np.mean(root_mean_squared_error_per_voxel)
    root_mean_squared_error_per_voxel_std = np.std(root_mean_squared_error_per_voxel)
    logging.info(f'Root mean squared error per voxel: {root_mean_squared_error_per_voxel_mean:.2f} +- {root_mean_squared_error_per_voxel_std:.2f}')


if __name__ == '__main__':
    main()
