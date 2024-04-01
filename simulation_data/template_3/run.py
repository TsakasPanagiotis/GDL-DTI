''''''


import os
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm


class ProcessedDataPaths(Protocol):
    b_values_file: str
    b_vectors_file: str


class GroundTruthHyperparameters(Protocol):
    processed_data_paths_pkl: str


class GroundTruthPaths(Protocol):
    hyperparameters_file: str
    d_tensors_file: str


class SegmentationHyperparameters(Protocol):
    processed_data_paths_pkl: str


class SegmentationPaths(Protocol):
    hyperparameters_file: str


class WhiteMatterHyperparameters(Protocol):
    segmentation_paths_pkl: str


class WhiteMatterPaths(Protocol):
    hyperparameters_file: str
    b0_mean_file: str


@dataclass
class SimulationDataHyperparameters:
    snr: float
    ground_truth_paths_pkl: str
    white_matter_paths_pkl: str


class SimulationDataPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('simulation_data', 'template_3', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')


def main():

    ## SIMULATION DATA PATHS

    simulation_data_paths = SimulationDataPaths()

    print(f'Experiment path: {simulation_data_paths.experiment_path}')

    os.makedirs(simulation_data_paths.experiment_path)

    with open(simulation_data_paths.paths_file, 'wb') as f:
        pickle.dump(simulation_data_paths, f)
    

    ## LOGGING
    
    logging.basicConfig(
        filename=simulation_data_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f'Simulation data experiment:')
    logging.info(simulation_data_paths.experiment_path)
    logging.info('')


    ## SIMULATION DATA HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--snr', type=float, required=True)
    parser.add_argument('--ground_truth_paths_pkl', type=str, required=True)
    parser.add_argument('--white_matter_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    simulation_data_hparams = SimulationDataHyperparameters(**vars(args))

    with open(simulation_data_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(simulation_data_hparams, f)
    
    logging.info('Simulation data hyperparameters:')
    for key, value in vars(simulation_data_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## GROUND TRUTH PATHS

    with open(simulation_data_hparams.ground_truth_paths_pkl, 'rb') as f:
        ground_truth_paths: GroundTruthPaths = pickle.load(f)
    

    ## GROUND TRUTH HYPERPARAMETERS
    
    with open(ground_truth_paths.hyperparameters_file, 'rb') as f:
        ground_truth_hparams: GroundTruthHyperparameters = pickle.load(f)
    

    ## WHITE MATTER PATHS
    
    with open(simulation_data_hparams.white_matter_paths_pkl, 'rb') as f:
        white_matter_paths: WhiteMatterPaths = pickle.load(f)
    

    ## WHITE MATTER HYPERPARAMETERS
    
    with open(white_matter_paths.hyperparameters_file, 'rb') as f:
        white_matter_hparams: WhiteMatterHyperparameters = pickle.load(f)
    

    ## SEGMENTATION PATHS
    
    with open(white_matter_hparams.segmentation_paths_pkl, 'rb') as f:
        segmentation_paths: SegmentationPaths = pickle.load(f)
    

    ## SEGMENATION HYPERPARAMETERS
    
    with open(segmentation_paths.hyperparameters_file, 'rb') as f:
        segmentation_hparams: SegmentationHyperparameters = pickle.load(f)
    

    if ground_truth_hparams.processed_data_paths_pkl \
        != segmentation_hparams.processed_data_paths_pkl:
        logging.error('Processed data paths do not match.')
        raise AssertionError('Processed data paths do not match.')
    

    ## PROCESSED DATA PATHS

    with open(ground_truth_hparams.processed_data_paths_pkl, 'rb') as f:
        proc_data_paths: ProcessedDataPaths = pickle.load(f)
    

    ## DATA
    
    b_values: np.ndarray = np.load(proc_data_paths.b_values_file)
    b_vectors: np.ndarray = np.load(proc_data_paths.b_vectors_file)

    d_tensors: np.ndarray = np.load(ground_truth_paths.d_tensors_file)

    b0_mean: np.ndarray = np.load(white_matter_paths.b0_mean_file)


    ## SIMULATION DATA

    


if __name__ == '__main__':
    main()

