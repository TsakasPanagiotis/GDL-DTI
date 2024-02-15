'''Load two sets of results from lstsq
and compare common and equal common voxels results.'''


import os
import pickle
import logging
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm


@dataclass
class Hyperparameters:
    
    lstsq_results_1_path: str
    lstsq_results_2_path: str


@dataclass
class Paths:
    experiments_dir = os.path.join('ground_truth', 'template_4', 'experiments')
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    @property
    def experiment_path(self):
        return os.path.join(self.experiments_dir, self.experiment_name)
    
    @property
    def log_file(self):
        return os.path.join(self.experiment_path, 'log.txt')


def main():

    paths = Paths()


    ## HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--lstsq_results_1_path', type=str, required=True)
    parser.add_argument('--lstsq_results_2_path', type=str, required=True)
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


    ## LOAD RESULTS
    
    with open(HP.lstsq_results_1_path, 'rb') as f:
        lstsq_results_1: dict[tuple[int,int,int], np.ndarray] = pickle.load(f)

    with open(HP.lstsq_results_2_path, 'rb') as f:
        lstsq_results_2: dict[tuple[int,int,int], np.ndarray] = pickle.load(f)
    
    logging.info(f'lstsq_results_1: {len(lstsq_results_1)}')
    logging.info(f'lstsq_results_2: {len(lstsq_results_2)}')
    

    ## COMPARE RESULTS
    
    equal_common_count = 0

    common_voxels = set(lstsq_results_1.keys()) & (set(lstsq_results_2.keys()))

    # count equal d-tensors for common voxels
    for voxel in tqdm(common_voxels):
        if np.allclose(lstsq_results_1[voxel], lstsq_results_2[voxel], rtol=1e-3, atol=1e-3):
            equal_common_count += 1

    logging.info(f'common voxels: {len(common_voxels)}')
    logging.info(f'equal common voxels: {equal_common_count}')


if __name__ == '__main__':
    main()
