'''Load processed b0 images.
Load white matter segmentation.
Apply white matter mask to b0 images.
Calculate mean b0 value in masked white matter
for selected slices.
Save mean b0 value and plot of a masked slice.'''

import os
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np


class SegmentationPaths(Protocol):
    segmentation_npy_file: str
    b0_images_npy_file: str


@dataclass
class WhiteMatterHyperparameters:
    segmentation_paths_pkl: str
    slices: list[int]


class WhiteMatterPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('white_matter', 'template_1', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')

        self.b0_mean_file = os.path.join(self.experiment_path, 'b0_mean.npy')


def main():
    
    ## GROUND TRUTH PATHS

    white_matter_paths = WhiteMatterPaths()

    print(f'Experiment path: {white_matter_paths.experiment_path}')

    os.makedirs(white_matter_paths.experiment_path)

    with open(white_matter_paths.paths_file, 'wb') as f:
        pickle.dump(white_matter_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=white_matter_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info('White matter experiment:')
    logging.info(white_matter_paths.experiment_path)
    logging.info('')


    ## WHITE MATTER HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--segmentation_paths_pkl', type=str, required=True)
    parser.add_argument('--slices', type=int, nargs=2, required=True)
    args = parser.parse_args()

    white_matter_hparams = WhiteMatterHyperparameters(**vars(args))

    with open(white_matter_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(white_matter_hparams, f)
    
    logging.info('White matter hyperparameters:')
    for key, value in vars(white_matter_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## SEGMENTATION PATHS

    with open(white_matter_hparams.segmentation_paths_pkl, 'rb') as f:
        segmentation_paths: SegmentationPaths = pickle.load(f)
    

    ## DATA
    
    # 3D float array of b0 images
    b0_images: np.ndarray = np.load(segmentation_paths.b0_images_npy_file)

    # 3D array of white matter boolean mask
    white_matter_mask: np.ndarray = np.load(segmentation_paths.segmentation_npy_file)

    
    ## B0 MEAN

    min_slice, max_slice = white_matter_hparams.slices
    
    white_matter_b0_images = b0_images[..., min_slice:max_slice] * white_matter_mask[..., min_slice:max_slice]

    white_matter_b0_mean = white_matter_b0_images.mean()

    np.save(white_matter_paths.b0_mean_file, white_matter_b0_mean)

    logging.info(f'Total white matter voxels: {white_matter_mask[..., min_slice:max_slice].sum()}')
    logging.info(f'Mean b0 value in white matter: {white_matter_b0_mean}')


if __name__ == '__main__':
    main()
