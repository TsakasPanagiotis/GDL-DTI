'''Load processed b0 images nii.gz.
Load white matter segmentation mask nii.gz.
Save them as numpy arrays.
Save an overlap plot of a slice and the mask.'''


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


class ProcessedDataPaths(Protocol):
    b0_images_file: str


@dataclass
class SegmentationHyperparameters:
    processed_data_paths_pkl: str


class SegmentationPaths:
    def __init__(self) -> None:
        template_path = os.path.join('segmentation', 'template_1')
        self.experiment_path = os.path.join(template_path, 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')

        self.segmentation_nii_gz_file = os.path.join(template_path, 'segmentation.nii.gz')
        self.segmentation_npy_file = os.path.join(self.experiment_path, 'segmentation.npy')
        self.b0_images_npy_file = os.path.join(self.experiment_path, 'b0_images.npy')
        self.slice_plot_file = os.path.join(self.experiment_path, 'slice_plot.png')


def main():
    
    ## SEGMENTATION PATHS

    segmentation_paths = SegmentationPaths()

    print(f'Experiment path: {segmentation_paths.experiment_path}')

    os.makedirs(segmentation_paths.experiment_path)

    with open(segmentation_paths.paths_file, 'wb') as f:
        pickle.dump(segmentation_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=segmentation_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info('Segmentation experiment:')
    logging.info(segmentation_paths.experiment_path)
    logging.info('')


    ## SEGMENTATION HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--processed_data_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    segmentation_hparams = SegmentationHyperparameters(**vars(args))

    with open(segmentation_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(segmentation_hparams, f)
    
    logging.info('Segmentation hyperparameters:')
    for key, value in vars(segmentation_hparams).items():
        logging.info(f'{key}: {value}')


    ## PROCESSED DATA PATHS

    with open(segmentation_hparams.processed_data_paths_pkl, 'rb') as f:
        processed_data_paths: ProcessedDataPaths = pickle.load(f)

    
    ## SEGMENTATION

    # 3D array of white matter boolean mask
    segmentation: np.ndarray = nib.loadsave.load(segmentation_paths.segmentation_nii_gz_file).get_fdata() > 0 # type: ignore
    
    np.save(segmentation_paths.segmentation_npy_file, segmentation)


    ## B0 IMAGES
    
    # 3D float array of b0 images
    b0_images: np.ndarray = nib.loadsave.load(processed_data_paths.b0_images_file).get_fdata() # type: ignore

    np.save(segmentation_paths.b0_images_npy_file, b0_images)


    ## PLOT

    slice = 50

    plt.imshow(b0_images[:, :, slice], cmap='gray')
    plt.imshow(segmentation[:, :, slice], cmap='viridis', alpha=0.5)
    plt.savefig(segmentation_paths.slice_plot_file)


if __name__ == '__main__':
    main()
