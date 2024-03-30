'''Basic processing of raw data.
Store b-values, b-vectors, median_otsu mask,
processed data based on mask as 2D array
and b0 images.'''


import os
import pickle
import logging
from typing import Protocol, cast
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
import nibabel as nib
from dipy.segment.threshold import otsu
from dipy.segment.mask import multi_median
from nibabel.spatialimages import SpatialImage


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
        self.processed_data_file = os.path.join(self.experiment_path, 'data.npy')
        self.b0_images_file = os.path.join(self.experiment_path, 'b0_images.nii.gz')


def custom_median_otsu(input_volume: np.ndarray, vol_idx: np.ndarray) -> np.ndarray:

    b0vol: np.ndarray = np.mean(input_volume[..., tuple(vol_idx)], axis=3)
    mask: np.ndarray = multi_median(b0vol, median_radius=4, numpass=4)
    thresh: float = otsu(mask)
    mask = mask > thresh

    return mask.astype(bool)


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
    
    logging.info('Processed data experiment:')
    logging.info(proc_data_paths.experiment_path)
    logging.info('')


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

    logging.info(f'b_vectors: {b_vectors.shape}')
    logging.info('')


    ## B-VALUES

    b_values = np.genfromtxt(raw_data_paths.b_values_file)
    
    b_values[(b_values > 9_900) & (b_values < 10_100)] = 10_000
    b_values /= 1_000.0

    np.save(proc_data_paths.b_values_file, b_values)

    logging.info(f'b_values: {np.unique(b_values)}')
    logging.info('')


    ## MEDIAN-OTSU MASK
    
    images  = cast(SpatialImage, nib.loadsave.load(raw_data_paths.raw_data_file))
    
    data = images.get_fdata()
    
    # use b-value = 0.0 for stronger brain signal to get the mask
    mask = custom_median_otsu(data, vol_idx=np.where(b_values == 0.0)[0])

    np.save(proc_data_paths.mask_file, mask)

    logging.info(f'Mask: {mask.shape}')
    logging.info('')


    ## PROCESSED DATA

    processed_data = data[mask]

    np.save(proc_data_paths.processed_data_file, processed_data)

    logging.info(f'Processed data: {processed_data.shape}')
    logging.info('')


    ## B0 IMAGE

    b0_images = data[:, :, :, b_values == 0.0].mean(axis=-1)

    b0_images_nii = nib.nifti1.Nifti1Image(b0_images, images.affine)

    nib.loadsave.save(b0_images_nii, proc_data_paths.b0_images_file)

    logging.info(f'b0_images: {b0_images.shape}')


if __name__ == '__main__':
    main()
