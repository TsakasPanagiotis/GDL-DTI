'''Connected to fc_network_2 template 1.
Load simulated b-values and b-vectors.
Load simulated test data for given snr.
Use signals from selected b-values that include zero.
Pass through the best model.
Save D and D*, f values and S0_correction.
Measure reconstruction errors.'''


import os
import sys
sys.path.append(os.getcwd())
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import numpy as np
from torch.utils.data import DataLoader

from fc_network_2.template_2.run import (
    get_selection_mask,
    DiffusionDataset,
    DiffusionNet,
    reconstruct
)


class SimulationDataHyperparameters(Protocol):
    test_snrs: list[float]


class SimulationDataPaths(Protocol):
    hyperparameters_file: str
    new_b_values_file: str
    new_b_vectors_file: str
    test_data_template: str
    test_f_values_template: str
    test_d_tensors_template: str
    test_d_star_tensors_template: str


class FCNetworkHyperparameters(Protocol):
    b_values_to_select: list[float]
    simulation_data_paths_pkl: str


class FCNetworkPaths(Protocol):
    hyperparameters_file: str
    best_model_file: str


@dataclass
class FCNetworkAnalysisHyperparameters:
    seed: int
    snr: float
    batch_size: int
    fc_network_paths_pkl: str


class FCNetworkAnalysisPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('fc_network_analysis_2', 'template_2', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')

        self.pred_d_tensors_file = os.path.join(self.experiment_path, 'pred_d_tensors.npy')
        self.pred_d_star_tensors_file = os.path.join(self.experiment_path, 'pred_d_star_tensors.npy')
        self.pred_f_values_file = os.path.join(self.experiment_path, 'pred_f_values.npy')
        self.pred_S0_corrections_file = os.path.join(self.experiment_path, 'pred_S0_corrections.npy')


def main():

    ## FC NETWORK ANALYSIS PATHS

    fc_network_analysis_paths = FCNetworkAnalysisPaths()

    print(f'Experiment path: {fc_network_analysis_paths.experiment_path}')

    os.makedirs(fc_network_analysis_paths.experiment_path)

    with open(fc_network_analysis_paths.paths_file, 'wb') as f:
        pickle.dump(fc_network_analysis_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=fc_network_analysis_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f'FC network analysis experiment:')
    logging.info(fc_network_analysis_paths.experiment_path)
    logging.info('')


    ## FC NETWORK ANALYSIS HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--fc_network_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    fc_network_analysis_hparams = FCNetworkAnalysisHyperparameters(**vars(args))

    with open(fc_network_analysis_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(fc_network_analysis_hparams, f)
    
    logging.info('FC network analysis hyperparameters:')
    for key, value in vars(fc_network_analysis_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## DEVICE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f'Using {device} device')
    logging.info('')


    ## FC NETWORK PATHS

    with open(fc_network_analysis_hparams.fc_network_paths_pkl, 'rb') as f:
        fc_network_paths: FCNetworkPaths = pickle.load(f)
    

    ## FC NETWORK HYPERPARAMETERS

    with open(fc_network_paths.hyperparameters_file, 'rb') as f:
        fc_network_hparams: FCNetworkHyperparameters = pickle.load(f)
    

    ## SIMULATION DATA PATHS

    with open(fc_network_hparams.simulation_data_paths_pkl, 'rb') as f:
        sim_data_paths: SimulationDataPaths = pickle.load(f)
    

    ## SIMULATION DATA HYPERPARAMETERS

    with open(sim_data_paths.hyperparameters_file, 'rb') as f:
        sim_data_hparams: SimulationDataHyperparameters = pickle.load(f)
    

    ## REPRODUCIBILITY

    np.random.seed(fc_network_analysis_hparams.seed)


    ## NUMPY DATA

    assert fc_network_analysis_hparams.snr in sim_data_hparams.test_snrs, \
        f'SNR {fc_network_analysis_hparams.snr} not in test SNRs: {sim_data_hparams.test_snrs}'

    b_values: np.ndarray = np.load(sim_data_paths.new_b_values_file)
    b_vectors: np.ndarray = np.load(sim_data_paths.new_b_vectors_file)

    selection_mask = get_selection_mask(fc_network_hparams.b_values_to_select, b_values)

    
    ## TEST DATA

    test_data: np.ndarray = np.load(sim_data_paths.test_data_template.format(int(fc_network_analysis_hparams.snr)))

    logging.info(f'Test data shape: {test_data.shape}')
    logging.info('')
    
    
    ## TORCH CONVERSION

    bvals = torch.from_numpy(b_values[selection_mask]).float().to(device)
    bvecs = torch.from_numpy(b_vectors[selection_mask]).float().to(device)

    test_signals = torch.from_numpy(test_data[:, selection_mask]).float()


    ## DATASET AND DATALOADER

    test_dataset = DiffusionDataset(test_signals)

    logging.info(f'Test dataset: {len(test_dataset)}')
    logging.info('')
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=fc_network_analysis_hparams.batch_size, 
        shuffle=False, 
        drop_last=False)
    

    ## LOAD MODEL

    model: DiffusionNet = torch.load(fc_network_paths.best_model_file).to(device)

    logging.info('Model loaded')
    logging.info(model)
    logging.info('')

    model.eval()


    ## EVALUATION

    errors = []
    pred_d_tensors = []
    pred_d_star_tensors = []
    pred_f_values = []
    pred_S0_corrections = []

    with torch.no_grad():
        for signals in test_loader:
            signals = signals.to(device)
            D, D_star, f, S0 = model(signals)
            recon_signals = reconstruct(S0, bvals, bvecs, D, D_star, f)
            errors.append(recon_signals - signals)
            pred_d_tensors.append(D)
            pred_d_star_tensors.append(D_star)
            pred_f_values.append(f)
            pred_S0_corrections.append(S0)
    
    pred_d_tensors = torch.cat(pred_d_tensors, dim=0)
    pred_d_star_tensors = torch.cat(pred_d_star_tensors, dim=0)
    pred_f_values = torch.cat(pred_f_values, dim=0)
    pred_S0_corrections = torch.cat(pred_S0_corrections, dim=0)
    
    np.save(fc_network_analysis_paths.pred_d_tensors_file, pred_d_tensors.cpu().numpy())
    np.save(fc_network_analysis_paths.pred_d_star_tensors_file, pred_d_star_tensors.cpu().numpy())
    np.save(fc_network_analysis_paths.pred_f_values_file, pred_f_values.cpu().numpy())
    np.save(fc_network_analysis_paths.pred_S0_corrections_file, pred_S0_corrections.cpu().numpy())
    
    logging.info(f'Predicted diffusion tensors shape: {pred_d_tensors.shape}')
    logging.info('')
    
    all_errors = torch.cat(errors, dim=0).cpu().numpy()    
    all_errors_mean = np.mean(all_errors)
    all_errors_std = np.std(all_errors)
    logging.info(f'Errors: {all_errors_mean} +- {all_errors_std}')

    mean_error_per_voxel = np.mean(all_errors, axis=1)
    mean_error_per_voxel_mean = np.mean(mean_error_per_voxel)
    mean_error_per_voxel_std = np.std(mean_error_per_voxel)
    logging.info(f'Mean error per voxel: {mean_error_per_voxel_mean} +- {mean_error_per_voxel_std}')

    mean_absolute_error_per_voxel = np.mean(np.abs(all_errors), axis=1)
    mean_absolute_error_per_voxel_mean = np.mean(mean_absolute_error_per_voxel)
    mean_absolute_error_per_voxel_std = np.std(mean_absolute_error_per_voxel)
    logging.info(f'Mean absolute error per voxel: {mean_absolute_error_per_voxel_mean} +- {mean_absolute_error_per_voxel_std}')

    root_mean_squared_error_per_voxel = np.sqrt(np.mean(all_errors**2, axis=1))
    root_mean_squared_error_per_voxel_mean = np.mean(root_mean_squared_error_per_voxel)
    root_mean_squared_error_per_voxel_std = np.std(root_mean_squared_error_per_voxel)
    logging.info(f'Root mean squared error per voxel: {root_mean_squared_error_per_voxel_mean} +- {root_mean_squared_error_per_voxel_std}')


if __name__ == '__main__':
    main()
