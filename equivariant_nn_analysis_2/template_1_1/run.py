'''Connected to equivariant_nn_2 template 1_1.
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
from tqdm import tqdm
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import numpy as np
from torch.utils.data import DataLoader

from equivariant_nn_2.template_1_1.run import (
    get_selection_masks,
    SignalsDataset,
    SignalToIrreps,
    DiffusionDataset,
    HiddenGatedLinear,
    LastGatedLinear,
    EquivariantNet,
    CustomActivation,
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


class EquivariantNNHyperparameters(Protocol):
    b_values_to_select: list[float]
    simulation_data_paths_pkl: str


class EquivariantNNPaths(Protocol):
    hyperparameters_file: str
    best_model_file: str


@dataclass
class EquivariantNNAnalysisHyperparameters:
    seed: int
    snr: float
    batch_size: int
    equivariant_nn_paths_pkl: str


class EquivariantNNAnalysisPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('equivariant_nn_analysis_2', 'template_1_1', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')

        self.pred_d_tensors_file = os.path.join(self.experiment_path, 'pred_d_tensors.npy')
        self.pred_d_star_tensors_file = os.path.join(self.experiment_path, 'pred_d_star_tensors.npy')
        self.pred_f_values_file = os.path.join(self.experiment_path, 'pred_f_values.npy')
        self.pred_S0_corrections_file = os.path.join(self.experiment_path, 'pred_S0_corrections.npy')


def main():

    ## EQUIVARIANT NN ANALYSIS PATHS

    equivariant_nn_analysis_paths = EquivariantNNAnalysisPaths()

    print(f'Experiment path: {equivariant_nn_analysis_paths.experiment_path}')

    os.makedirs(equivariant_nn_analysis_paths.experiment_path)

    with open(equivariant_nn_analysis_paths.paths_file, 'wb') as f:
        pickle.dump(equivariant_nn_analysis_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=equivariant_nn_analysis_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f'Equivariant NN analysis experiment:')
    logging.info(equivariant_nn_analysis_paths.experiment_path)
    logging.info('')


    ## EQUIVARIANT NN ANALYSIS HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--equivariant_nn_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    equivariant_nn_analysis_hparams = EquivariantNNAnalysisHyperparameters(**vars(args))

    with open(equivariant_nn_analysis_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(equivariant_nn_analysis_hparams, f)
    
    logging.info('Equivariant NN analysis hyperparameters:')
    for key, value in vars(equivariant_nn_analysis_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## DEVICE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f'Using {device} device')
    logging.info('')


    ## EQUIVARIANT NN PATHS

    with open(equivariant_nn_analysis_hparams.equivariant_nn_paths_pkl, 'rb') as f:
        equivariant_nn_paths: EquivariantNNPaths = pickle.load(f)
    

    ## EQUIVARIANT NN HYPERPARAMETERS

    with open(equivariant_nn_paths.hyperparameters_file, 'rb') as f:
        equivariant_nn_hparams: EquivariantNNHyperparameters = pickle.load(f)
    

    ## SIMULATION DATA PATHS

    with open(equivariant_nn_hparams.simulation_data_paths_pkl, 'rb') as f:
        sim_data_paths: SimulationDataPaths = pickle.load(f)
    

    ## SIMULATION DATA HYPERPARAMETERS

    with open(sim_data_paths.hyperparameters_file, 'rb') as f:
        sim_data_hparams: SimulationDataHyperparameters = pickle.load(f)
    

    ## REPRODUCIBILITY

    np.random.seed(equivariant_nn_analysis_hparams.seed)


    ## NUMPY DATA

    assert equivariant_nn_analysis_hparams.snr in sim_data_hparams.test_snrs, \
        f'SNR {equivariant_nn_analysis_hparams.snr} not in test SNRs: {sim_data_hparams.test_snrs}'

    b_values: np.ndarray = np.load(sim_data_paths.new_b_values_file)
    b_vectors: np.ndarray = np.load(sim_data_paths.new_b_vectors_file)

    selection_masks = get_selection_masks(equivariant_nn_hparams.b_values_to_select, b_values)

    
    ## TEST DATA

    test_data: np.ndarray = np.load(sim_data_paths.test_data_template.format(int(equivariant_nn_analysis_hparams.snr)))

    logging.info(f'Test data shape: {test_data.shape}')
    logging.info('')
    
    
    ## TORCH CONVERSION

    b_values = torch.from_numpy(b_values).float()
    b_vectors = torch.from_numpy(b_vectors).float()

    test_data = torch.from_numpy(test_data).float()


    # SIGNAL TO IRREPS IN BATCHES

    signal_to_irreps = SignalToIrreps(equivariant_nn_hparams.lmax)

    bvals = []
    bvecs = []
    
    test_signals = []

    test_coeffs = []

    for b_value, selection_mask in selection_masks.items():

        print(f'Computing test coeffs for b-value: {b_value}')

        selected_bvals = b_values[selection_mask]
        selected_bvecs = b_vectors[selection_mask]

        selected_test_signals = test_data[:, selection_mask]

        selected_test_dataset = SignalsDataset(selected_test_signals)

        selected_test_loader = DataLoader(
            dataset=selected_test_dataset, 
            batch_size=equivariant_nn_analysis_hparams.batch_size, 
            shuffle=False,
            drop_last=False)
        
        selected_test_coeffs = []
        for signals in tqdm(selected_test_loader):
            selected_test_coeffs.append(
                signal_to_irreps(
                    signals, 
                    selected_bvecs.unsqueeze(0).repeat(signals.shape[0], 1, 1)))
        
        bvals.append(selected_bvals)
        bvecs.append(selected_bvecs)
        
        test_signals.append(selected_test_signals)

        test_coeffs.append(torch.cat(selected_test_coeffs, dim=0))

    bvals = torch.cat(bvals, dim=0).to(device)
    bvecs = torch.cat(bvecs, dim=0).to(device)
    
    test_signals = torch.cat(test_signals, dim=1)

    test_coeffs = torch.cat(test_coeffs, dim=1)


    ## DATASET AND DATALOADER

    test_dataset = DiffusionDataset(test_signals, test_coeffs)

    logging.info(f'Test dataset: {len(test_dataset)}')
    logging.info('')
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=equivariant_nn_analysis_hparams.batch_size, 
        shuffle=False, 
        drop_last=False)
    

    ## LOAD MODEL

    model: EquivariantNet = torch.load(equivariant_nn_paths.best_model_file).to(device)

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
        for signals, coeffs in test_loader:
            signals = signals.to(device)
            coeffs = coeffs.to(device)
            D, D_star, f, S0 = model(coeffs)
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
    
    np.save(equivariant_nn_analysis_paths.pred_d_tensors_file, pred_d_tensors.cpu().numpy())
    np.save(equivariant_nn_analysis_paths.pred_d_star_tensors_file, pred_d_star_tensors.cpu().numpy())
    np.save(equivariant_nn_analysis_paths.pred_f_values_file, pred_f_values.cpu().numpy())
    np.save(equivariant_nn_analysis_paths.pred_S0_corrections_file, pred_S0_corrections.cpu().numpy())
    
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
