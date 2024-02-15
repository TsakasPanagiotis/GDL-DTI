'''Go from b-values, direction vectors and least squares results
to noisy signals and d-tensors
and store them as torch tensors.'''


import os
import pickle
import logging
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm


@dataclass
class Hyperparameters:
    snr: int
    b_values_to_select: list[float]
    lstsq_results_path: str


@dataclass
class Paths:
    experiments_dir = os.path.join('simulation_data', 'template_1', 'experiments')
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    direction_vectors_path = 'C:/Users/panag/Desktop/Test/mgh_1001/diff/preproc/bvecs_moco_norm.txt'
    b_values_path = 'C:/Users/panag/Desktop/Test/mgh_1001/diff/preproc/bvals.txt'

    @property
    def experiment_path(self):
        return os.path.join(self.experiments_dir, self.experiment_name)
    
    @property
    def log_file(self):
        return os.path.join(self.experiment_path, 'log.txt')
    
    @property
    def d_tensors_file(self):
        return os.path.join(self.experiment_path, 'd_tensors.pt')
    
    @property
    def noisy_signals_file(self):
        return os.path.join(self.experiment_path, 'noisy_signals.pt')


def create_selection_mask(b_values_to_select_list: list[float], b_values: np.ndarray) -> np.ndarray:

    b_values_to_select = set(b_values_to_select_list)
    unique_b_values = set(b_values)

    if len(b_values_to_select) == 0:
        b_values_to_select = unique_b_values
    
    if 0.0 not in b_values_to_select:
        logging.error('b_values_to_select must contain 0.0')
        raise ValueError('b_values_to_select must contain 0.0')

    if not b_values_to_select.issubset(unique_b_values):
        logging.error(f'Invalid b_values_to_select: {b_values_to_select.difference(unique_b_values)}. Valid values are: {unique_b_values}')
        raise ValueError(f'Invalid b_values_to_select: {b_values_to_select.difference(unique_b_values)}. Valid values are: {unique_b_values}')

    selection_mask = np.any(
        np.stack(
            [(b_values == value) for value in b_values_to_select]
        ), 
        axis=0
    )

    return selection_mask


def main():

    paths = Paths()

    
    ## HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--snr', type=int, required=True)
    parser.add_argument('--b_values_to_select', type=float, nargs='+', required=True)
    parser.add_argument('--lstsq_results_path', type=str, required=True)
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


    ## DIRECTION VECTORS

    direction_vectors = np.genfromtxt(paths.direction_vectors_path)


    ## B-VALUES

    b_values = np.genfromtxt(paths.b_values_path)
    b_values[(b_values > 9_900) & (b_values < 10_100)] = 10_000
    b_values /= 1_000.0


    ## LOAD LEAST SQUARES RESULTS

    with open(HP.lstsq_results_path, 'rb') as f:
        lstsq_results: dict[tuple[int,int,int], np.ndarray] = pickle.load(f)
    

    ## CALCULATE D-TENSORS AND NOISY SIGNALS
        
    selection_mask = create_selection_mask(HP.b_values_to_select, b_values)
    selected_b_values = torch.from_numpy(b_values[selection_mask]).float()
    selected_gradients = torch.from_numpy(direction_vectors[selection_mask]).float()

    d_tensors = []
    noisy_signals = []

    for d_array in tqdm(lstsq_results.values()):

        d_tensor = torch.tensor(d_array).float()
        
        signal = torch.exp(- selected_b_values * torch.einsum('bi, ij, bj -> b', selected_gradients, d_tensor, selected_gradients))
        
        noise = torch.normal(mean=torch.zeros_like(signal), std=torch.ones_like(signal) / HP.snr)

        noisy_signal = signal + noise

        noisy_signal /= torch.mean(noisy_signal[selected_b_values == 0.0])

        d_tensors.append(d_tensor)
        noisy_signals.append(noisy_signal)

    d_tensors = torch.stack(d_tensors) # shape (num_d_tensors, 3, 3)
    noisy_signals = torch.stack(noisy_signals) # shape (num_d_tensors, len(selection_mask))

    logging.info(f'd_tensors: {d_tensors.shape}')
    logging.info(f'noisy_signals: {noisy_signals.shape}')
    logging.info(f'noisy_signals min: {noisy_signals.min()}')
    logging.info(f'noisy_signals max: {noisy_signals.max()}')

    
    ## SAVE SIMULATION DATA

    torch.save(d_tensors, paths.d_tensors_file)
    torch.save(noisy_signals, paths.noisy_signals_file)


if __name__ == '__main__':
    main()
