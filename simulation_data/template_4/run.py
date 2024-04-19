'''Load processed b-values and b-vectors.
Load ground truth diffusion tensors.
Shuffle and split the indices based on the splits.
Save indices for train, eval and test splits.
Create SNR distribution based on the probabilities and bounds.
Sample SNRs for training and evaluation data.
Save SNRs for training and evaluation data.
Create noisy signals for training and evaluation data.
Save noisy signals for training and evaluation data.'''


import os
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class ProcessedDataPaths(Protocol):
    b_values_file: str
    b_vectors_file: str


class GroundTruthHyperparameters(Protocol):
    processed_data_paths_pkl: str


class GroundTruthPaths(Protocol):
    hyperparameters_file: str
    d_tensors_file: str


@dataclass
class SimulationDataHyperparameters:
    seed: int
    splits: list[float]
    probabilities: list[float]
    snr_bounds: list[float]
    ground_truth_paths_pkl: str


class SimulationDataPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('simulation_data', 'template_4', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')

        self.train_indices_file = os.path.join(self.experiment_path, 'train_indices.npy')
        self.eval_indices_file = os.path.join(self.experiment_path, 'eval_indices.npy')
        self.test_indices_file = os.path.join(self.experiment_path, 'test_indices.npy')

        self.train_snrs_file = os.path.join(self.experiment_path, 'train_snrs.npy')
        self.eval_snrs_file = os.path.join(self.experiment_path, 'eval_snrs.npy')

        self.simulation_train_data_file = os.path.join(self.experiment_path, 'train_data.npy')
        self.simulation_eval_data_file = os.path.join(self.experiment_path, 'eval_data.npy')

        self.snr_histogram_file = os.path.join(self.experiment_path, 'snr.png')


def get_snr_distribution(probs: list[float], snr_bounds: list[float], num_samples: int) -> np.ndarray:

    # Define the range for each probability
    min_snr, max_snr = snr_bounds
    step = (max_snr - min_snr) / len(probs)    
    ranges = []
    low = min_snr
    for _ in probs:
        high = low + step
        ranges.append((low, high))
        low = high

    # Calculate the number of samples to generate from each range
    samples_per_range = [int(num_samples * p) for p in probs]

    # Generate random samples within each range
    noise_snrs = []
    for (low, high), num in zip(ranges, samples_per_range):
        noise_snrs.append(np.random.uniform(low, high, num))

    # Combine the samples from all ranges
    noise_snrs = np.concatenate(noise_snrs)

    return noise_snrs


def get_noisy_signals(indices: np.ndarray, snrs: np.ndarray, d_tensors: np.ndarray, 
                      b_values: np.ndarray, b_vectors: np.ndarray) -> np.ndarray:
    
    signals = []
    for i in tqdm(range(indices.shape[0])):

        index = indices[i]
        d_tensor = d_tensors[index]
        signal = np.exp(- b_values * np.einsum('bi, ij, bj -> b', b_vectors, d_tensor, b_vectors))

        noise_std = 1 / snrs[i]
        noise = np.random.normal(loc=0.0, scale=noise_std, size=signal.shape)
        signal += noise
        signal /= signal[b_values == 0.0].mean()
        
        signals.append(signal)
    
    return np.stack(signals)


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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--splits', type=float, nargs=3, default=[0.75, 0.15, 0.10])
    parser.add_argument('--probabilities', type=float, nargs='+', default=[0.5, 0.25, 0.15, 0.1])
    parser.add_argument('--snr_bounds', type=float, nargs=2, default=[10.0, 50.0])
    parser.add_argument('--ground_truth_paths_pkl', type=str, required=True)
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
    

    ## PROCESSED DATA PATHS
    
    with open(ground_truth_hparams.processed_data_paths_pkl, 'rb') as f:
        proc_data_paths: ProcessedDataPaths = pickle.load(f)
    

    ## DATA
    
    b_values: np.ndarray = np.load(proc_data_paths.b_values_file)
    b_vectors: np.ndarray = np.load(proc_data_paths.b_vectors_file)

    d_tensors: np.ndarray = np.load(ground_truth_paths.d_tensors_file)


    ## REPRODUCIBILITY

    np.random.seed(simulation_data_hparams.seed)


    ## SHUFFLE AND SPLIT INDICES

    train_percentage, eval_percentage, test_percentage = simulation_data_hparams.splits

    num_samples = d_tensors.shape[0]
    train_samples = int(train_percentage * num_samples)
    eval_samples = int(eval_percentage * num_samples)
    test_samples = num_samples - train_samples - eval_samples

    logging.info(f'Train samples: {train_samples}')
    logging.info(f'Evaluation samples: {eval_samples}')
    logging.info(f'Test samples: {test_samples}')
    logging.info('')

    indices = np.random.permutation(num_samples)
    train_indices = indices[:train_samples]
    eval_indices = indices[train_samples:train_samples + eval_samples]
    test_indices = indices[train_samples + eval_samples:]

    np.save(simulation_data_paths.train_indices_file, train_indices)
    np.save(simulation_data_paths.eval_indices_file, eval_indices)
    np.save(simulation_data_paths.test_indices_file, test_indices)


    ## SNR DISTRIBUTION AND SAMPLING

    snr_distribution = get_snr_distribution(simulation_data_hparams.probabilities, 
                                            simulation_data_hparams.snr_bounds, num_samples)

   
    train_snrs = np.random.choice(snr_distribution, size=train_samples)
    eval_snrs = np.random.choice(snr_distribution, size=eval_samples)

    np.save(simulation_data_paths.train_snrs_file, train_snrs)
    np.save(simulation_data_paths.eval_snrs_file, eval_snrs)

    # save histogram of SNRs
    plt.hist(snr_distribution, bins=100, label='All')
    plt.hist(train_snrs, bins=100, label='Train')
    plt.hist(eval_snrs, bins=100, label='Eval')
    plt.legend()
    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.title('SNR Distribution')
    plt.savefig(simulation_data_paths.snr_histogram_file)
    plt.clf()  # Clear the figure            


    ## SIMULATION DATA

    noisy_train_signals = get_noisy_signals(train_indices, train_snrs, d_tensors, b_values, b_vectors)
    noisy_eval_signals = get_noisy_signals(eval_indices, eval_snrs, d_tensors, b_values, b_vectors)

    np.save(simulation_data_paths.simulation_train_data_file, noisy_train_signals)
    np.save(simulation_data_paths.simulation_eval_data_file, noisy_eval_signals)

    logging.info(f'Training data: min = {noisy_train_signals.min()} - max = {noisy_train_signals.max()}')
    logging.info(f'Evaluation data: min = {noisy_eval_signals.min()} - max = {noisy_eval_signals.max()}')


if __name__ == '__main__':
    main()
