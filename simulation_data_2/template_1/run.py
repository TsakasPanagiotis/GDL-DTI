'''Load processed b-values and b-vectors.
Load ground truth diffusion tensors 
that were fitted for the uni-exponential equation.
Create new b-values between zero and one
and new b-vectors using farthest point sampling. 
Shuffle and split the indices based on the splits.
Save indices for train, eval and test splits
in pairs for d_tensor and d_tensor_star.
Create SNR distribution based on the probabilities and bounds.
Sample SNRs for training and evaluation data.
Save SNRs for training and evaluation data.
Create noisy signals for training and evaluation data
based on the bi-exponential decay equation.
Save noisy signals, D, D* and f-value
for training and evaluation data
and for test data for the given test snrs.'''


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
    test_snrs: list[float]
    new_b_values: list[float]
    new_b_vectors_nums: list[int]
    ground_truth_paths_pkl: str


class SimulationDataPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('simulation_data_2', 'template_1', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')

        self.new_b_values_file = os.path.join(self.experiment_path, 'new_b_values.npy')
        self.new_b_vectors_file = os.path.join(self.experiment_path, 'new_b_vectors.npy')

        self.snr_histogram_file = os.path.join(self.experiment_path, 'snr.png')
        
        self.train_snrs_file = os.path.join(self.experiment_path, 'train_snrs.npy')
        self.train_indices_file = os.path.join(self.experiment_path, 'train_indices.npy')
        self.train_sim_data_file = os.path.join(self.experiment_path, 'train_data.npy')
        self.train_f_values_file = os.path.join(self.experiment_path, 'train_f_values.npy')
        self.train_d_tensors_file = os.path.join(self.experiment_path, 'train_d_tensors.npy')
        self.train_d_star_tensors_file = os.path.join(self.experiment_path, 'train_d_star_tensors.npy')
        
        self.eval_snrs_file = os.path.join(self.experiment_path, 'eval_snrs.npy')
        self.eval_indices_file = os.path.join(self.experiment_path, 'eval_indices.npy')
        self.eval_sim_data_file = os.path.join(self.experiment_path, 'eval_data.npy')
        self.eval_f_values_file = os.path.join(self.experiment_path, 'eval_f_values.npy')
        self.eval_d_tensors_file = os.path.join(self.experiment_path, 'eval_d_tensors.npy')
        self.eval_d_star_tensors_file = os.path.join(self.experiment_path, 'eval_d_star_tensors.npy')
        
        self.test_indices_file = os.path.join(self.experiment_path, 'test_indices.npy')
        self.test_data_template = os.path.join(self.experiment_path, 'test_snr_{}_data.npy')
        self.test_f_values_template = os.path.join(self.experiment_path, 'test_snr_{}_f_values.npy')
        self.test_d_tensors_template = os.path.join(self.experiment_path, 'test_snr_{}_d_tensors.npy')
        self.test_d_star_tensors_template = os.path.join(self.experiment_path, 'test_snr_{}_d_star_tensors.npy')


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    
    farthest_pts = np.zeros((num_samples, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]  # start with random point
    distances = np.linalg.norm(points - farthest_pts[0], axis=1)
    
    for i in range(1, num_samples):
        farthest_pts[i] = points[np.argmax(distances)]
        distances = np.minimum(distances, np.linalg.norm(points - farthest_pts[i], axis=1))
    
    return farthest_pts


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


def get_noisy_biexp_signals(indices: np.ndarray, snrs: np.ndarray, d_tensors: np.ndarray, 
                            b_values: np.ndarray, b_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    f_low = 0.05
    f_high = 0.7 # 0.3
    
    md_low = 0.5 # 0.0005
    md_high = 3.5 # 0.0035

    md_star_low = 5.0 # 0.005
    md_star_high = 70.0 # 0.07

    f_values = np.random.uniform(f_low, f_high, indices.shape[0])
    md_news = np.random.uniform(md_low, md_high, indices.shape[0])
    md_star_news = np.random.uniform(md_star_low, md_star_high, indices.shape[0])
    
    signals = []
    new_d_tensors = []
    new_d_star_tensors = []
    for (index_1, index_2), snr, f, md_new, md_star_new in tqdm(zip(indices, snrs, f_values, md_news, md_star_news), total=indices.shape[0]):

        d_tensor_old = d_tensors[index_1]
        d_tensor_star_old = d_tensors[index_2]

        md_old = np.linalg.eigvals(d_tensor_old).real.mean()
        md_star_old = np.linalg.eigvals(d_tensor_star_old).real.mean()

        d_tensor_new = d_tensor_old * (md_new / md_old)
        d_tensor_star_new = d_tensor_star_old * (md_star_new / md_star_old)

        signal = f * np.exp(- b_values * np.einsum('bi, ij, bj -> b', b_vectors, d_tensor_star_new, b_vectors)) \
                + (1-f) * np.exp(- b_values * np.einsum('bi, ij, bj -> b', b_vectors, d_tensor_new, b_vectors))

        noise_std = 1 / snr
        noise = np.random.normal(loc=0.0, scale=noise_std, size=signal.shape)
        signal += noise
        signal /= signal[b_values == 0.0].mean()
        
        signals.append(signal)
        new_d_tensors.append(d_tensor_new)
        new_d_star_tensors.append(d_tensor_star_new)
    
    return np.stack(signals), np.stack(new_d_tensors), np.stack(new_d_star_tensors), f_values


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
    parser.add_argument('--test_snrs', type=float, nargs='+', default=[15.0, 25.0])
    parser.add_argument('--splits', type=float, nargs=3, default=[0.75, 0.15, 0.10])
    parser.add_argument('--probabilities', type=float, nargs='+', default=[0.5, 0.25, 0.15, 0.1])
    parser.add_argument('--snr_bounds', type=float, nargs=2, default=[10.0, 50.0])
    parser.add_argument('--new_b_values', type=float, nargs='+',
                        default=[ x / 1000.0 for x in [ 5, 10, 20, 40, 80, 120, 170, 250, 400, 600, 800]])
    parser.add_argument('--new_b_vectors_nums', type=int, nargs='+', 
                                              default=[12, 12, 12, 12, 12,  12,  64,  12,  12,  12,  12])
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
    
    old_b_values: np.ndarray = np.load(proc_data_paths.b_values_file)
    old_b_vectors: np.ndarray = np.load(proc_data_paths.b_vectors_file)

    d_tensors: np.ndarray = np.load(ground_truth_paths.d_tensors_file)


    ## REPRODUCIBILITY

    np.random.seed(simulation_data_hparams.seed)


    ## NEW B-VALUES AND B-VECTORS

    unique, counts = np.unique(old_b_values, return_counts=True)
    most_common_b_value = unique[np.argmax(counts)]

    b_vectors_pool = old_b_vectors[old_b_values == most_common_b_value]

    new_b_values = []
    new_b_vectors = []

    # keep the same values and vectors for b=0
    b_value = 0.0
    num = counts[unique == b_value][0]
    new_b_values.extend([b_value] * num)
    new_b_vectors.extend(farthest_point_sampling(b_vectors_pool, num))

    # sample new values and vectors for other b-values
    for b_value, num in zip(simulation_data_hparams.new_b_values, simulation_data_hparams.new_b_vectors_nums):
        new_b_values.extend([b_value] * num)
        new_b_vectors.extend(farthest_point_sampling(b_vectors_pool, num))

    # keep the same values and vectors for b=1
    b_value = 1.0
    num = counts[unique == b_value][0]
    new_b_values.extend([b_value] * num)
    new_b_vectors.extend(farthest_point_sampling(b_vectors_pool, num))

    b_values = np.array(new_b_values)
    b_vectors = np.stack(new_b_vectors)

    np.save(simulation_data_paths.new_b_values_file, b_values)
    np.save(simulation_data_paths.new_b_vectors_file, b_vectors)

    logging.info(f'New b-values shape: {b_values.shape}')
    logging.info(f'New b-vectors shape: {b_vectors.shape}')
    logging.info('')


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

    indices_1 = np.random.permutation(num_samples)
    indices_2 = np.random.permutation(num_samples)
    
    train_indices = np.column_stack((indices_1[:train_samples], 
                                     indices_2[:train_samples]))
    eval_indices = np.column_stack((indices_1[train_samples:train_samples + eval_samples], 
                                    indices_2[train_samples:train_samples + eval_samples]))
    test_indices = np.column_stack((indices_1[train_samples + eval_samples:], 
                                    indices_2[train_samples + eval_samples:]))

    np.save(simulation_data_paths.train_indices_file, train_indices)
    np.save(simulation_data_paths.eval_indices_file, eval_indices)
    np.save(simulation_data_paths.test_indices_file, test_indices)


    ## SNR DISTRIBUTION AND SAMPLING

    snr_distribution = get_snr_distribution(simulation_data_hparams.probabilities, 
                                            simulation_data_hparams.snr_bounds, 
                                            num_samples)
   
    train_snrs = np.random.choice(snr_distribution, size=train_indices.shape[0])
    eval_snrs = np.random.choice(snr_distribution, size=eval_indices.shape[0])

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
    plt.close()


    ## SIMULATION TRAIN DATA

    print('Creating training data')
    noisy_train_signals, train_d_tensors, train_d_star_tensors, train_f_values \
        = get_noisy_biexp_signals(train_indices, train_snrs, d_tensors, b_values, b_vectors)
    
    logging.info(f'Training data: min = {noisy_train_signals.min()} - max = {noisy_train_signals.max()}')

    np.save(simulation_data_paths.train_sim_data_file, noisy_train_signals)
    np.save(simulation_data_paths.train_d_tensors_file, train_d_tensors)
    np.save(simulation_data_paths.train_d_star_tensors_file, train_d_star_tensors)
    np.save(simulation_data_paths.train_f_values_file, train_f_values)
    
    
    ## SIMULATION EVAL DATA

    print('Creating evaluation data')
    noisy_eval_signals, eval_d_tensors, eval_d_star_tensors, eval_f_values \
        = get_noisy_biexp_signals(eval_indices, eval_snrs, d_tensors, b_values, b_vectors)

    logging.info(f'Evaluation data: min = {noisy_eval_signals.min()} - max = {noisy_eval_signals.max()}')

    np.save(simulation_data_paths.eval_sim_data_file, noisy_eval_signals)
    np.save(simulation_data_paths.eval_d_tensors_file, eval_d_tensors)
    np.save(simulation_data_paths.eval_d_star_tensors_file, eval_d_star_tensors)
    np.save(simulation_data_paths.eval_f_values_file, eval_f_values)


    ## SIMULATION TEST DATA

    for test_snr in simulation_data_hparams.test_snrs:
        
        test_snrs = np.full(test_indices.shape[0], test_snr)

        print(f'Creating test data for SNR {test_snr}')
        noisy_test_signals, test_d_tensors, test_d_star_tensors, test_f_values \
            = get_noisy_biexp_signals(test_indices, test_snrs, d_tensors, b_values, b_vectors)

        logging.info(f'Test snr {test_snr} data: min = {noisy_test_signals.min()} - max = {noisy_test_signals.max()}')

        np.save(simulation_data_paths.test_data_template.format(int(test_snr)), noisy_test_signals)
        np.save(simulation_data_paths.test_d_tensors_template.format(int(test_snr)), test_d_tensors)
        np.save(simulation_data_paths.test_d_star_tensors_template.format(int(test_snr)), test_d_star_tensors)
        np.save(simulation_data_paths.test_f_values_template.format(int(test_snr)), test_f_values)


if __name__ == '__main__':
    main()
