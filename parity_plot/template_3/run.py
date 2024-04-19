'''Load ground truth diffusion tensors.
Keep only the test set diffusion tensors.
Load the predicted diffusion tensors on the test set
from a fc_network_analysis experiment.
Compute the eigenvalues and eigenvectors
of the ground truth and predicted diffusion tensors.
Sort them in descending order.
Compute the Pearson correlation between 
the ground truth and predicted eigenvalues.
Create a parity plot comparing the eigenvalues.
Create a histogram plot comparing the eigenvectors.'''


import os
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class GroundTruthHyperparameters(Protocol):
    threshold_eigval: float


class GroundTruthPaths(Protocol):
    d_tensors_file: str
    hyperparameters_file: str


class SimulationDataHyperparameters(Protocol):
    ground_truth_paths_pkl: str


class SimulationDataPaths(Protocol):
    hyperparameters_file: str
    test_indices_file: str


class FCNetworkHyperparameters(Protocol):
    simulation_data_paths_pkl: str
    threshold_eigval: float


class FCNetworkPaths(Protocol):
    hyperparameters_file: str


class FCNetworkAnalysisHyperparameters(Protocol):
    fc_network_paths_pkl: str


class FCNetworkAnalysisPaths(Protocol):
    hyperparameters_file: str
    pred_d_tensors_file: str


@dataclass
class ParityPlotHyperparameters:
    fc_network_analysis_paths_pkl: str


class ParityPlotPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('parity_plot', 'template_3', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')
        
        self.parity_plot_file = os.path.join(self.experiment_path, 'parity_plot.png')
        self.histogram_plot_file = os.path.join(self.experiment_path, 'histogram.png')


def main():

    ## PARITY PLOT PATHS

    parity_plot_paths = ParityPlotPaths()

    print(f'Experiment path: {parity_plot_paths.experiment_path}')

    os.makedirs(parity_plot_paths.experiment_path)

    with open(parity_plot_paths.paths_file, 'wb') as f:
        pickle.dump(parity_plot_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=parity_plot_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info('Parity plot experiment:')
    logging.info(parity_plot_paths.experiment_path)
    logging.info('')


    ## PARITY PLOT HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--fc_network_analysis_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    parity_plot_hparams = ParityPlotHyperparameters(**vars(args))

    with open(parity_plot_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(parity_plot_hparams, f)
    
    logging.info('Parity plot hyperparameters:')
    for key, value in vars(parity_plot_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## FC NETWORK ANALYSIS PATHS

    with open(parity_plot_hparams.fc_network_analysis_paths_pkl, 'rb') as f:
        fc_network_analysis_paths: FCNetworkAnalysisPaths = pickle.load(f)
    

    ## FC NETWORK ANALYSIS HYPERPARAMETERS

    with open(fc_network_analysis_paths.hyperparameters_file, 'rb') as f:
        fc_network_analysis_hparams: FCNetworkAnalysisHyperparameters = pickle.load(f)
    

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
    

    ## GROUND TRUTH PATHS

    with open(sim_data_hparams.ground_truth_paths_pkl, 'rb') as f:
        ground_truth_paths: GroundTruthPaths = pickle.load(f)
    

    ## GROUND TRUTH HYPERPARAMETERS

    with open(ground_truth_paths.hyperparameters_file, 'rb') as f:
        ground_truth_hparams: GroundTruthHyperparameters = pickle.load(f)
    

    ## DATA

    d_tensors: np.ndarray = np.load(ground_truth_paths.d_tensors_file)

    test_indices: np.ndarray = np.load(sim_data_paths.test_indices_file)

    test_d_tensors = d_tensors[test_indices]

    pred_d_tensors = np.load(fc_network_analysis_paths.pred_d_tensors_file)


    ## EIGENVALUES AND EIGENVECTORS

    test_eigvals, test_eigvecs = np.linalg.eig(test_d_tensors)
    test_eigvals = test_eigvals.real
    test_eigvecs = test_eigvecs.real

    pred_eigvals, pred_eigvecs = np.linalg.eig(pred_d_tensors)
    pred_eigvals = pred_eigvals.real
    pred_eigvecs = pred_eigvecs.real

    # Get the indices that would sort the eigenvalues in descending order    
    test_sort_indices = np.argsort(test_eigvals, axis=1)[:, ::-1]    
    pred_sort_indices = np.argsort(pred_eigvals, axis=1)[:, ::-1]

    # Use these indices to sort the eigenvalues and eigenvectors
    
    test_eigvals = np.take_along_axis(test_eigvals, test_sort_indices, axis=1)
    test_eigvecs = np.take_along_axis(test_eigvecs, np.expand_dims(test_sort_indices, axis=1), axis=2)
    
    pred_eigvals = np.take_along_axis(pred_eigvals, pred_sort_indices, axis=1)
    pred_eigvecs = np.take_along_axis(pred_eigvecs, np.expand_dims(pred_sort_indices, axis=1), axis=2)


    ## PEARSON CORRELATION

    for col in range(3):
        pearson_result = scipy.stats.pearsonr(test_eigvals[:, col], pred_eigvals[:, col])
        logging.info(f'Eigenvalue {col+1} : ' + \
                     f'Pearson correlation coefficient = {pearson_result.statistic} ' + \
                     f'p-value = {pearson_result.pvalue}')
    logging.info('')

    
    ## PARITY PLOT

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Parity plot for diffusion tensor eigenvalues')

    for col in range(3):

        x = test_eigvals[:, col]
        y = pred_eigvals[:, col]

        # compute the point densities
        hist, xedges, yedges = np.histogram2d(x, y, bins=100)
        xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
        yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
        c = hist[xidx, yidx]

        # scatter plot with points colored by density
        scatter = axs[col].scatter(
            x, y, 
            c=c, 
            s=1, 
            norm=LogNorm(),  # Emphasize differences in density
            cmap='viridis'  # Use the 'viridis' colormap
        )

        # add a colorbar for the scatter plot
        fig.colorbar(scatter, ax=axs[col], label='Density')

        # identity line
        axs[col].plot(
            [0.0, ground_truth_hparams.threshold_eigval], 
            [0.0, ground_truth_hparams.threshold_eigval], 
            'k--')

        # labels
        axs[col].set_xlabel('Ground Truth')
        axs[col].set_ylabel('Predictions')
        axs[col].set_title(f'Eigenvalue {col+1}')
    
    plt.tight_layout()
    plt.savefig(parity_plot_paths.parity_plot_file)


    ## HISTOGRAM PLOT

    test_eigvecs_norms = np.linalg.norm(test_eigvecs, axis=1)
    pred_eigvecs_norms = np.linalg.norm(pred_eigvecs, axis=1)

    dot_products = np.sum(test_eigvecs * pred_eigvecs, axis=1)

    cos_sims = dot_products / (test_eigvecs_norms * pred_eigvecs_norms)

    abs_cos_sims = np.abs(cos_sims)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Histogram plot for diffusion tensor eigenvectors')

    for col in range(3):

        x = abs_cos_sims[:, col]

        axs[col].hist(x, bins=100)
        axs[col].set_xlabel('Cosine similarity')
        axs[col].set_ylabel('Count')
        axs[col].set_title(f'Eigenvector {col+1}')

    plt.tight_layout()
    plt.savefig(parity_plot_paths.histogram_plot_file)


if __name__ == '__main__':
    main()
