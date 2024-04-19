'''Load ground truth diffusion tensors.
Keep only the test set diffusion tensors.
Load the predicted diffusion tensors on the test set
from a fc_network_analysis experiment.
Compute the eigenvalues
of the ground truth and predicted diffusion tensors.
Sort them in descending order.
Create a Bland-Altman plot comparing the eigenvalues.'''


import os
import pickle
import logging
from typing import Protocol
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


class GroundTruthPaths(Protocol):
    d_tensors_file: str


class SimulationDataHyperparameters(Protocol):
    ground_truth_paths_pkl: str


class SimulationDataPaths(Protocol):
    hyperparameters_file: str
    test_indices_file: str


class FCNetworkHyperparameters(Protocol):
    simulation_data_paths_pkl: str


class FCNetworkPaths(Protocol):
    hyperparameters_file: str


class FCNetworkAnalysisHyperparameters(Protocol):
    fc_network_paths_pkl: str


class FCNetworkAnalysisPaths(Protocol):
    hyperparameters_file: str
    pred_d_tensors_file: str


@dataclass
class BlandAltmanPlotHyperparameters:
    fc_network_analysis_paths_pkl: str


class BlandAltmanPlotPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('bland_altman_plot', 'template_4', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')

        self.bland_altman_plot_file = os.path.join(self.experiment_path, 'bland_altman_plot.png')


def main():
    
    ## BLAND-ALTMAN PLOT PATHS

    bland_altman_plot_paths = BlandAltmanPlotPaths()

    print(f'Experiment path: {bland_altman_plot_paths.experiment_path}')

    os.makedirs(bland_altman_plot_paths.experiment_path)

    with open(bland_altman_plot_paths.paths_file, 'wb') as f:
        pickle.dump(bland_altman_plot_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=bland_altman_plot_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info('Blant-Altman plot experiment:')
    logging.info(bland_altman_plot_paths.experiment_path)
    logging.info('')


    ## HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--fc_network_analysis_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    bland_altman_plot_hparams = BlandAltmanPlotHyperparameters(**vars(args))

    with open(bland_altman_plot_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(bland_altman_plot_hparams, f)
    
    logging.info('Blant-Altman plot hyperparameters:')
    for key, value in vars(bland_altman_plot_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## FC NETWORK ANALYSIS PATHS

    with open(bland_altman_plot_hparams.fc_network_analysis_paths_pkl, 'rb') as f:
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
    

    ## DATA

    d_tensors: np.ndarray = np.load(ground_truth_paths.d_tensors_file)

    test_indices: np.ndarray = np.load(sim_data_paths.test_indices_file)

    test_d_tensors = d_tensors[test_indices]

    pred_d_tensors = np.load(fc_network_analysis_paths.pred_d_tensors_file)


    ## EIGENVALUES

    test_eigvals = np.linalg.eigvals(test_d_tensors).real
    pred_eigvals = np.linalg.eigvals(pred_d_tensors).real

    # Get the indices that would sort the eigenvalues in descending order    
    test_sort_indices = np.argsort(test_eigvals, axis=1)[:, ::-1]    
    pred_sort_indices = np.argsort(pred_eigvals, axis=1)[:, ::-1]

    # Use these indices to sort the eigenvalues    
    test_eigvals = np.take_along_axis(test_eigvals, test_sort_indices, axis=1)
    pred_eigvals = np.take_along_axis(pred_eigvals, pred_sort_indices, axis=1)


    ## BLAND-ALTMAN PLOT

    fig, axs = plt.subplots(1, 3, figsize=(21, 5))
    fig.suptitle('Bland-Altman plot for the eigenvalues')

    for col in range(3):

        mean = (test_eigvals[:, col] + pred_eigvals[:, col]) / 2
        diff = pred_eigvals[:, col] - test_eigvals[:, col]

        x = mean
        y = diff

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
        
        diff_mean = diff.mean()
        diff_std = diff.std()
        upper_limit = diff_mean + 1.96 * diff_std
        lower_limit = diff_mean - 1.96 * diff_std
        
        axs[col].axhline(y=diff_mean, color='k', linestyle='--')
        axs[col].axhline(y=upper_limit, color='r', linestyle='--')
        axs[col].axhline(y=lower_limit, color='r', linestyle='--')

        axs[col].set_xlabel('mean')
        axs[col].set_ylabel('diff (pred - true)')
        axs[col].set_title(f'Eigenvalue {col+1}')

        # Create a new axes for the histogram, on the right of the current axes
        divider = make_axes_locatable(axs[col])
        ax_hist = divider.append_axes("right", size=1.2, pad=0.1, sharey=axs[col])

        # Plot the histogram
        ax_hist.hist(diff, bins=200, orientation='horizontal', color='gray', alpha=0.5)

        # Hide ticks from both axes
        ax_hist.yaxis.set_visible(False)
        ax_hist.xaxis.set_visible(False)

        logging.info(f'Eigenvalue {col+1}: ' + \
                     f'mean diff (pred - true): {diff_mean} ' + \
                     f'std: {diff_std} ' + \
                     f'upper limit: {upper_limit} ' + \
                     f'lower limit: {lower_limit}')

    plt.tight_layout()
    plt.savefig(bland_altman_plot_paths.bland_altman_plot_file)


if __name__ == "__main__":
    main()
