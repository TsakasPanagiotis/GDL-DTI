'''Load d-tensors from simulation data.
Load predicted d-tensors from baseline model
for spectral composition parameters.
Create Bland-Altman plot for every element of the d-tensors 
just only for the upper triangular part.
The plot is colored by the density of the points.
The plot has a histogram of the y-axis.'''


import os
import logging
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


@dataclass
class Hyperparameters:
    baseline_tensors_path: str
    simulation_tensors_path: str


@dataclass
class Paths:
    experiments_dir = os.path.join('bland_altman_plot', 'template_2.1', 'experiments')
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    @property
    def experiment_path(self):
        return os.path.join(self.experiments_dir, self.experiment_name)
    
    @property
    def log_file(self):
        return os.path.join(self.experiment_path, 'log.txt')
    
    @property
    def bland_altman_plot_file(self):
        return os.path.join(self.experiment_path, 'bland_altman_plot.png')


def main():
    
    paths = Paths()


    ## HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--baseline_tensors_path', type=str, required=True)
    parser.add_argument('--simulation_tensors_path', type=str, required=True)
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


    ## LOAD DATA

    baseline_tensors = torch.load(HP.baseline_tensors_path)
    simulation_tensors = torch.load(HP.simulation_tensors_path)

    logging.info(f'baseline_tensors: {baseline_tensors.shape}')
    logging.info(f'simulation_tensors: {simulation_tensors.shape}')

    if baseline_tensors.shape != simulation_tensors.shape:
        logging.error('baseline_tensors and simulation_tensors have different shapes')
        raise ValueError('baseline_tensors and simulation_tensors have different shapes')

    
    ## BLAND-ALTMAN PLOT

    fig, axs = plt.subplots(3, 3, figsize=(21, 15))
    fig.suptitle('Bland-Altman plot for the diffusion tensor elements')

    # iterate over the elements of the upper triangular part
    for row in range(3):
        for col in range(row, 3):

            mean = (simulation_tensors[:, row, col] + baseline_tensors[:, row, col]) / 2
            diff = simulation_tensors[:, row, col] - baseline_tensors[:, row, col]

            x = mean.numpy()
            y = diff.numpy()

            # compute the point densities
            hist, xedges, yedges = np.histogram2d(x, y, bins=100)
            xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
            yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
            c = hist[xidx, yidx]

            # scatter plot with points colored by density
            scatter = axs[row, col].scatter(
                x, y, 
                c=c, 
                s=1, 
                norm=LogNorm(),  # Emphasize differences in density
                cmap='viridis'  # Use the 'viridis' colormap
            )

            # add a colorbar for the scatter plot
            fig.colorbar(scatter, ax=axs[row, col], label='Density')
            
            diff_mean = diff.mean()
            upper_limit = diff_mean + 1.96 * diff.std()
            lower_limit = diff_mean - 1.96 * diff.std()
            
            axs[row, col].axhline(y=diff_mean, color='k', linestyle='--')
            axs[row, col].axhline(y=upper_limit, color='r', linestyle='--')
            axs[row, col].axhline(y=lower_limit, color='r', linestyle='--')

            axs[row, col].set_xlabel('mean')
            axs[row, col].set_ylabel('diff')
            axs[row, col].set_title(f'Element ({row},{col})')

            # Create a new axes for the histogram, on the right of the current axes
            divider = make_axes_locatable(axs[row, col])
            ax_hist = divider.append_axes("right", size=1.2, pad=0.1, sharey=axs[row, col])

            # Plot the histogram
            ax_hist.hist(diff, bins=200, orientation='horizontal', color='gray', alpha=0.5)

            # Hide ticks from both axes
            ax_hist.yaxis.set_visible(False)
            ax_hist.xaxis.set_visible(False)

    # remove the empty subplots of the lower triangular part
    for row in range(3):
        for col in range(row):
            fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.savefig(paths.bland_altman_plot_file)


if __name__ == "__main__":
    main()
