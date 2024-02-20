'''Load checkpoint of trained basic MLP model that uses spectral composition
but predict sin and cos of angles instead of angles 
by using tanh instead of sigmoid.
Load d-tensors and noisy signals from simulation data 
according to checkpoint's hyperparameters.
Filter d-tensors and noisy signals according to checkpoint's hyperparameters.
Evaluate model on all filtered d-tensors and noisy signals.
Create Bland-Altman plot for every element of the d-tensors 
just only for the upper triangular part.
The plot is colored by the density of the points.
The plot has a histogram of the y-axis.'''


import os
import sys
sys.path.append(os.getcwd())
import random
import logging
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fc_network.template_2.run import Hyperparameters, DiffusionDataset, SpectralDiffusionNet


@dataclass
class Paths:
    experiments_dir = os.path.join('bland_altman_plot', 'template_3.1', 'experiments')
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    simulation_data_experiment_path = ''
    
    @property
    def experiment_path(self):
        return os.path.join(self.experiments_dir, self.experiment_name)
    
    @property
    def log_file(self):
        return os.path.join(self.experiment_path, 'log.txt')
    
    @property
    def bland_altman_plot_file(self):
        return os.path.join(self.experiment_path, 'bland_altman_plot.png')
    
    @property
    def d_tensors_file(self):
        return os.path.join(self.simulation_data_experiment_path, 'd_tensors.pt')
    
    @property
    def noisy_signals_file(self):
        return os.path.join(self.simulation_data_experiment_path, 'noisy_signals.pt')


def evaluate_model(model, dataloader, device) -> torch.Tensor:
      
    pred_d_tensors = []
    model.eval()    
    
    with torch.no_grad():
        for d_tensors, noisy_signals in tqdm(dataloader):
            
            d_tensors = d_tensors.to(device)
            noisy_signals = noisy_signals.to(device)
            
            batch_pred_d_tensors = model(noisy_signals)
            pred_d_tensors.append(batch_pred_d_tensors)

    pred_d_tensors = torch.cat(pred_d_tensors, dim=0)
    
    return pred_d_tensors


def main():
    
    paths = Paths()


    ## HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path)
    
    HP = Hyperparameters(**checkpoint['hyperparameters'])

    paths.simulation_data_experiment_path = HP.simulation_data_experiment_path


    ## LOGGING

    os.makedirs(paths.experiment_path)
    
    logging.basicConfig(
        filename=paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f'checkpoint_path: {args.checkpoint_path}')
    logging.info(f'simulation_tensors_path: {paths.d_tensors_file}')


    ## DEVICE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f'Using {device} device')


    ## REPRODUCIBILITY

    # Set seed for Python's random module
    random.seed(HP.seed)

    # Set seed for NumPy's random module
    np.random.seed(HP.seed)

    # Set seed for PyTorch's CPU RNG
    torch.manual_seed(HP.seed)

    # If CUDA is available, set seed for CUDA RNGs and enable deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed(HP.seed)
        torch.cuda.manual_seed_all(HP.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    ## FILTER DATA

    d_tensors = torch.load(paths.d_tensors_file)
    noisy_signals = torch.load(paths.noisy_signals_file)

    logging.info(f'Before filtering - d_tensors: {d_tensors.shape}')

    eigenvalues = torch.linalg.eigvalsh(d_tensors)
    filter_indices = torch.where(eigenvalues[:,-1] <= HP.threshold_eigval)[0]
    
    d_tensors = d_tensors[filter_indices]
    noisy_signals = noisy_signals[filter_indices]

    logging.info(f'After filtering - d_tensors: {d_tensors.shape}')


    ## DATASETS AND DATALOADERS

    dataset = DiffusionDataset(d_tensors, noisy_signals)

    dataloader = DataLoader(dataset, batch_size=HP.batch_size)


    ## LOAD TRAINED MODEL

    spectral_dnet = SpectralDiffusionNet(
        input_size=noisy_signals.shape[1],
        hidden_sizes=HP.hidden_sizes,
        output_size=HP.output_size,
        threshold_eigval=HP.threshold_eigval
    ).to(device)

    spectral_dnet.load_state_dict(checkpoint['model_state_dict'])


    ## EVALUATE MODEL

    pred_d_tensors = evaluate_model(spectral_dnet, dataloader, device)


    ## BLAND-ALTMAN PLOT

    fig, axs = plt.subplots(3, 3, figsize=(21, 15))
    fig.suptitle('Bland-Altman plot for the diffusion tensor elements')

    # iterate over the elements of the upper triangular part
    for row in range(3):
        for col in range(row, 3):

            mean = (d_tensors[:, row, col] + pred_d_tensors[:, row, col]) / 2
            diff = d_tensors[:, row, col] - pred_d_tensors[:, row, col]

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
