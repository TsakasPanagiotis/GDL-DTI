'''Load predicted d-tensors from baseline model
for spectral composition parameters.
Load checkpoint of trained basic MLP model that uses spectral composition
but predict sin and cos of angles instead of angles 
by using tanh instead of sigmoid.
Load d-tensors and noisy signals from simulation data 
according to checkpoint's hyperparameters.
Filter d-tensors and noisy signals according to checkpoint's hyperparameters.
Filter predicted d-tensors from baseline model according to checkpoint's hyperparameters.
Evaluate model on all filtered d-tensors and noisy signals.
Create Bland-Altman plot for every element of the d-tensors 
just only for the upper triangular part.'''


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
from torch.utils.data import DataLoader

from fc_network.template_2.run import Hyperparameters, DiffusionDataset, SpectralDiffusionNet


@dataclass
class Paths:
    experiments_dir = os.path.join('bland_altman_plot', 'template_1', 'experiments')
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
    parser.add_argument('--baseline_tensors_path', type=str, required=True)
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


    ## LOAD DATA

    baseline_tensors = torch.load(args.baseline_tensors_path)

    simulation_tensors = torch.load(paths.d_tensors_file)
    noisy_signals = torch.load(paths.noisy_signals_file)

    logging.info(f'Before filtering - simulation_tensors: {simulation_tensors.shape}')
    logging.info(f'Before filtering - baseline_tensors: {baseline_tensors.shape}')

    if baseline_tensors.shape != simulation_tensors.shape:
        logging.error('baseline_tensors and simulation_tensors have different shapes')
        raise ValueError('baseline_tensors and simulation_tensors have different shapes')


    ## FILTER DATA

    eigenvalues = torch.linalg.eigvalsh(simulation_tensors)
    filter_indices = torch.where(eigenvalues[:,-1] <= HP.threshold_eigval)[0]
    
    simulation_tensors = simulation_tensors[filter_indices]
    noisy_signals = noisy_signals[filter_indices]

    baseline_tensors = baseline_tensors[filter_indices]

    logging.info(f'After filtering - simulation_tensors: {simulation_tensors.shape}')
    logging.info(f'After filtering - baseline_tensors: {baseline_tensors.shape}')


    ## DATASETS AND DATALOADERS

    dataset = DiffusionDataset(simulation_tensors, noisy_signals)

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

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Bland-Altman plot for the diffusion tensor elements')

    # iterate over the elements of the upper triangular part
    for row in range(3):
        for col in range(row, 3):

            mean = (baseline_tensors[:, row, col] + pred_d_tensors[:, row, col]) / 2
            diff = baseline_tensors[:, row, col] - pred_d_tensors[:, row, col]

            axs[row, col].scatter(
                x=mean, 
                y=diff,
                s=1)
            
            diff_mean = diff.mean()
            upper_limit = diff_mean + 1.96 * diff.std()
            lower_limit = diff_mean - 1.96 * diff.std()
            
            axs[row, col].axhline(y=diff_mean, color='k', linestyle='--')
            axs[row, col].axhline(y=upper_limit, color='r', linestyle='--')
            axs[row, col].axhline(y=lower_limit, color='r', linestyle='--')

            axs[row, col].set_xlabel('mean')
            axs[row, col].set_ylabel('diff')
            axs[row, col].set_title(f'Element ({row},{col})')

    # remove the empty subplots of the lower triangular part
    for row in range(3):
        for col in range(row):
            fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.savefig(paths.bland_altman_plot_file)


if __name__ == "__main__":
    main()