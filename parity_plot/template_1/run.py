'''Load d-tensors from simulation data.
Load predicted d-tensors from baseline model.
Create parity plot for every element of the d-tensors
just only for the upper triangular part.'''


import os
import logging
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import matplotlib.pyplot as plt


@dataclass
class Hyperparameters:
    baseline_tensors_path: str
    simulation_tensors_path: str


@dataclass
class Paths:
    experiments_dir = os.path.join('parity_plot', 'template_1', 'experiments')
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    @property
    def experiment_path(self):
        return os.path.join(self.experiments_dir, self.experiment_name)
    
    @property
    def log_file(self):
        return os.path.join(self.experiment_path, 'log.txt')
    
    @property
    def parity_plot_file(self):
        return os.path.join(self.experiment_path, 'parity_plot.png')


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

    
    ## PARITY PLOT

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Parity plot for the diffusion tensor elements')

    # iterate over the elements of the upper triangular part
    for row in range(3):
        for col in range(row, 3):

            # scatter plot
            axs[row, col].scatter(
                x=simulation_tensors[:, row, col], 
                y=baseline_tensors[:, row, col],
                s=1)

            # identity line
            axs[row, col].plot(
                [simulation_tensors[:, row, col].min(), simulation_tensors[:, row, col].max()], 
                [simulation_tensors[:, row, col].min(), simulation_tensors[:, row, col].max()], 
                'k--')

            # labels
            axs[row, col].set_xlabel('Ground Truth')
            axs[row, col].set_ylabel('Predictions')
            axs[row, col].set_title(f'Element ({row},{col})')

    # remove the empty subplots of the lower triangular part
    for row in range(3):
        for col in range(row):
            fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.savefig(paths.parity_plot_file)


if __name__ == "__main__":
    main()
