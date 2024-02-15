'''Load simulation data
then fit non-linear least squares
on the 6 values of the upper triangle of the d-tensor
and store the predicted d-tensors.'''


import os
import logging
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import least_squares


@dataclass
class Hyperparameters:
    
    simulation_data_experiment_path: str


@dataclass
class Paths:
    experiments_dir = os.path.join('baseline', 'template_1', 'experiments')
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    simulation_data_experiment_path = ''
    
    @property
    def experiment_path(self):
        return os.path.join(self.experiments_dir, self.experiment_name)
    
    @property
    def log_file(self):
        return os.path.join(self.experiment_path, 'log.txt')
    
    @property
    def predicted_d_tensors_file(self):
        return os.path.join(self.experiment_path, 'predicted_d_tensors.pt')
    
    @property
    def d_tensors_file(self):
        return os.path.join(self.simulation_data_experiment_path, 'd_tensors.pt')
    
    @property
    def noisy_signals_file(self):
        return os.path.join(self.simulation_data_experiment_path, 'noisy_signals.pt')
    
    @property
    def b_values_file(self):
        return os.path.join(self.simulation_data_experiment_path, 'b_values.pt')
    
    @property
    def gradients_file(self):
        return os.path.join(self.simulation_data_experiment_path, 'gradients.pt')


def main():

    paths = Paths()


    ## HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--simulation_data_experiment_path', type=str, required=True)
    args = parser.parse_args()

    HP = Hyperparameters(**vars(args))

    paths.simulation_data_experiment_path = HP.simulation_data_experiment_path


    ## LOGGING

    os.makedirs(paths.experiment_path)
    
    logging.basicConfig(
        filename=paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    for key, value in vars(HP).items():
        logging.info(f'{key}: {value}')


    ## LOAD RESULTS
    
    d_tensors = torch.load(paths.d_tensors_file).numpy()
    noisy_signals = torch.load(paths.noisy_signals_file).numpy()
    b_values = torch.load(paths.b_values_file).numpy()
    gradients = torch.load(paths.gradients_file).numpy()

    logging.info(f'd_tensors: {d_tensors.shape}')
    logging.info(f'noisy_signals: {noisy_signals.shape}')
    logging.info(f'b_values: {b_values.shape}')
    logging.info(f'gradients: {gradients.shape}')

    
    ## NON LINEAR LEAST SQUARES

    def loss(params, X, S, S0, b_values):
        return S0 * np.exp( - b_values * (X @ params)) - S

    d_tensor_error = 0.0
    recon_signal_error = 0.0
    invalid_d_tensors = 0
    predicted_d_tensors = np.zeros_like(d_tensors)

    for i in tqdm(range(noisy_signals.shape[0])):
                    
        # calculate y
        S = noisy_signals[i, :]
        S0 = 1.0 # noisy_signals[i, b_values == 0].mean()

        # calculate X
        g = gradients
        X = np.array([g[:,0]**2, 2*g[:,0]*g[:,1], 2*g[:,0]*g[:,2], 
                      g[:,1]**2, 2*g[:,1]*g[:,2], g[:,2]**2]).T
        
        # randomly initialize params in [-0.5, 0.5]
        params = np.random.rand(6) - 0.5

        # minimize the loss function
        result = least_squares(loss, params, args=(X, S, S0, b_values))

        # reconstruct tensor
        D = np.array([[result.x[0], result.x[1], result.x[2]],
                    [result.x[1], result.x[3], result.x[4]],
                    [result.x[2], result.x[4], result.x[5]]])
        
        # try Cholesky decomposition
        try:
            _ = np.linalg.cholesky(D)
        except:
            invalid_d_tensors += 1

        predicted_d_tensors[i] = D
        
        d_tensor_error += np.mean((D - d_tensors[i])**2)
        recon_signal_error += np.mean((S0 * np.exp( - b_values * (X @ params)) - S)**2)
    
    d_tensor_error = d_tensor_error / noisy_signals.shape[0]
    recon_signal_error = recon_signal_error / noisy_signals.shape[0]

    torch.save(torch.tensor(predicted_d_tensors).float(), paths.predicted_d_tensors_file)

    logging.info(f'invalid_d_tensors: {invalid_d_tensors}')
    logging.info(f'mean squared error: {d_tensor_error}')
    logging.info(f'reconstructed signal error: {recon_signal_error}')


if __name__ == '__main__':
    main()
