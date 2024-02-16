'''Load simulation data
then fit non-linear least squares
on the 6 values of the spectral composition of the d-tensor
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
    threshold_eigval: float


@dataclass
class Paths:
    experiments_dir = os.path.join('baseline', 'template_2', 'experiments')
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


def reconstruct_tensor(params: np.ndarray, threshold_eigval: float) -> np.ndarray:

    # Split parameters after applying sigmoid
    x_angle, y_angle, z_angle, eigval_1, eigval_2_over_1, eigval_3_over_2 = 1/(1 + np.exp(-params))

    # Compute angles
    x_angle = x_angle * 2 * np.pi
    y_angle = y_angle * 2 * np.pi
    z_angle = z_angle * 2 * np.pi

    # Compute eigenvalues
    eigval_1 = eigval_1 * threshold_eigval
    eigval_2 = eigval_2_over_1 * eigval_1
    eigval_3 = eigval_3_over_2 * eigval_2

    # Create the roation matrices around the x axis.
    R_x = np.zeros((3, 3))
    R_x[0, 0] = 1
    R_x[1, 1] = np.cos(x_angle)
    R_x[1, 2] = -np.sin(x_angle)
    R_x[2, 1] = np.sin(x_angle)
    R_x[2, 2] = np.cos(x_angle)

    # Create the roation matrices around the y axis.
    R_y = np.zeros((3, 3))
    R_y[0, 0] = np.cos(y_angle)
    R_y[0, 2] = np.sin(y_angle)
    R_y[1, 1] = 1
    R_y[2, 0] = -np.sin(y_angle)
    R_y[2, 2] = np.cos(y_angle)

    # Create the roation matrices around the z axis.
    R_z = np.zeros((3, 3))
    R_z[0, 0] = np.cos(z_angle)
    R_z[0, 1] = -np.sin(z_angle)
    R_z[1, 0] = np.sin(z_angle)
    R_z[1, 1] = np.cos(z_angle)
    R_z[2, 2] = 1

    # Create the rotation matrix
    R = R_x @ R_y @ R_z

    # Create the diagonal matrix of eigenvalues
    D = np.diag([eigval_1, eigval_2, eigval_3])

    # Reconstruct the diffusion tensor
    D = R @ D @ R.T

    return D


def main():

    paths = Paths()


    ## HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--simulation_data_experiment_path', type=str, required=True)
    parser.add_argument('--threshold_eigval', type=float, required=True)
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

    def loss(params, S, S0, g, b_values, threshold_eigval):
        D = reconstruct_tensor(params, threshold_eigval)
        return S0 * np.exp(- b_values * np.einsum('bi,ij,bj->b', g, D, g)) - S

    d_tensor_error = 0.0
    recon_signal_error = 0.0
    invalid_d_tensors = 0
    predicted_d_tensors = np.zeros_like(d_tensors)

    for i in tqdm(range(noisy_signals.shape[0])):
        
        S = noisy_signals[i, :]
        S0 = 1.0 # noisy_signals[i, b_values == 0].mean()

        g = gradients
        
        # randomly initialize params in [-1, 1]
        params = np.random.rand(6) * 2 - 1

        # minimize the loss function
        result = least_squares(loss, params, args=(S, S0, g, b_values, HP.threshold_eigval))

        # reconstruct tensor
        D = reconstruct_tensor(result.x, HP.threshold_eigval)
        
        # try Cholesky decomposition
        try:
            _ = np.linalg.cholesky(D)
        except:
            invalid_d_tensors += 1

        predicted_d_tensors[i] = D
        
        d_tensor_error += np.mean((D - d_tensors[i])**2)
        recon_signal_error += np.mean((S0 * np.exp(- b_values * np.einsum('bi,ij,bj->b', g, D, g)) - S)**2)
    
    d_tensor_error = d_tensor_error / noisy_signals.shape[0]
    recon_signal_error = recon_signal_error / noisy_signals.shape[0]

    torch.save(torch.tensor(predicted_d_tensors).float(), paths.predicted_d_tensors_file)

    logging.info(f'invalid_d_tensors: {invalid_d_tensors}')
    logging.info(f'mean squared error: {d_tensor_error}')
    logging.info(f'reconstructed signal error: {recon_signal_error}')


if __name__ == '__main__':
    main()
