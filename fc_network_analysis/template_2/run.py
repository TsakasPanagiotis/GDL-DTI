'''Load checkpoint of trained basic MLP model that uses spectral composition
but predict sin and cos of angles instead of angles by using tanh instead of sigmoid.
Re-create the training and validation sets.
Count the number of invalid predicted d-tensors.'''

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
from torch.utils.data import DataLoader

from fc_network.template_2.run import Hyperparameters, DiffusionDataset, SpectralDiffusionNet


@dataclass
class Paths:
    experiments_dir = os.path.join('analysis', 'template_2', 'experiments')
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    simulation_data_experiment_path = ''

    @property
    def experiment_path(self):
        return os.path.join(self.experiments_dir, self.experiment_name)
    
    @property
    def log_file(self):
        return os.path.join(self.experiment_path, 'log.txt')
    
    @property
    def d_tensors_file(self):
        return os.path.join(self.simulation_data_experiment_path, 'd_tensors.pt')
    
    @property
    def noisy_signals_file(self):
        return os.path.join(self.simulation_data_experiment_path, 'noisy_signals.pt')


def evaluate_model(model, dataloader, device) -> int:

    invalid_count = 0        
    model.eval()    
    
    with torch.no_grad():
        for d_tensor, noisy_signal in tqdm(dataloader):
            
            d_tensor = d_tensor.to(device)
            noisy_signal = noisy_signal.to(device)
            
            pred_d_tensor = model(noisy_signal)
            _, info = torch.linalg.cholesky_ex(pred_d_tensor)
            invalid_count += (info != 0).sum().item()
    
    return invalid_count


def main() -> None:

    
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
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f'checkpoint_path: {args.checkpoint_path}')


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

    eigenvalues = torch.linalg.eigvalsh(d_tensors)
    filter_indices = torch.where(eigenvalues[:,-1] <= HP.threshold_eigval)[0]
    
    d_tensors = d_tensors[filter_indices]
    noisy_signals = noisy_signals[filter_indices]


    ## SHUFFLE AND SPLIT INDICES

    num_data = len(d_tensors)
    num_train = int(HP.train_percent * num_data)

    generator = torch.Generator().manual_seed(HP.seed)
    indices = torch.randperm(num_data, generator=generator)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]


    ## DATASETS AND DATALOADERS

    train_dataset = DiffusionDataset(d_tensors[train_indices], noisy_signals[train_indices])
    val_dataset = DiffusionDataset(d_tensors[val_indices], noisy_signals[val_indices])

    train_dataloader = DataLoader(train_dataset, batch_size=HP.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=HP.batch_size)


    ## LOAD TRAINED MODEL

    spectral_dnet = SpectralDiffusionNet(
        input_size=noisy_signals.shape[1],
        hidden_sizes=HP.hidden_sizes,
        output_size=HP.output_size,
        threshold_eigval=HP.threshold_eigval
    ).to(device)

    spectral_dnet.load_state_dict(checkpoint['model_state_dict'])
    

    ## EVALUATE MODEL

    invalid_train_count = evaluate_model(spectral_dnet, train_dataloader, device)
    invalid_val_count = evaluate_model(spectral_dnet, val_dataloader, device)

    logging.info(f'Invalid train count: {invalid_train_count}')
    logging.info(f'Invalid val count: {invalid_val_count}')


if __name__ == '__main__':
    main()
