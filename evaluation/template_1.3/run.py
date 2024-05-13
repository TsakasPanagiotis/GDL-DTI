'''Load ground truth diffusion tensors.
Keep only the test set diffusion tensors.
Load the predicted diffusion tensors on the test set
from a multiple fc_network_analysis experiment.
For each network alaysis sub-experiment create a subfolder.
Compute the eigenvalues and eigenvectors
of the ground truth and predicted diffusion tensors.
Sort them in descending order.
Create a parity plot comparing
either eigenvalues or Mean Diffusivity or Fractional Anisotropy
and Pearson correlation with p-value.
Create a histogram plot comparing the eigenvectors.
Create a Bland-Altman plot comparing
either eigenvalues or Mean Diffusivity or Fractional Anisotropy
with mean difference and standard deviation.'''


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
from mpl_toolkits.axes_grid1 import make_axes_locatable


class PearsonResult:
    def __init__(self, result) -> None:
        self.statistic = result.statistic
        self.pvalue = result.pvalue


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
    pred_d_tensors_files: dict[int, str]


@dataclass
class EvaluationHyperparameters:
    fc_network_analysis_paths_pkl: str


class EvaluationPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('evaluation', 'template_1.3', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')
        
        self.subfolder = os.path.join(self.experiment_path, '{epoch}')
        self.eigenvalues_parity_plot_file = os.path.join(self.experiment_path, '{epoch}', 'eigvals_parity_plot.png')
        self.mean_diffusivity_parity_plot_file = os.path.join(self.experiment_path, '{epoch}', 'md_parity_plot.png')
        self.fractional_anisotropy_parity_plot_file = os.path.join(self.experiment_path, '{epoch}', 'fa_parity_plot.png')
        self.eigenvectors_histogram_plot_file = os.path.join(self.experiment_path, '{epoch}', 'eigvecs_histogram.png')
        self.eigenvalues_bland_altman_plot_file = os.path.join(self.experiment_path, '{epoch}', 'eigvals_bland_altman_plot.png')
        self.mean_diffusivity_bland_altman_plot_file = os.path.join(self.experiment_path, '{epoch}', 'md_bland_altman_plot.png')
        self.fractional_anisotropy_bland_altman_plot_file = os.path.join(self.experiment_path, '{epoch}', 'fa_bland_altman_plot.png')


def main():

    ## EVALUATION PATHS

    evaluation_paths = EvaluationPaths()

    print(f'Experiment path: {evaluation_paths.experiment_path}')

    os.makedirs(evaluation_paths.experiment_path)

    with open(evaluation_paths.paths_file, 'wb') as f:
        pickle.dump(evaluation_paths, f)
    

    ## LOGGING

    logging.basicConfig(
        filename=evaluation_paths.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info('Evaluation experiment:')
    logging.info(evaluation_paths.experiment_path)
    logging.info('')


    ## EVALUATION HYPERPARAMETERS

    parser = ArgumentParser()
    parser.add_argument('--fc_network_analysis_paths_pkl', type=str, required=True)
    args = parser.parse_args()

    evaluation_hparams = EvaluationHyperparameters(**vars(args))

    with open(evaluation_paths.hyperparameters_file, 'wb') as f:
        pickle.dump(evaluation_hparams, f)
    
    logging.info('Evaluation hyperparameters:')
    for key, value in vars(evaluation_hparams).items():
        logging.info(f'{key}: {value}')
    logging.info('')


    ## FC NETWORK ANALYSIS PATHS

    with open(evaluation_hparams.fc_network_analysis_paths_pkl, 'rb') as f:
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
    

    for epoch, pred_d_tensors_file in fc_network_analysis_paths.pred_d_tensors_files.items():

        logging.info(f'Epoch: {epoch}')
        logging.info('')

        os.makedirs(evaluation_paths.subfolder.format(epoch=epoch))

        
        ## DATA

        d_tensors: np.ndarray = np.load(ground_truth_paths.d_tensors_file)

        test_indices: np.ndarray = np.load(sim_data_paths.test_indices_file)

        test_d_tensors = d_tensors[test_indices]

        pred_d_tensors = np.load(pred_d_tensors_file)


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


        ## ONLY KEEP EIGENVALUES LOWER THAN THE THRESHOLD

        valid_indices = pred_eigvals[:, 0] < ground_truth_hparams.threshold_eigval

        logging.info(f'Number of invalid diffusion tensors: {np.sum(~valid_indices)} / {pred_d_tensors.shape[0]}')
        logging.info(f'Percentage of invalid diffusion tensors: {np.sum(~valid_indices) / pred_d_tensors.shape[0] * 100:.2f}%')
        logging.info('')

        test_eigvals = test_eigvals[valid_indices]
        test_eigvecs = test_eigvecs[valid_indices]

        pred_eigvals = pred_eigvals[valid_indices]
        pred_eigvecs = pred_eigvecs[valid_indices]


        ## MEAN DIFFUSIVITY

        # MD = (lambda_1 + lambda_2 + lambda_3) / 3
        test_md = np.mean(test_eigvals, axis=1)
        pred_md = np.mean(pred_eigvals, axis=1)


        ## FRACTIONAL ANISOTROPY
        
        # FA = sqrt(1.5 * sum((lambda_i - MD)^2) / sum(lambda_i^2))
        test_fa = np.sqrt(1.5 * np.sum((test_eigvals - test_md[:, None])**2, axis=1) / np.sum(test_eigvals**2, axis=1))
        pred_fa = np.sqrt(1.5 * np.sum((pred_eigvals - pred_md[:, None])**2, axis=1) / np.sum(pred_eigvals**2, axis=1))


        ## PEARSON CORRELATION

        # Eigenvalues
        for col in range(3):
            pearson_result = PearsonResult(scipy.stats.pearsonr(test_eigvals[:, col], pred_eigvals[:, col]))
            logging.info(f'Eigenvalue {col+1} : ' + \
                        f'Pearson correlation coefficient = {pearson_result.statistic} ' + \
                        f'p-value = {pearson_result.pvalue}')
        logging.info('')

        # Mean Diffusivity
        pearson_result = PearsonResult(scipy.stats.pearsonr(test_md, pred_md))
        logging.info(f'Mean Diffusivity : ' + \
                    f'Pearson correlation coefficient = {pearson_result.statistic} ' + \
                    f'p-value = {pearson_result.pvalue}')
        logging.info('')

        # Fractional Anisotropy
        pearson_result = PearsonResult(scipy.stats.pearsonr(test_fa, pred_fa))
        logging.info(f'Fractional Anisotropy : ' + \
                    f'Pearson correlation coefficient = {pearson_result.statistic} ' + \
                    f'p-value = {pearson_result.pvalue}')
        logging.info('')

        
        ## PARITY PLOT FOR EIGENVALUES

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Parity Plot for Eigenvalues')

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
        plt.savefig(evaluation_paths.eigenvalues_parity_plot_file.format(epoch=epoch))
        plt.close()


        ## PARITY PLOT FOR MEAN DIFFUSIVITY

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.suptitle('Parity Plot for Mean Diffusivity')

        x = test_md
        y = pred_md

        # compute the point densities
        hist, xedges, yedges = np.histogram2d(x, y, bins=100)
        xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
        yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
        c = hist[xidx, yidx]

        # scatter plot with points colored by density
        scatter = ax.scatter(
            x, y, 
            c=c, 
            s=1, 
            norm=LogNorm(),  # Emphasize differences in density
            cmap='viridis'  # Use the 'viridis' colormap
        )

        # add a colorbar for the scatter plot
        fig.colorbar(scatter, ax=ax, label='Density')

        # identity line
        ax.plot(
            [0.0, ground_truth_hparams.threshold_eigval], 
            [0.0, ground_truth_hparams.threshold_eigval], 
            'k--')
        
        # labels
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predictions')

        plt.tight_layout()
        plt.savefig(evaluation_paths.mean_diffusivity_parity_plot_file.format(epoch=epoch))
        plt.close()


        ## PARITY PLOT FOR FRACTIONAL ANISOTROPY

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.suptitle('Parity Plot for Fractional Anisotropy')

        x = test_fa
        y = pred_fa

        # compute the point densities
        hist, xedges, yedges = np.histogram2d(x, y, bins=100)
        xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
        yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
        c = hist[xidx, yidx]

        # scatter plot with points colored by density
        scatter = ax.scatter(
            x, y, 
            c=c, 
            s=1, 
            norm=LogNorm(),  # Emphasize differences in density
            cmap='viridis'  # Use the 'viridis' colormap
        )

        # add a colorbar for the scatter plot
        fig.colorbar(scatter, ax=ax, label='Density')

        # identity line
        ax.plot(
            [0.0, 1.0], 
            [0.0, 1.0], 
            'k--')
        
        # labels
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predictions')

        plt.tight_layout()
        plt.savefig(evaluation_paths.fractional_anisotropy_parity_plot_file.format(epoch=epoch))
        plt.close()


        ## HISTOGRAM PLOT FOR EIGENVECTORS

        test_eigvecs_norms = np.linalg.norm(test_eigvecs, axis=1)
        pred_eigvecs_norms = np.linalg.norm(pred_eigvecs, axis=1)

        dot_products = np.sum(test_eigvecs * pred_eigvecs, axis=1)

        cos_sims = dot_products / (test_eigvecs_norms * pred_eigvecs_norms)

        abs_cos_sims = np.abs(cos_sims)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Histogram Plot for Eigenvectors')

        for col in range(3):

            x = abs_cos_sims[:, col]

            axs[col].hist(x, bins=100)
            axs[col].set_xlabel('Cosine similarity')
            axs[col].set_ylabel('Count')
            axs[col].set_title(f'Eigenvector {col+1}')

        plt.tight_layout()
        plt.savefig(evaluation_paths.eigenvectors_histogram_plot_file.format(epoch=epoch))
        plt.close()


        ## BLAND-ALTMAN PLOT FOR EIGENVALUES

        fig, axs = plt.subplots(1, 3, figsize=(21, 5))
        fig.suptitle('Bland-Altman Plot for Eigenvalues')

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
            axs[col].set_ylim(-1, 1)

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
        
        logging.info('')

        plt.tight_layout()
        plt.savefig(evaluation_paths.eigenvalues_bland_altman_plot_file.format(epoch=epoch))
        plt.close()


        ## BLAND-ALTMAN PLOT FOR MEAN DIFFUSIVITY

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle('Bland-Altman Plot for Mean Diffusivity')

        mean = (test_md + pred_md) / 2
        diff = pred_md - test_md

        x = mean
        y = diff

        # compute the point densities
        hist, xedges, yedges = np.histogram2d(x, y, bins=100)
        xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
        yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
        c = hist[xidx, yidx]

        # scatter plot with points colored by density
        scatter = ax.scatter(
            x, y, 
            c=c, 
            s=1, 
            norm=LogNorm(),  # Emphasize differences in density
            cmap='viridis'  # Use the 'viridis' colormap
        )

        # add a colorbar for the scatter plot
        fig.colorbar(scatter, ax=ax, label='Density')

        diff_mean = diff.mean()
        diff_std = diff.std()
        upper_limit = diff_mean + 1.96 * diff_std
        lower_limit = diff_mean - 1.96 * diff_std

        ax.axhline(y=diff_mean, color='k', linestyle='--')
        ax.axhline(y=upper_limit, color='r', linestyle='--')
        ax.axhline(y=lower_limit, color='r', linestyle='--')

        ax.set_xlabel('mean')
        ax.set_ylabel('diff (pred - true)')
        ax.set_ylim(-1, 1)

        # Create a new axes for the histogram, on the right of the current axes
        divider = make_axes_locatable(ax)
        ax_hist = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax)
        
        # Plot the histogram
        ax_hist.hist(diff, bins=200, orientation='horizontal', color='gray', alpha=0.5)

        # Hide ticks from both axes
        ax_hist.yaxis.set_visible(False)
        ax_hist.xaxis.set_visible(False)

        logging.info(f'Mean Diffusivity: ' + \
                    f'mean diff (pred - true): {diff_mean} ' + \
                    f'std: {diff_std} ' + \
                    f'upper limit: {upper_limit} ' + \
                    f'lower limit: {lower_limit}')
        logging.info('')
        
        plt.tight_layout()
        plt.savefig(evaluation_paths.mean_diffusivity_bland_altman_plot_file.format(epoch=epoch))
        plt.close()


        ## BLAND-ALTMAN PLOT FOR FRACTIONAL ANISOTROPY

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle('Bland-Altman Plot for Fractional Anisotropy')

        mean = (test_fa + pred_fa) / 2
        diff = pred_fa - test_fa

        x = mean
        y = diff

        # compute the point densities
        hist, xedges, yedges = np.histogram2d(x, y, bins=100)
        xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
        yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
        c = hist[xidx, yidx]
        
        # scatter plot with points colored by density
        scatter = ax.scatter(
            x, y, 
            c=c, 
            s=1, 
            norm=LogNorm(),  # Emphasize differences in density
            cmap='viridis'  # Use the 'viridis' colormap
        )

        # add a colorbar for the scatter plot
        fig.colorbar(scatter, ax=ax, label='Density')

        diff_mean = diff.mean()
        diff_std = diff.std()
        upper_limit = diff_mean + 1.96 * diff_std
        lower_limit = diff_mean - 1.96 * diff_std

        ax.axhline(y=diff_mean, color='k', linestyle='--')
        ax.axhline(y=upper_limit, color='r', linestyle='--')
        ax.axhline(y=lower_limit, color='r', linestyle='--')
        
        ax.set_xlabel('mean')
        ax.set_ylabel('diff (pred - true)')
        ax.set_ylim(-1, 1)

        # Create a new axes for the histogram, on the right of the current axes
        divider = make_axes_locatable(ax)
        ax_hist = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax)

        # Plot the histogram
        ax_hist.hist(diff, bins=200, orientation='horizontal', color='gray', alpha=0.5)

        # Hide ticks from both axes
        ax_hist.yaxis.set_visible(False)
        ax_hist.xaxis.set_visible(False)

        logging.info(f'Fractional Anisotropy: ' + \
                    f'mean diff (pred - true): {diff_mean} ' + \
                    f'std: {diff_std} ' + \
                    f'upper limit: {upper_limit} ' + \
                    f'lower limit: {lower_limit}')
        logging.info('')
        
        plt.tight_layout()
        plt.savefig(evaluation_paths.fractional_anisotropy_bland_altman_plot_file.format(epoch=epoch))
        plt.close()

        logging.info('-' * 50)
        logging.info('')


if __name__ == '__main__':
    main()
