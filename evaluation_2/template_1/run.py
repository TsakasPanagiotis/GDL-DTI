'''Load the test set D, D* and f-values.
Load the predicted D, D* and f-values on the test set
from a fc_network_analysis experiment.
Compute the eigenvalues and eigenvectors
of the ground truth and predicted diffusion tensors.
Sort them in descending order.
Create a parity plot comparing for D and D*
either eigenvalues or Mean Diffusivity or Fractional Anisotropy
or f-values, all with Pearson correlation p-value.
Create a histogram plot comparing the eigenvectors for D and D*.
Create a Bland-Altman plot comparing fro D and D*
either eigenvalues or Mean Diffusivity or Fractional Anisotropy
or f-values, all with mean difference and standard deviation.'''


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


class SimulationDataPaths(Protocol):
    test_f_values_template: str
    test_d_tensors_template: str
    test_d_star_tensors_template: str


class FCNetworkHyperparameters(Protocol):
    simulation_data_paths_pkl: str


class FCNetworkPaths(Protocol):
    hyperparameters_file: str


class FCNetworkAnalysisHyperparameters(Protocol):
    fc_network_paths_pkl: str
    snr: str


class FCNetworkAnalysisPaths(Protocol):
    hyperparameters_file: str
    pred_d_tensors_file: str
    pred_d_star_tensors_file: str
    pred_f_values_file: str


@dataclass
class EvaluationHyperparameters:
    fc_network_analysis_paths_pkl: str


class EvaluationPaths:
    def __init__(self) -> None:
        self.experiment_path = os.path.join('evaluation_2', 'template_1', 'experiments', 
                                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_file = os.path.join(self.experiment_path, 'log.txt')
        self.paths_file = os.path.join(self.experiment_path, 'paths.pkl')
        self.hyperparameters_file = os.path.join(self.experiment_path, 'hparams.pkl')
        
        self.eigenvalues_parity_plot_file = os.path.join(self.experiment_path, 'eigvals_parity_plot.png')
        self.mean_diffusivity_parity_plot_file = os.path.join(self.experiment_path, 'md_parity_plot.png')
        self.fractional_anisotropy_parity_plot_file = os.path.join(self.experiment_path, 'fa_parity_plot.png')
        self.eigenvectors_histogram_plot_file = os.path.join(self.experiment_path, 'eigvecs_histogram.png')
        self.eigenvalues_bland_altman_plot_file = os.path.join(self.experiment_path, 'eigvals_bland_altman_plot.png')
        self.mean_diffusivity_bland_altman_plot_file = os.path.join(self.experiment_path, 'md_bland_altman_plot.png')
        self.fractional_anisotropy_bland_altman_plot_file = os.path.join(self.experiment_path, 'fa_bland_altman_plot.png')

        self.star_eigenvalues_parity_plot_file = os.path.join(self.experiment_path, 'star_eigvals_parity_plot.png')
        self.star_mean_diffusivity_parity_plot_file = os.path.join(self.experiment_path, 'star_md_parity_plot.png')
        self.star_fractional_anisotropy_parity_plot_file = os.path.join(self.experiment_path, 'star_fa_parity_plot.png')
        self.star_eigenvectors_histogram_plot_file = os.path.join(self.experiment_path, 'star_eigvecs_histogram.png')
        self.star_eigenvalues_bland_altman_plot_file = os.path.join(self.experiment_path, 'star_eigvals_bland_altman_plot.png')
        self.star_mean_diffusivity_bland_altman_plot_file = os.path.join(self.experiment_path, 'star_md_bland_altman_plot.png')
        self.star_fractional_anisotropy_bland_altman_plot_file = os.path.join(self.experiment_path, 'star_fa_bland_altman_plot.png')

        self.f_values_parity_plot_file = os.path.join(self.experiment_path, 'f_values_parity_plot.png')
        self.f_values_bland_altman_plot_file = os.path.join(self.experiment_path, 'f_values_bland_altman_plot.png')


def save_eigenvalues_parity_plot(test_eigvals, pred_eigvals, title, save_path):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

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
        # max_val = max(np.percentile(x, 98), np.percentile(y, 98))
        max_val = x.max()
        min_val = x.min()
        axs[col].plot(
            [min_val, max_val], 
            [min_val, max_val], 
            'k--')
        axs[col].set_ylim(min_val, max_val)

        # labels
        axs[col].set_xlabel('Ground Truth')
        axs[col].set_ylabel('Predictions')
        axs[col].set_title(f'Eigenvalue {col+1}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_parity_plot(test_values, pred_values, title, save_path):

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.suptitle(title)

    x = test_values
    y = pred_values

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
    # max_val = max(np.percentile(x, 98), np.percentile(y, 98))
    max_val = x.max()
    min_val = x.min()
    ax.plot(
        [min_val, max_val], 
        [min_val, max_val], 
        'k--')
    ax.set_ylim(min_val, max_val)
    
    # labels
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predictions')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_eigenvectors_histogram_plot(test_eigvecs, pred_eigvecs, title, save_path):

    test_eigvecs_norms = np.linalg.norm(test_eigvecs, axis=1)
    pred_eigvecs_norms = np.linalg.norm(pred_eigvecs, axis=1)

    dot_products = np.sum(test_eigvecs * pred_eigvecs, axis=1)

    cos_sims = dot_products / (test_eigvecs_norms * pred_eigvecs_norms)

    abs_cos_sims = np.abs(cos_sims)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    for col in range(3):

        x = abs_cos_sims[:, col]

        axs[col].hist(x, bins=100)
        axs[col].set_xlabel('Cosine similarity')
        axs[col].set_ylabel('Count')
        axs[col].set_title(f'Eigenvector {col+1}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_eigenvalues_bland_altman_plot(test_eigvals, pred_eigvals, title, save_path):

    fig, axs = plt.subplots(1, 3, figsize=(21, 5))
    fig.suptitle(title)

    logging.info(title)

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
        # fig.colorbar(scatter, ax=axs[col], label='Density')
        
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
        axs[col].set_ylim(diff_mean - 3.0 * diff_std, diff_mean + 3.0 * diff_std)

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
    plt.savefig(save_path)
    plt.close()


def save_bland_altman_plot(test_values, pred_values, title, save_path):

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(title)

    logging.info(title)

    mean = (test_values + pred_values) / 2
    diff = pred_values - test_values

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
    ax.set_ylim(diff_mean - 3.0 * diff_std, diff_mean + 3.0 * diff_std)

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
    plt.savefig(save_path)
    plt.close()


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

    
    ## DATA

    test_d_tensors = np.load(sim_data_paths.test_d_tensors_template.format(int(fc_network_analysis_hparams.snr)))
    test_d_star_tensors = np.load(sim_data_paths.test_d_star_tensors_template.format(int(fc_network_analysis_hparams.snr)))
    test_f_values = np.load(sim_data_paths.test_f_values_template.format(int(fc_network_analysis_hparams.snr)))

    pred_d_tensors = np.load(fc_network_analysis_paths.pred_d_tensors_file)
    pred_d_star_tensors = np.load(fc_network_analysis_paths.pred_d_star_tensors_file)
    pred_f_values = np.load(fc_network_analysis_paths.pred_f_values_file).squeeze()


    ## EIGENVALUES AND EIGENVECTORS OF D

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


    ## EIGENVALUES AND EIGENVECTORS OF D*

    test_eigvals_star, test_eigvecs_star = np.linalg.eig(test_d_star_tensors)
    test_eigvals_star = test_eigvals_star.real
    test_eigvecs_star = test_eigvecs_star.real

    pred_eigvals_star, pred_eigvecs_star = np.linalg.eig(pred_d_star_tensors)
    pred_eigvals_star = pred_eigvals_star.real
    pred_eigvecs_star = pred_eigvecs_star.real

    # Get the indices that would sort the eigenvalues in descending order    
    test_sort_indices_star = np.argsort(test_eigvals_star, axis=1)[:, ::-1]    
    pred_sort_indices_star = np.argsort(pred_eigvals_star, axis=1)[:, ::-1]

    # Use these indices to sort the eigenvalues and eigenvectors
    
    test_eigvals_star = np.take_along_axis(test_eigvals_star, test_sort_indices_star, axis=1)
    test_eigvecs_star = np.take_along_axis(test_eigvecs_star, np.expand_dims(test_sort_indices_star, axis=1), axis=2)
    
    pred_eigvals_star = np.take_along_axis(pred_eigvals_star, pred_sort_indices_star, axis=1)
    pred_eigvecs_star = np.take_along_axis(pred_eigvecs_star, np.expand_dims(pred_sort_indices_star, axis=1), axis=2)


    ## ONLY KEEP EIGENVALUES LOWER THAN THE THRESHOLD

    # valid_indices = pred_eigvals[:, 0] < ground_truth_hparams.threshold_eigval

    # logging.info(f'Number of invalid diffusion tensors: {np.sum(~valid_indices)} / {pred_d_tensors.shape[0]}')
    # logging.info(f'Percentage of invalid diffusion tensors: {np.sum(~valid_indices) / pred_d_tensors.shape[0] * 100:.2f}%')
    # logging.info('')

    # test_eigvals = test_eigvals[valid_indices]
    # test_eigvecs = test_eigvecs[valid_indices]

    # pred_eigvals = pred_eigvals[valid_indices]
    # pred_eigvecs = pred_eigvecs[valid_indices]


    ## MEAN DIFFUSIVITY

    # MD = (lambda_1 + lambda_2 + lambda_3) / 3
    test_md = np.mean(test_eigvals, axis=1)
    pred_md = np.mean(pred_eigvals, axis=1)

    test_md_star = np.mean(test_eigvals_star, axis=1)
    pred_md_star = np.mean(pred_eigvals_star, axis=1)


    ## FRACTIONAL ANISOTROPY
    
    # FA = sqrt(1.5 * sum((lambda_i - MD)^2) / sum(lambda_i^2))
    test_fa = np.sqrt(1.5 * np.sum((test_eigvals - test_md[:, None])**2, axis=1) / np.sum(test_eigvals**2, axis=1))
    pred_fa = np.sqrt(1.5 * np.sum((pred_eigvals - pred_md[:, None])**2, axis=1) / np.sum(pred_eigvals**2, axis=1))

    test_fa_star = np.sqrt(1.5 * np.sum((test_eigvals_star - test_md_star[:, None])**2, axis=1) / np.sum(test_eigvals_star**2, axis=1))
    pred_fa_star = np.sqrt(1.5 * np.sum((pred_eigvals_star - pred_md_star[:, None])**2, axis=1) / np.sum(pred_eigvals_star**2, axis=1))


    ## PEARSON CORRELATIONS OF D TENSOR

    logging.info('D tensor:')
    logging.info('')

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

    
    ## PEARSON CORRELATIONS OF D* TENSOR

    logging.info('D* tensor:')
    logging.info('')

    # Eigenvalues
    for col in range(3):
        pearson_result = PearsonResult(scipy.stats.pearsonr(test_eigvals_star[:, col], pred_eigvals_star[:, col]))
        logging.info(f'Eigenvalue {col+1} : ' + \
                     f'Pearson correlation coefficient = {pearson_result.statistic} ' + \
                     f'p-value = {pearson_result.pvalue}')
    logging.info('')

    # Mean Diffusivity
    pearson_result = PearsonResult(scipy.stats.pearsonr(test_md_star, pred_md_star))
    logging.info(f'Mean Diffusivity : ' + \
                 f'Pearson correlation coefficient = {pearson_result.statistic} ' + \
                 f'p-value = {pearson_result.pvalue}')
    logging.info('')

    # Fractional Anisotropy
    pearson_result = PearsonResult(scipy.stats.pearsonr(test_fa_star, pred_fa_star))
    logging.info(f'Fractional Anisotropy : ' + \
                 f'Pearson correlation coefficient = {pearson_result.statistic} ' + \
                 f'p-value = {pearson_result.pvalue}')
    logging.info('')

    
    ## PEARSON CORRELATIONS OF F VALUES

    pearson_result = PearsonResult(scipy.stats.pearsonr(test_f_values, pred_f_values))
    logging.info(f'F values : ' + \
                 f'Pearson correlation coefficient = {pearson_result.statistic} ' + \
                 f'p-value = {pearson_result.pvalue}')
    logging.info('')

    
    ## PARITY PLOT FOR EIGENVALUES

    save_eigenvalues_parity_plot(test_eigvals, pred_eigvals, 
                                 'Parity Plot for Eigenvalues of D Tensor', 
                                 evaluation_paths.eigenvalues_parity_plot_file)
    
    save_eigenvalues_parity_plot(test_eigvals_star, pred_eigvals_star, 
                                 'Parity Plot for Eigenvalues of D* Tensor', 
                                 evaluation_paths.star_eigenvalues_parity_plot_file)

    
    ## PARITY PLOT FOR MEAN DIFFUSIVITY

    save_parity_plot(test_md, pred_md, 
                     'Parity Plot for Mean Diffusivity of D Tensor', 
                     evaluation_paths.mean_diffusivity_parity_plot_file)
    
    save_parity_plot(test_md_star, pred_md_star, 
                     'Parity Plot for Mean Diffusivity of D* Tensor', 
                     evaluation_paths.star_mean_diffusivity_parity_plot_file)


    ## PARITY PLOT FOR FRACTIONAL ANISOTROPY

    save_parity_plot(test_fa, pred_fa, 
                     'Parity Plot for Fractional Anisotropy of D Tensor', 
                     evaluation_paths.fractional_anisotropy_parity_plot_file)
    
    save_parity_plot(test_fa_star, pred_fa_star,
                     'Parity Plot for Fractional Anisotropy of D* Tensor', 
                     evaluation_paths.star_fractional_anisotropy_parity_plot_file)


    ## PARITY PLOT FOR F VALUES

    save_parity_plot(test_f_values, pred_f_values,
                     'Parity Plot for F Values', 
                     evaluation_paths.f_values_parity_plot_file)


    ## HISTOGRAM PLOT FOR EIGENVECTORS

    save_eigenvectors_histogram_plot(test_eigvecs, pred_eigvecs, 
                                     'Histogram Plot for Eigenvectors of D Tensor', 
                                     evaluation_paths.eigenvectors_histogram_plot_file)
    
    save_eigenvectors_histogram_plot(test_eigvecs_star, pred_eigvecs_star, 
                                     'Histogram Plot for Eigenvectors of D* Tensor', 
                                     evaluation_paths.star_eigenvectors_histogram_plot_file)


    ## BLAND-ALTMAN PLOT FOR EIGENVALUES

    save_eigenvalues_bland_altman_plot(test_eigvals, pred_eigvals, 
                                       'Bland-Altman Plot for Eigenvalues of D Tensor', 
                                       evaluation_paths.eigenvalues_bland_altman_plot_file)
    
    save_eigenvalues_bland_altman_plot(test_eigvals_star, pred_eigvals_star, 
                                       'Bland-Altman Plot for Eigenvalues of D* Tensor', 
                                       evaluation_paths.star_eigenvalues_bland_altman_plot_file)


    ## BLAND-ALTMAN PLOT FOR MEAN DIFFUSIVITY

    save_bland_altman_plot(test_md, pred_md, 
                           'Bland-Altman Plot for Mean Diffusivity of D Tensor', 
                           evaluation_paths.mean_diffusivity_bland_altman_plot_file)
    
    save_bland_altman_plot(test_md_star, pred_md_star, 
                           'Bland-Altman Plot for Mean Diffusivity of D* Tensor', 
                           evaluation_paths.star_mean_diffusivity_bland_altman_plot_file)


    ## BLAND-ALTMAN PLOT FOR FRACTIONAL ANISOTROPY

    save_bland_altman_plot(test_fa, pred_fa,
                           'Bland-Altman Plot for Fractional Anisotropy of D Tensor',
                           evaluation_paths.fractional_anisotropy_bland_altman_plot_file)
    
    save_bland_altman_plot(test_fa_star, pred_fa_star,
                           'Bland-Altman Plot for Fractional Anisotropy of D* Tensor',
                           evaluation_paths.star_fractional_anisotropy_bland_altman_plot_file)
    
    
    ## BLAND-ALTMAN PLOT FOR F VALUES

    save_bland_altman_plot(test_f_values, pred_f_values,
                           'Bland-Altman Plot for F Values',
                           evaluation_paths.f_values_bland_altman_plot_file)


if __name__ == '__main__':
    main()
