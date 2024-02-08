# GDL-DTI

## visualization.ipynb

- load the raw mri data (not necessary for this notebook)
- load the b-values and process them
- load the direction vectors and plot 3d sphere of direction vectors
- multiply b-values with direction vectors to get spherical shells and plot them
- isolate data of a single voxel (or load it if the file exists)
- use the signal measurements to vary the size and color of the points of the spherical shells
    - the shells can be plotted together, separately, and then separate but synchronized when rotated

## ground_truth.ipynb

- load the raw mri data and store it in a numpy file
    - or load the numpy file if it already exists
- load the direction vectors
- load the b-values and process them
    - keep the indices of each unique b-value
- use median_otsu function to mask the voxels that belong to the brain
- analyze the distribution of signal values per b-value in all the voxels
    - plot histograms with common x-axis, showing the minimum and maximum signal value per b-value
- use linear least squares to fit d-tensors on the masked data and keep valid tensors
    - meanwhile measure the mean square reconstruction error and standard deviation
    - store the results
- use non linear least squares to fit d-tensors on the masked data and keep valid tensors
    - meanwhile measure the mean square reconstruction error and standard deviation
    - store the results
- compare the results of the two approaches

## representations.ipynb

- define a function that craetes random diffusion tensors
- define a function for spectral decomposition and spectral decomposition
    - run two tests and measure accuracy of going from decomposition to the original d-tensor
- define a function for quaternion decomposition and quaternion decomposition
    - run two tests and measure accuracy of going from decomposition to the original d-tensor

## simulation_data.ipynb

- load the b-values and process them
- load the direction vectors
- load the results of linear or non-linear least squares created from `ground_truth.ipynb`
- choose b-values of 0.0 and 1.0
    - reconstruct the signal using the simple exponential decay formula
    - add noise with zero mean and 1/SNR standard deviation
    - renormalize the noisy signal
- store the d-tensors, clean signals and noisy signals

## train with sim data for d-tensor loss.ipynb

- load the d-tensors, clean signals and noisy signals created by `simulation_data.ipynb`
- choose a threshold for the maximum eigenvalue based on the percentile it covers
- shuffle, split and put the data in datasets and dataloaders for training and validation
- define neural network that works with spectral composition
- define the training and validation functions that return the epoch loss
    - wrapped to return the execution time as well
- train the model for an arbitrary number of epochs
    - it is possible to continue training after last epoch or restart
- plot the training and validation loss curves

