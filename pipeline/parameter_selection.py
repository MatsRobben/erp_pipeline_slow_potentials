from pipeline.load_data import load_epochs
from pipeline.utilities import get_participants, make_dir, clear_figs

import numpy as np
import mne

def bad_channels(epochs):
    """
    Identify bad channels based on standard deviation.

    Args:
        epochs (mne.Epochs): MNE Epochs object.

    Returns:
        list: List of bad channel names.
    """
    ch_names = epochs.info['ch_names']
    data = np.hstack(epochs.get_data())

    # Calculate the standard deviation for each channel
    SD_channels = np.std(data, axis=1)
    
    # Determine outliers based on median and threshold values
    median_outliers = np.abs(SD_channels - np.median(SD_channels)) > np.percentile(SD_channels, 75)
    low_thresh_outliers = SD_channels < 0.0001e-6
    high_thresh_outliers = SD_channels > 100e-6

    # Identify bad channels based on outliers
    bad_channels_bool = np.logical_or(median_outliers, 
                    np.logical_or(low_thresh_outliers, high_thresh_outliers))
    return np.array(ch_names)[bad_channels_bool].tolist()

def apply_window(array, num_samples):
    """
    Apply a window function to the input array.

    Args:
        array (numpy.ndarray): Input array.
        num_samples (int): Number of samples for the window.

    Returns:
        numpy.ndarray: Array with the window applied.
    """
    epochs, channels, samples = array.shape

    # Create the window array
    window = np.ones((1, 1, samples))
    window[:, :, :num_samples] = np.linspace(0, 1, num_samples)
    window[:, :, -num_samples:] = np.linspace(1, 0, num_samples)
    
    # Apply the window to the input array using broadcasting
    squeezed_array = array * window

    return squeezed_array

def squeeze_epochs(epochs, num_samples):
    """
    Apply a window function to the data in the epochs object.

    Args:
        epochs (mne.Epochs): MNE Epochs object.
        num_samples (int): Number of samples for the window.

    Returns:
        mne.Epochs: Epochs object with the window applied.
    """
    # Make a copy of the data from the epochs object
    epochs_copy = epochs.copy()

    # Apply the window function to the data
    epochs_copy._data = apply_window(epochs_copy._data, num_samples)

    return epochs_copy

def save_ica(path, ica):
    """
    Save an ICA object to disk.

    Args:
        path (str): Path to the directory for saving the ICA object.
        ica (mne.preprocessing.ICA): ICA object to save.
    """
    make_dir(path)
    ica.save(f"{path}/_ica.fif", overwrite=True)

def load_ica(path):
    """
    Load an ICA object from disk.

    Args:
        path (str): Path to the directory containing the ICA object.

    Returns:
        mne.preprocessing.ICA: Loaded ICA object.
    """
    return mne.preprocessing.read_ica(f"{path}/_ica.fif")

def train_ica(data, n_ica_components, max_iter):
    """
    Train an Independent Component Analysis (ICA) model on the given MNE Raw or MNE Epochs object.

    Args:
        data (mne.Raw or mne.Epochs): MNE Raw or MNE Epochs object to apply the ICA on.
        n_ica_components (int): Number of components to be extracted by the ICA.
        max_iter (int or str): Maximum number of iterations for the ICA algorithm.

    Returns:
        mne.preprocessing.ICA: Trained ICA model.
    """
    # Initialize an ICA object with specified parameters.
    ica = mne.preprocessing.ICA(n_components=n_ica_components, random_state=0, method='fastica', max_iter=max_iter)
    
    # Fit the ICA object to the input data.
    ica.fit(data)
    
    # Print a message indicating the completion of the ICA training.
    print('Successfully trained the ICA model')

    # Return the trained ICA model.
    return ica


def load_save_ica(data_dir, data_name, info, fmax, max_iter='auto', par_list=[]):
    """
    Load, preprocess, train, and save ICA models for multiple participants.

    Args:
        data_dir (str): Directory name for the data.
        data_name (str): Name of the data file.
        info (dict): Dictionary containing participant information.
        fmax (float): Highest frequency to be plotted.
        max_iter (int or str, optional): Maximum number of iterations for the ICA algorithm (default is 'auto').
        par_list (list, optional): List of participant indices to process (default is an empty list).

    Note:
        The function assumes that the 'load_epochs' function is defined in a module called 'load_data',
        and the 'get_participants', 'make_dir', and 'clear_figs' functions are defined in a module called 'utilities'.
    """
    # Get the list of participants
    participants = get_participants()

    if par_list:
        # If a participant list is provided, update the participants list accordingly
        participants = list(map(participants.__getitem__, par_list))
        print(participants)

    # Process each participant
    for par in participants:
        # Set the target path for saving the ICA model
        target_path = f"saved-data/{par}"

        # Load the epochs data for the participant
        epochs = load_epochs(f'{target_path}/{data_dir}', data_name)
        epochs.pick('eeg')
        epochs.drop_channels(info[par]['bad_chn'])

        # Apply windowing to the epochs data
        squeezed_epochs = squeeze_epochs(epochs, 50)

        # Train the ICA model on the squeezed epochs data
        ica = train_ica(data=squeezed_epochs, n_ica_components=squeezed_epochs.info['nchan'], max_iter=max_iter)

        # Save the trained ICA model
        save_ica(target_path, ica)

        # Print a message indicating the saved ICA model for the participant
        print(f"Saved trained ICA model for participant {par}")

        # Plot and save the ICA properties
        figs = ica.plot_properties(squeezed_epochs, show=False, picks=range(ica.n_components), psd_args={'fmax': fmax})

        path = f'{target_path}/plots/ica'
        make_dir(path)

        for i, fig in enumerate(figs):
            fig.savefig(f'{path}/ica{i}.png')

        # Clear the figures and delete the ICA object to free up memory
        clear_figs()
        del ica

