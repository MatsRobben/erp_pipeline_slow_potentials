from pipeline.load_data import load_epochs
from pipeline.utilities import get_participants, make_dir, clear_figs

import numpy as np
import mne


def bad_channels(epochs):
    ch_names = epochs.info['ch_names']
    data = np.hstack(epochs.get_data())

    SD_channels = np.std(data, axis=1)
    
    median_outliers = np.abs(SD_channels - np.median(SD_channels)) > np.percentile(SD_channels, 75)
    low_thresh_outliers = SD_channels < 0.0001e-6
    high_thresh_outliers = SD_channels > 100e-6

    bad_chanels_bool = np.logical_or(median_outliers, 
                    np.logical_or(low_thresh_outliers, high_thresh_outliers))
    return np.array(ch_names)[bad_chanels_bool].tolist()

def apply_window(array, num_samples):
    epochs, channels, samples = array.shape

    # Create the window array
    window = np.ones((1, 1, samples))
    window[:, :, :num_samples] = np.linspace(0, 1, num_samples)
    window[:, :, -num_samples:] = np.linspace(1, 0, num_samples)
    
    # Apply the window to the input array using broadcasting
    squeezed_array = array * window

    return squeezed_array

def squeeze_epochs(epochs, num_samples):
    # Make a copy of the data from the epochs object
    epochs_copy = epochs.copy()

    # Apply the window function to the data
    epochs_copy._data = apply_window(epochs_copy._data, num_samples)

    return epochs_copy

def save_ica(path, ica):
    make_dir(path)
    ica.save(f"{path}/_ica.fif", overwrite=True)

def load_ica(path):
    return mne.preprocessing.read_ica(f"{path}/_ica.fif")

def train_ica(data, n_ica_components, max_iter):
    """
    Trains an Independent Component Analysis (ICA) model on the given MNE Raw or MNE Epochs object.

    Parameters
    ----------
    data : MNE Raw object or MNE Epochs object
        The data to apply the ICA on.
    n_ica_components : int
        The number of components to be extracted by the ICA.
    plot : bool, optional
        Whether or not to plot the results of the ICA (default is False).
    fmax : float, optional
        The highest frequency to be plotted.

    Returns
    -------
    ica : MNE ICA object
        The trained ICA model.

    """
    # Initialize an ICA object with specified parameters.
    ica = mne.preprocessing.ICA(n_components=n_ica_components, random_state=0, method='fastica', max_iter=max_iter)
    
    # Fit the ICA object to the input data.
    ica.fit(data)
    
    # Print a message indicating the completion of the ICA training.
    print('Successfully trained the ICA model')

    # Return the trained ICA model.
    return ica


def load_save_ica(data_dir, data_name, info, fmax, max_iter='auto', par_list = []):
    participants = get_participants()

    if par_list:
        participants = list(map(participants.__getitem__, par_list))
        print(participants)
    
    for par in participants:
        target_path = f"saved-data/{par}"

        epochs = load_epochs(f'{target_path}/{data_dir}', data_name)
        epochs.pick('eeg')
        epochs.drop_channels(info[par]['bad_chn'])

        squeezed_epochs = squeeze_epochs(epochs, 50)

        ica = train_ica(data=squeezed_epochs, n_ica_components=squeezed_epochs.info['nchan'], max_iter=max_iter)

        save_ica(target_path, ica)

        print(f"Saved trainded ICA model for participant {par}")

        figs = ica.plot_properties(squeezed_epochs, show=False, picks=range(ica.n_components), psd_args={'fmax': 48})

        path = f'{target_path}/plots/ica'
        make_dir(path)

        for i, fig in enumerate(figs):
            fig.savefig(f'{path}/ica{i}.png')

        clear_figs()
        del ica