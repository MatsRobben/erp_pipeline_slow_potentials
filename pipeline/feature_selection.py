from pipeline.epoch_outliers import reaction_times
from pipeline.utilities import save_plot, get_participant_nr

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_avg_epochs(epochs, boundaries, par):
    scalp_times = []
    for b in range(0, len(boundaries), 2):
        scalp_times.append((boundaries[b+1]+boundaries[b]) / 2)

    t_evoked = epochs.average(picks=epochs.info['ch_names'])

    fig = t_evoked.plot_joint(title=f'Average channel activity and intervals used for features, participant {get_participant_nr(par)}', times=scalp_times, show=False)  

    axes = fig.axes

    for b in range(0, len(boundaries), 2):
        axes[0].axvspan(boundaries[b], boundaries[b+1], color='gray', alpha=0.6, lw=0, label='Feature intervals')

    return fig

def preprocess_epochs(epochs, ica, events, outliers, participant, boundaries, icaSpace=False, gocue=True, train=True):
    """
    Preprocesses the raw EEG data by applying Independent Component Analysis (ICA), frequency filtering, and epoching.
    The specified outliers are dropped from the epochs and reaction times arrays.

    Parameters
    ----------
    raw_data : mne.io.Raw
        The raw EEG data to be preprocessed.
    ica : mne.preprocessing.ICA
        The ICA object to apply to the data.
    events : numpy.ndarray
        A 2-dimensional numpy array containing event information, where each row represents an event and each 
        column represents a different attribute of the event (e.g., time, event type, etc.).
    event_id : dict
        A dictionary mapping event types to integer IDs.
    outliers : numpy.ndarray
        A 1-dimensional numpy array containing the indices of the outlier epochs.
    l_freq : float or None
        The lower frequency bound of the filter. If None, no filtering is applied.
    h_freq : float or None
        The upper frequency bound of the filter. If None, no filtering is applied.
    tmin : float
        The start time of the epochs relative to the event onset.
    tmax : float
        The end time of the epochs relative to the event onset.
    decim : int
        The decimation factor to use. If 1, no decimation is applied.

    Returns
    -------
    new_epochs : mne.Epochs
        The preprocessed epochs.
    RTs : numpy.ndarray
        The reaction times for the preprocessed epochs.

    """
    RTs, bad_rts = reaction_times(events, gocue)
    if not train and bad_rts:
        outliers = np.zeros(RTs.shape, dtype=bool)
        outliers[bad_rts] = True 
    
    # comp_include = np.delete(np.arange(ica.n_components), ica.exclude)
    new_epochs_ica_space = ica.get_sources(epochs.copy())

    # Apply ICA and frequency filter to the raw data
    new_epochs_sensor_space = ica.apply(epochs.copy(), exclude=ica.exclude, n_pca_components=ica.n_components_)

    # Drop the outlier epochs and their corresponding reaction times
    new_epochs_ica_space.drop(outliers)
    new_epochs_sensor_space.drop(outliers)
    
    RTs = np.delete(RTs, outliers)

    fig_sensor = plot_avg_epochs(new_epochs_sensor_space, boundaries, participant)

    folder = 'train' if train else 'test'
    marker = 'gocue' if gocue else 'activation'
    
    save_plot(fig_sensor, f'plots/{folder}', f'{marker}_avg_epochs', '.svg', par=participant)

    print("Successfully applied the spatial (ICA) filter, frequency filter, and epoching to the data")

    if icaSpace:
        return new_epochs_ica_space, RTs
    else:
        return new_epochs_sensor_space, RTs


def get_jumping_means(epo, boundaries, per_two=False):
    """
    Computes the jumping means of the epoched data.

    Parameters
    ----------
    epo : mne.Epochs object
        The epoched data.
    boundaries : list
        A list of time boundaries for computing the means. Must be in seconds.
    per_two : Bool
        If true the function only uses the interval between each pair of bounds.
        Example: bounds = [1, 3, 5, 6] => [mean(1,3), mean(5,6)] instead of 
        [mean(1,3), mean(3,5), mean(5,6)].

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_jumps)
        The computed jumping means for each channel in each epoch for each jump.
    
    Notes
    -------
    This function was taken from assignment 6 of the Brain-Computer Interfacing (SOW-BKI323) course
    """
    # Get the original shape of the data
    shape_orig = epo.get_data().shape
    
    if per_two:
        X = np.zeros((shape_orig[0], shape_orig[1], len(boundaries)//2))
        range_bound = range(0, len(boundaries)-1, 2)
    else:
        # Create an empty array to store the jumping means
        X = np.zeros((shape_orig[0], shape_orig[1], len(boundaries)-1))
        range_bound = range(len(boundaries)-1)

    # Compute the jumping means for each boundary
    for i in range_bound:
        # Find the indices corresponding to the time range
        idx = epo.time_as_index((boundaries[i], boundaries[i+1]))
        idx_range = list(range(idx[0], idx[1]))
        
        # Compute the mean of the data in the time range for each epoch and channel
        X[:,:,i//2] = epo.get_data()[:,:,idx_range].mean(axis=2)
        
    return X


def extract_features(epochs, RTs, boundaries, per_two=False):
    """Extracts features from epochs and reaction time data
    
    Args:
    epochs: mne Epochs object containing epoched data
    RTs: numpy array containing reaction times
    boundaries: list of boundaries to group sections of the epoch. 
    Smallest value can not be smaller then tmin and largest 
    can not be greater then tmax.
    
    Returns:
    X: numpy array containing feature data
    y: numpy array containing reaction time data
    
    Notes
    -------
    This function was taken from assignment 6 of the Brain-Computer Interfacing (SOW-BKI323) course
    """
    # Define number of channels and features
    n_channels = len(epochs.info["chs"])
    if per_two:
        n_features = (len(boundaries)//2) * n_channels
    else:
        n_features = (len(boundaries) - 1) * n_channels
    
    # Get jumping means and flatten into feature matrix
    X = get_jumping_means(epochs, boundaries, per_two=per_two).squeeze().reshape((-1, n_features))
    
    # Assign reaction time data to target variable
    y = RTs
    
    return X, y


def reduce_dimensionality(X, n_components):
    """
    Reduce the dimensionality of the feature space using PCA.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    n_components : int or None, optional
        Number of components to keep. If None, all components are kept.

    Returns
    -------
    X_reduced : array-like of shape (n_samples, n_components)
        The reduced data.

    Raises
    ------
    ValueError
        If both `n_components` and `explained_variance` are None, or if `explained_variance` is not between 0 and 1.

    """
    pca = PCA(n_components=n_components, svd_solver='full')
    X_reduced = pca.fit_transform(X)
    return X_reduced

