from pipeline.utilities import create_folders, make_dir, get_participants

import mne
import os
import warnings

def get_file_paths(file_range, participant):
    """
    Generates a list of file paths for a given range of EEG data files and a given participant.

    Args:
        file_range (range): The range of file numbers to load, e.g., range(1, 11) to load files '001.vhdr' to '010.vhdr'.
        participant (str): The name of the folder corresponding to the participant of interest.

    Returns:
        eeg_filepaths (list of str): The file paths for the specified range of EEG data files.

    Raises:
        AssertionError: If `file_range` is not of the `range` type or if the range is not within the expected bounds.
    """
    
    # Set the path to the local data directory and the session identifier
    sep = os.path.sep
    local_path = os.path.dirname(os.path.realpath(__file__))
    local_data_path = f"{local_path}\Data"
    
    # Create an empty list to store the file paths
    eeg_filepaths = []
    # Iterate over the input range and generate a file path for each file number
    for f in file_range:
        search_path = f'{local_data_path}{sep}{participant}{sep}svipt_mainStudy_run{f}_.vhdr'
        eeg_filepaths.append(search_path)
            
    # Return the list of file paths
    return eeg_filepaths


def load_raw_from_files(file_range, participant, misc=None):
    """
    Loads raw EEG data from multiple files, concatenates them, and returns a single MNE raw object.

    Args:
        file_range (tuple of ints): The range of file numbers to load, e.g., range(1, 11) to load files '001.vhdr' to '010.vhdr'.
        misc (list of str or None): The names of channels to load as misc, or none eeg channels. Defaults to None.
        participant (str): The name of the participant from whom the raw data is loaded.
    
    Returns:
        raw (mne.io.Raw): The concatenated raw EEG data.

    Raises:
        IOError: If a file in the range specified by `file_range` cannot be found or loaded.
    """
    # Ignore runtime warnings.
    warnings.simplefilter('ignore', RuntimeWarning)
    
    # Get the file paths for the given range of file numbers.
    eeg_filepaths = get_file_paths(file_range, participant)

    # Initialize an empty list to store the raw data from each file.
    raw_arr = []

    # Loop over each file in the range and load its raw data.
    for eeg_filepath in eeg_filepaths:
        try:
            # load the EEG data from the file
            raw = mne.io.read_raw_brainvision(eeg_filepath, misc=misc)
            
            # set the montage to the standard 10/20 system and load the data
            raw.set_montage('standard_1020').load_data()

            raw_arr.append(raw)
        except IOError:
            # If the file cannot be found or loaded, raise an error.
            raise IOError(f'File "{eeg_filepath}" cannot be found or loaded.')
    
    # Concatenate the raw data from all files into a single object and return it.
    return mne.concatenate_raws(raw_arr)


def save_events(path, events):
    """
    Save events to disk.

    Args:
        path (str): Path to save the events.
        events (numpy.ndarray): Event array.
    """
    mne.write_events(f"{path}/_eve.fif", events, overwrite=True)

def save_epochs(path, prefix, epochs):
    """
    Save epochs to disk.

    Args:
        path (str): Path to save the epochs.
        prefix (str): Prefix for the filename.
        epochs (mne.Epochs): Epochs data to save.
    """
    epochs.save(f"{path}/{prefix}_epo.fif", overwrite=True)

def load_events(path):
    """
    Load events from a file.

    Args:
        path (str): Path to the event file.

    Returns:
        numpy.ndarray: Event array.
    """
    return mne.read_events(f"{path}/_eve.fif")

def load_epochs(path, prefix):
    """
    Load epochs from a file.

    Args:
        path (str): Path to the epochs file.
        prefix (str): Prefix for the filename.

    Returns:
        mne.Epochs: Loaded epochs data.
    """
    return mne.read_epochs(f"{path}/{prefix}_epo.fif", preload=True)

def process_data(raw_data, events, tmin, tmax, l_freq, h_freq, decim, event_splits, action_event_exists, path, data_type):
    """
    Process raw data and save epochs to disk.

    Args:
        raw_data (mne.io.Raw): Raw data to process.
        events (numpy.ndarray): Event array.
        tmin (list): List of start times for epoching.
        tmax (list): List of end times for epoching.
        l_freq (list): List of lower frequency bounds for filtering.
        h_freq (list): List of upper frequency bounds for filtering.
        decim (int): Decimation factor for downsampling.
        event_splits (list): List of event split parameters.
        action_event_exists (bool): Flag indicating if action event exists.
        path (str): Path to save the processed data.
        data_type (str): Type of data (train or test).
    """
    # Frequency filter for parameter selection
    raw_ps = raw_data.copy().filter(l_freq[0], h_freq[0])
    epochs_ps_gocue = mne.Epochs(raw_ps, events, event_splits[0], tmin=tmin[0], tmax=tmax[0], 
                                 baseline=None, decim=decim, preload=True, reject=None)
    if action_event_exists:
        epochs_ps_action = mne.Epochs(raw_ps, events, event_splits[1], tmin=tmin[0], tmax=tmax[0], 
                                      baseline=None, decim=decim, preload=True, reject=None)

    # Frequency filter for feature selection
    raw_fs = raw_data.copy().filter(l_freq[1], h_freq[1])
    epochs_fs_gocue = mne.Epochs(raw_fs, events, event_splits[0], tmin=tmin[1], tmax=tmax[1], 
                                 baseline=None, decim=decim, preload=True, reject=None)
    if action_event_exists:
        epochs_fs_action = mne.Epochs(raw_fs, events, event_splits[1], tmin=tmin[1], tmax=tmax[1], 
                                      baseline=None, decim=decim, preload=True, reject=None)

    # Save epochs and events to disk
    path_data = f'{path}/{data_type}'
    make_dir(path_data) 

    save_events(path_data, events)
    save_epochs(path_data, 'ps_gocue', epochs_ps_gocue)
    save_epochs(path_data, 'fs_gocue', epochs_fs_gocue)
    if action_event_exists:
        save_epochs(path_data, 'ps_action', epochs_ps_action)
        save_epochs(path_data, 'fs_action', epochs_fs_action)

    # Delete raw files to conserve memory
    del raw_data, raw_ps, raw_fs

def load_save_epochs(tmin, tmax, l_freq, h_freq, decim, event_splits, non_eeg_channels=None):
    """
    Load and save epochs for each participant.

    Args:
        tmin (list): List of start times for epoching.
        tmax (list): List of end times for epoching.
        l_freq (list): List of lower frequency bounds for filtering.
        h_freq (list): List of upper frequency bounds for filtering.
        decim (int): Decimation factor for downsampling.
        event_splits (list): List of event split parameters.
        non_eeg_channels (list, optional): List of non-EEG channels to load. Defaults to None.
    """
    # Ignore runtime warnings.
    warnings.simplefilter('ignore', RuntimeWarning)

    participants = get_participants()
    paths = create_folders(participants, 'saved-data')
    
    train_range = range(1, 18)
    test_range = range(18, 21)

    for i, par in enumerate(participants):
        # Load train and test data
        raws_train = load_raw_from_files(file_range=train_range, participant=par, misc=non_eeg_channels)
        raws_test = load_raw_from_files(file_range=test_range, participant=par, misc=non_eeg_channels)

        # Process train data
        events_train = mne.events_from_annotations(raws_train)[0]
        action_event_exists = 150 in events_train[:, 2]
        process_data(raws_train, events_train, tmin, tmax, l_freq, h_freq, decim, event_splits, action_event_exists, paths[i], 'train')

        # Process test data
        events_test = mne.events_from_annotations(raws_test)[0]
        action_event_exists = 150 in events_test[:, 2]
        process_data(raws_test, events_test, tmin, tmax, l_freq, h_freq, decim, event_splits, action_event_exists, paths[i], 'test')

        print(f"Saved event and epoch files to {paths[i]}")
   


