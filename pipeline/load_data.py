from pipeline.utilities import create_folders, make_dir, get_participants

import mne
import os
import warnings

def get_file_paths(file_range, participant):
    """
    Generates a list of file paths for a given range of EEG data files.

    Parameters
    ----------
    file_range : range
        The range of file numbers to load, e.g., range(1, 11) to load files '001.eeg' to '010.eeg'.

    Returns
    -------
    eeg_filepaths : list of str
        The file paths for the specified range of EEG data files.

    Raises
    ------
    AssertionError
        If `file_range` is not of the `range` type or if the range is not within the expected bounds.
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

    Parameters
    ----------
    file_range : tuple of ints
        The range of file numbers to load, e.g., range(1, 11) to load files '001.eeg' to '010.eeg'.
    misc : list of str | None
        The names of channels to load as misc, or none eeg channels. Defaults to None.

    Returns
    -------
    raw : instance of mne.io.Raw
        The concatenated raw EEG data.

    Raises
    ------
    IOError
        If a file in the range specified by `file_range` cannot be found or loaded.
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
    
    # Print a message indicating how many files were successfully loaded.
    # print(f'Successfully loaded {len(raw_arr)} EEG files from {len(raw_per_part)}.')
    
    # Concatenate the raw data from all files into a single object and return it.
    return mne.concatenate_raws(raw_arr)


def save_events(path, events):
    mne.write_events(f"{path}/_eve.fif", events, overwrite=True)

def save_epochs(path, name, epochs):
    epochs.save(f"{path}/{name}_epo.fif", overwrite=True)

def load_events(path):
    return mne.read_events(f"{path}/_eve.fif")

def load_epochs(path, name):
    return mne.read_epochs(f"{path}/{name}_epo.fif", preload=True)

def filter_resample_epoch(raw_data, tmin, tmax, l_freq, h_freq, decim, events, event_splits):
    """
    Applies filtering, resampling and epoching to raw MNE data.

    Parameters
    ----------
    raw_data : mne.io.Raw
        The raw MNE data to filter, resample, and epoch.
    tmin : float
        The start time of the epoch in seconds.
    tmax : float
        The end time of the epoch in seconds.
    l_freq : float
        The lower frequency bound for the filter.
    h_freq : float
        The upper frequency bound for the filter.
    decim : int
        The amount of downsampling to perform.
    events : array-like of shape (n_events, 3)
        The events array containing the event onsets.
    split_events : dict
        Dictionary containing information on how to split the events.

    Returns
    -------
    mne.Epochs
        The filtered, resampled, and epoched copy of the MNE data.

     Notes:
    ------
    - The function first filters the raw data using the specified frequency cutoffs.
    - It then down-samples the data using the specified decimation factor.
    - Finally, it epochs the data based on the specified event timings and returns the epoched data.
    """
    # Ignore warnings raised by MNE functions
    warnings.simplefilter('ignore')
    
    # Create a filtered copy of the raw data
    raw_copy = raw_data.copy().filter(l_freq,h_freq)
    
    # Epoch the filtered data and resample it with a factor of `decim`
    epochs = mne.Epochs(
        raw_copy, events, event_splits, tmin=tmin, tmax=tmax, 
        baseline=None, decim=decim, preload=True, reject=None)
    
    del raw_copy

    # Print success message and return the epochs object
    print('Successfully created a filtered, resampled, and epoched copy of the data.')
    return epochs


def _load_save_epochs(path, file_name, files, participant, tmin, tmax, l_freq, h_freq, event_splits, decim, misc):
    raws = load_raw_from_files(file_range=files, participant=participant, misc=misc)
    events = mne.events_from_annotations(raws)[0]

    epochs_os = filter_resample_epoch(
        raw_data=raws, tmin=tmin[0], tmax=tmax[0], l_freq=l_freq[0], h_freq=h_freq[0], 
        decim=decim, events=events, event_splits=event_splits)

    epochs_fe = filter_resample_epoch(
        raw_data=raws, tmin=tmin[1], tmax=tmax[1], l_freq=l_freq[1], h_freq=h_freq[1], 
        decim=decim, events=events, event_splits=event_splits)
    
    del raws

    store_data(path, file_name, events, epochs_os, epochs_fe)


def load_save_epochs(tmin, tmax, l_freq, h_freq, decim, event_splits, non_eeg_channels=None):
    # Ignore runtime warnings.
    warnings.simplefilter('ignore', RuntimeWarning)

    participants = get_participants()
    paths = create_folders(participants, 'saved-data')
    
    train_range = range(1, 18)
    test_range = range(18, 21)

    for i, par in enumerate(participants):
        # Load train data
        raws_train = load_raw_from_files(file_range=train_range, participant=par, misc=non_eeg_channels)
        events_train = mne.events_from_annotations(raws_train)[0]
        action_event_exisits = 150 in events_train[:, 2]

        # Frequency filter for perameter selection
        raw_train_ps = raws_train.copy().filter(l_freq[0],h_freq[0])

        epochs_train_ps_gocue = mne.Epochs(raw_train_ps, events_train, event_splits[0], tmin=tmin[0], tmax=tmax[0], 
                                           baseline=None, decim=decim, preload=True, reject=None)
        if action_event_exisits:
            epochs_train_ps_action = mne.Epochs(raw_train_ps, events_train, event_splits[1], tmin=tmin[0], tmax=tmax[0], 
                                           baseline=None, decim=decim, preload=True, reject=None)

        # Frequency filter for feather selection
        raw_train_fs = raws_train.copy().filter(l_freq[1],h_freq[1])

        epochs_train_fs_gocue = mne.Epochs(raw_train_fs, events_train, event_splits[0], tmin=tmin[1], tmax=tmax[1], 
                                           baseline=None, decim=decim, preload=True, reject=None)
        if action_event_exisits:
            epochs_train_fs_action = mne.Epochs(raw_train_fs, events_train, event_splits[1], tmin=tmin[1], tmax=tmax[1], 
                                           baseline=None, decim=decim, preload=True, reject=None)

        # Save train epochs and events to disk
        path_train = f'{paths[i]}/train'
        make_dir(path_train) 

        save_events(path_train, events_train)
        save_epochs(path_train, 'ps_gocue', epochs_train_ps_gocue)
        save_epochs(path_train, 'fs_gocue', epochs_train_fs_gocue)
        if action_event_exisits:
            save_epochs(path_train, 'ps_action', epochs_train_ps_action)
            save_epochs(path_train, 'fs_action', epochs_train_fs_action)

        # Delete raw files to consurve memory
        del raws_train, raw_train_ps, raw_train_fs

        # Load test data
        raws_test = load_raw_from_files(file_range=test_range, participant=par, misc=non_eeg_channels)
        events_test = mne.events_from_annotations(raws_test)[0]
        action_event_exisits = 150 in events_test[:, 2]

        raw_test_ps = raws_test.copy().filter(l_freq[0],h_freq[0])

        epochs_test_ps_gocue = mne.Epochs(raw_test_ps, events_test, event_splits[0], tmin=tmin[0], tmax=tmax[0], 
                                           baseline=None, decim=decim, preload=True, reject=None)
        if action_event_exisits:
            epochs_test_ps_action = mne.Epochs(raw_test_ps, events_test, event_splits[1], tmin=tmin[0], tmax=tmax[0], 
                                           baseline=None, decim=decim, preload=True, reject=None)

        raw_test_fs = raws_test.copy().filter(l_freq[1],h_freq[1])

        epochs_test_fs_gocue = mne.Epochs(raw_test_fs, events_test, event_splits[0], tmin=tmin[1], tmax=tmax[1], 
                                           baseline=None, decim=decim, preload=True, reject=None)
        if action_event_exisits:
            epochs_test_fs_action = mne.Epochs(raw_test_fs, events_test, event_splits[1], tmin=tmin[1], tmax=tmax[1], 
                                           baseline=None, decim=decim, preload=True, reject=None)
        
        # Save test epochs and events to disk
        path_test = f'{paths[i]}/test'
        make_dir(path_test) 

        save_events(path_test, events_test)
        save_epochs(path_test, 'ps_gocue', epochs_test_ps_gocue)
        save_epochs(path_test, 'fs_gocue', epochs_test_fs_gocue)
        if action_event_exisits:
            save_epochs(path_test, 'ps_action', epochs_test_ps_action)
            save_epochs(path_test, 'fs_action', epochs_test_fs_action)

        # Delete raw files to consurve memory
        del raws_test, raw_test_ps, raw_test_fs

        print(f"Saved event and epoch files to {paths[i]}")     


