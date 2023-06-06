import mne
import os
from utilities import create_folders

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
    local_data_path = f"{os.getcwd()}\Data"
    
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
    # warnings.simplefilter('ignore', RuntimeWarning)
    
    # Get the file paths for the given range of file numbers.
    eeg_filepaths = get_file_paths(file_range, [participant])[0]

    # Initialize an empty list to store the raw data from each file.
    raw_arr = []

    # Loop over each file in the range and load its raw data.
    for eeg_filepath in eeg_filepaths:
        try:
            # load the EEG data from the file
            raw = mne.io.read_raw_brainvision(eeg_filepath, misc=misc)
            
            # set the montage to the standard 10/20 system and load the data
            raw.set_montage('standard_1020').load_data()
            
            # only select EEG channels
            raw.pick_types(eeg=True)

            raw_arr.append(raw)
        except IOError:
            # If the file cannot be found or loaded, raise an error.
            raise IOError(f'File "{eeg_filepath}" cannot be found or loaded.')
    
    # Print a message indicating how many files were successfully loaded.
    # print(f'Successfully loaded {len(raw_arr)} EEG files from {len(raw_per_part)}.')
    
    # Concatenate the raw data from all files into a single object and return it.
    return mne.concatenate_raws(raw_arr)


def store_data(path, name, events, epochs_os, epochs_fe):
    mne.write_events(f"{path}/{name}_eve.fif", events, overwrite=True)
    epochs_os.save(f"{path}/{name}_os_epo.fif", overwrite=True)
    epochs_fe.save(f"{path}/{name}_fe_epo.fif", overwrite=True)


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
    # warnings.simplefilter('ignore')
    
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
    
    del raws_test
    store_data(path, file_name, events, epochs_os, epochs_fe)


def load_save_epochs(tmin, tmax, l_freq, h_freq, decim, event_splits, non_eeg_channels=None):
    local_data_path = f"{os.getcwd()}\Data"
    participants = os.listdir(local_data_path)

    paths = create_folders(participants, 'saved-data')
    
    train_range = range(1, 18)
    test_range = range(18, 21)

    for i, par in enumerate(participants):
        _load_save_epochs(paths=paths[i], file_name='/train_gocue', files=train_range, 
                          participant=par, tmin=tmin, tmax=tmax,l_freq=l_freq, h_freq=h_freq,
                          event_splits=event_splits, decim=decim, misc=non_eeg_channels)
        
        _load_save_epochs(paths=paths[i], file_name='/test_gocue', files=test_range, 
                          participant=par, tmin=tmin, tmax=tmax,l_freq=l_freq, h_freq=h_freq,
                          event_splits=event_splits, decim=decim, misc=non_eeg_channels)
        
        _load_save_epochs(paths=paths[i], file_name='/train_activation', files=train_range, 
                          participant=par, tmin=tmin, tmax=tmax,l_freq=l_freq, h_freq=h_freq,
                          event_splits=event_splits, decim=decim, misc=non_eeg_channels)
        
        _load_save_epochs(paths=paths[i], file_name='/test_activation', files=test_range, 
                          participant=par, tmin=tmin, tmax=tmax,l_freq=l_freq, h_freq=h_freq,
                          event_splits=event_splits, decim=decim, misc=non_eeg_channels)

        print(f"Saved event and epoch files to {paths[i]}")