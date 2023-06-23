from pipeline.utilities import save_plot, get_participant_nr

import numpy as np
import matplotlib.pyplot as plt


def reaction_times(events, gocue=True):
    """
    Compute the reaction times for a set of trials.

    Parameters
    ----------
    events : ndarray, shape (n_events, 3)
        The events to compute the reaction times for. The array should have
        three columns: the sample number, the duration, and the event ID.

    Returns
    -------
    reaction_times : ndarray, shape (n_trials,)
        The reaction times, computed as the time difference between the
        first occurrence of events with ID 210 or 211 (the start of the trial)
        and the first occurrence of events with ID 150 (the first time pressure
        is detected on the pressure sensor).
    bad_trials : list
        A list of indices for trials that do not have event ID 150 following every 
        occurrence of event ID 210 or 211.

    Notes
    -----
    This function assumes that the events 210 and 211 are followed by an event with ID 150.

    """
    markers_go_cue = (events[:,2] == 210) | (events[:,2] == 211)
    markers_activation = (events[:,2] == 150)

    events_clean = events[(events[:,2] == 210) | (events[:,2] == 211) | (events[:,2] == 150)]

    go_cue_bads = []
    go_cue_idx = 0

    activation_bads = []
    activation_idx = 0

    for i in range(len(events_clean)-1):
        # A bad index has been found if the current event has id 210/211 and the next event has id 150.
        if (events_clean[i][2] == 210 or events_clean[i][2] == 211) and events_clean[i+1][2] != 150:
            go_cue_bads.append(go_cue_idx) 
        elif events_clean[i][2] == 150 and events_clean[i+1][2] == 150:
            activation_bads.append(activation_idx+1) 
            
        # Increase the go cue event counter if we have found an event with id 210 or 211
        if events_clean[i][2] == 210 or events_clean[i][2] == 211:
            go_cue_idx += 1
        elif events_clean[i][2] == 150:
            activation_idx += 1
        
    # If the last element is a go cue a bad trial has been found, because the 150 event is missing.
    if (events_clean[-1][2] == 210 or events_clean[-1][2] == 211):
        go_cue_bads.append(go_cue_idx)
    elif events_clean[0][2] == 150:
        activation_bads.append(0) 

    go_cue_times = np.delete(events[markers_go_cue][:,0], go_cue_bads)
    activation_times = np.delete(events[markers_activation][:,0], activation_bads)

    RTs = np.subtract(activation_times, go_cue_times)

    if gocue:
        for b in go_cue_bads:
            RTs = np.insert(RTs, b, 0)
        
        return RTs, np.array(go_cue_bads)
    else:
        for b in activation_bads:
            RTs = np.insert(RTs, b, 0)

        return RTs, np.array(activation_bads)

def RT_outliers(events, threshold, n_channels=1, gocue=True):
    """
    Identifies reaction time (RT) outliers based on a threshold value and returns a boolean array indicating which 
    trials contain outliers.
    
    Parameters
    ----------
    events : numpy.ndarray
        A 2-dimensional numpy array containing event information, where each row represents an event and each 
        column represents a different attribute of the event (e.g., time, event type, etc.).
    threshold : float
        A numeric value representing the threshold above which RTs will be considered outliers.
    n_channels : int, optional
        The number of channels in the data. 
    
    Returns
    -------
    numpy.ndarray
        A boolean array indicating which trials contain RT outliers. If `_2d` is True, the array is 2-dimensional 
        with the shape being (Epochs, channels).
    """
    # Compute reaction times for each trial
    RTs, bad_trials = reaction_times(events, gocue)

    # Identify outliers based on the specified threshold
    outliers = np.logical_or(RTs < threshold[0], RTs > threshold[1])
    
    if len(bad_trials)!=0:
        outliers[bad_trials] = True
    
    return np.tile(outliers, (n_channels, 1)).T if n_channels > 1 else outliers

def plot_reaction_times(par, reaction_times, bad_trials, min_threshold, max_threshold, gocue=True):
    fig = plt.figure(figsize=(10, 6))

    # Create an array of indices for good trials
    good_trials = np.arange(len(reaction_times))
    good_trials = np.delete(good_trials, bad_trials) if np.any(bad_trials) else good_trials

    # Create an array of reaction times for good trials
    good_reaction_times = np.array(reaction_times)
    good_reaction_times = np.delete(good_reaction_times, bad_trials) if np.any(bad_trials) else good_reaction_times

    # Plot reaction times for good trials
    plt.plot(good_trials, good_reaction_times, marker='o', linestyle='-', color='blue', label='Good Trials')
    
    # Plot bad trials at the bottom
    max_y = max(good_reaction_times) if len(good_reaction_times) > 0 else 0
    bad_trials_y = np.full(len(bad_trials), max_y * 0.9)
    plt.plot(bad_trials, bad_trials_y, marker='x', linestyle='None', markersize=3, color='red', label='Bad Trials')

    # Plot minimum and maximum thresholds
    plt.axhline(min_threshold, color='green', linestyle='--', label='Min Threshold')
    plt.axhline(max_threshold, color='orange', linestyle='--', label='Max Threshold')

    plt.xlabel('Trial')
    plt.ylabel('Reaction Time (ms)')
    plt.title(f'Reaction Times of participant {get_participant_nr(par)}')
    plt.grid(True)
    
    # Move the legend outside the plot
    plt.legend()

    if gocue:
        save_plot(fig, 'plots', 'gocue_reaction_times', '.svg', par=par)
    else:
        save_plot(fig, 'plots', 'activation_reaction_times', '.svg', par=par)

def ptp_outliers_thresh(data, thresh):
    """
    Identifies peak-to-peak (PTP) amplitude outliers based on a threshold value and returns a boolean array 
    indicating which epochs channel conbinations contain outliers.

    Parameters
    ----------
    data : numpy.ndarray
        A 3-dimensional numpy array containing the EEG data, where the shape is (epochs, channels, time_points)
    thresh : float
        A numeric value representing the threshold above which PTP amplitudes will be considered outliers.

    Returns
    -------
    numpy.ndarray
        A boolean array indicating which trials contain PTP amplitude outliers. The shape of the array is 
        (Epochs, Channels).
    """
    # Calculate the peak-to-peak amplitudes of the EEG data along the time axis
    ptp = np.ptp(data, axis=2)

    # Identify trials that have PTP amplitudes greater than the threshold value
    ptp_thresh = ptp > thresh

    return ptp_thresh

def ptp_outliers_stat(data, nstd):
    """
    Identifies epochs in the given data where the peak-to-peak amplitude values are above a certain number of
    standard deviations from the mean across channels.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_samples)
        The data to analyze.
    nstd : float
        The number of standard deviations from the mean above which an epoch is considered an outlier.

    Returns
    -------
    outliers : ndarray, shape (n_epochs, n_channels)
        A boolean array indicating which epochs and channels contain outliers, based on the peak-to-peak
        amplitude values being greater than `nstd` standard deviations from the mean.

    Notes
    -----
    Peak-to-peak (PTP) amplitude is calculated as the difference between the maximum and minimum values in each epoch.
    The mean and standard deviation are calculated across all epochs for each channel, and z-scores are computed
    for each epoch and channel based on these values. An epoch is considered an outlier if its PTP amplitude
    z-score is greater than the number of standard deviations specified `nstd`.
    """
    # Compute the PTP for each epoch
    ptp_epoch = np.ptp(data, axis=-1)

    # Calculate the mean and standard deviation for each channel across all epochs
    channel_means = np.mean(ptp_epoch, axis=0)
    channel_stds = np.std(ptp_epoch, axis=0)

    # Calculate the z-scores for each epoch and channel
    z_scores = (ptp_epoch - channel_means) / channel_stds

    # Return a boolean array indicating which epochs contain PTP outliers
    return z_scores > nstd

def plot_outlier(par, RT_outliers, ptp_outliers, ptp_stat_outliers, combined_outliers, gocue=True):
    """
    Plots different types of outliers as images using matplotlib.
    
    Parameters
    ----------
    RT_outliers : numpy.ndarray
        The 2-dimensional numpy array of RT outliers to be plotted.
    ptp_outliers : numpy.ndarray
        The 2-dimensional numpy array of ptp outliers to be plotted.
    ptp_stat_outliers : numpy.ndarray
        The 2-dimensional numpy array of ptp_stat outliers to be plotted.
    combined_outliers : numpy.ndarray
        The 2-dimensional numpy array of combined outliers to be plotted.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure object.
    
    """
    # Set the figure size and create a new figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Plot RT_outliers
    axes[0, 0].imshow(np.transpose(np.logical_not(RT_outliers)), cmap='Greys_r', interpolation='nearest')
    axes[0, 0].set_title('RT Outliers')
    axes[0, 0].set_ylabel('Channels')
    axes[0, 0].set_xlabel('Epochs')
    
    # Plot ptp_outliers
    axes[0, 1].imshow(np.transpose(np.logical_not(ptp_outliers)), cmap='Greys_r', interpolation='nearest')
    axes[0, 1].set_title('ptp Outliers')
    axes[0, 1].set_ylabel('Channels')
    axes[0, 1].set_xlabel('Epochs')
    
    # Plot ptp_stat_outliers
    axes[1, 0].imshow(np.transpose(np.logical_not(ptp_stat_outliers)), cmap='Greys_r', interpolation='nearest')
    axes[1, 0].set_title('ptp_stat Outliers')
    axes[1, 0].set_ylabel('Channels')
    axes[1, 0].set_xlabel('Epochs')
    
    # Plot combined_outliers
    axes[1, 1].imshow(np.transpose(np.logical_not(combined_outliers)), cmap='Greys_r', interpolation='nearest')
    axes[1, 1].set_title('Combined Outliers')
    axes[1, 1].set_ylabel('Channels')
    axes[1, 1].set_xlabel('Epochs')

    fig.suptitle(f"Epoch outliers for participant {get_participant_nr(par)}")
    
    # Adjust the spacing between subplots
    plt.tight_layout()

    if gocue:
        save_plot(fig, 'plots', 'gocue_outliers', '.svg', par=par)
    else:
        save_plot(fig, 'plots', 'activation_outliers', '.svg', par=par)

def epoch_outlier_indices(data, events, par, rt_thresh=(150, 1000), ptp_thresh=80e-6, nstd=3, plot=False, gocue=True):
    """
    Identifies outlier epochs based on reaction time (RT) and peak-to-peak (PTP) thresholds. Then removes those outliers 
    and apply statictical outlier removal. Finaly, it returns the indices of the outlier epochs 
    from both of the detection methods.
    
    Parameters
    ----------
    data : numpy.ndarray
        A 3-dimensional numpy array of shape (Epochs, Channels, Samples) containing the EEG data.
    events : numpy.ndarray
        A 2-dimensional numpy array containing event information, where each row represents an event and each 
        column represents a different attribute of the event (e.g., time, event type, etc.).
    rt_thresh : float, optional
        A numeric value representing the threshold above which RTs will be considered outliers. Default is 1000.
    ptp_thresh : float, optional
        A numeric value representing the threshold below which PTP values will be considered outliers. Default is 80e-6.
    nstd : float, optional
        A numeric value representing the number of standard deviations from the mean above which PTP values will be 
        considered outliers. Default is 3.
    
    Returns
    -------
    numpy.ndarray
        A 1-dimensional numpy array containing the indices of the outlier epochs in the original data array.
    """
    
    # Find the outliers for the RTs and peak to peak distance threshold.
    RT_outlier = RT_outliers(events, rt_thresh, n_channels=data.shape[1], gocue=gocue)
    ptp_outliers = ptp_outliers_thresh(data, ptp_thresh)

    # Combine the outliers from both groups of outliers.
    combine_outliers = np.logical_or(RT_outlier, ptp_outliers)
    
    # Create a boolean list (Epoch,) that indicates which epochs are outliers
    epoch_outliers = np.any(combine_outliers, axis=1)
    
    # Find the indices of the outliers
    index_outliers = np.where(epoch_outliers)[0]
    
    # Creates a list without the outlieres and a list of the original indices of the non-outliers
    new_index = np.delete(np.arange(ptp_outliers.shape[0]), index_outliers)
    data_del_outliers = np.delete(data, index_outliers, 0)
    
    # Finds the peak to peak outliers based on a given number of standard deviations
    ptp_stat_outliers = ptp_outliers_stat(data_del_outliers, nstd)
    
    # Create a boolean list that indicates which booleans are outliers
    epoch_outliers_stat = np.any(ptp_stat_outliers, axis=1)
    # Find the indices of the outliers
    index_outliers_stat = np.where(epoch_outliers_stat)[0]
    
    # Concatenate the outliers from the threshold functions and those from the statistical ones
    outlier_indices = np.concatenate((new_index[index_outliers_stat], index_outliers))
    outlier_indices.sort(kind='mergesort')

    complete_outliers = np.zeros(combine_outliers.shape)

    for i in range(len(combine_outliers)):
        if i in new_index:
            complete_outliers[i] = ptp_stat_outliers[np.where(new_index == i)[0][0]]
        else:
            complete_outliers[i] = combine_outliers[i]

    if plot:
        RTs, bad_trials = reaction_times(events, gocue=gocue)
        plot_reaction_times(par, RTs, bad_trials, rt_thresh[0], rt_thresh[1], gocue=gocue)
        plot_outlier(par, RT_outlier, ptp_outliers, ptp_stat_outliers, complete_outliers, gocue=gocue)
    
    print(f"Successfully detected the epoch outliers, found {len(outlier_indices)} outliers")
    return outlier_indices






