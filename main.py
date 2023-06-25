# Load functions from pipeline
from pipeline.load_data import load_save_epochs, load_epochs, load_events
from pipeline.parameter_selection import bad_channels, load_save_ica, load_ica
from pipeline.feature_selection import preprocess_epochs, extract_features
from pipeline.utilities import get_participants
from pipeline.visualisation import plot_scatter_lin_reg, plot_lin_class, plot_time_lin_reg
from pipeline.epoch_outliers import epoch_outlier_indices, reaction_times
from pipeline.models import assign_percentiles

# Load external libraries
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import mne
import numpy as np
import json

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

mne.set_log_level('WARNING')

# Load hyperparameters for each individual participant
with open('participants_config') as json_file:
    info = json.load(json_file)

trained_models = {}

participants = get_participants()

# Set variables
non_eeg_channels = ['XForce', 'EOGvu', 'x_EMGl', 'x_GSR', 'x_Respi', 'x_Pulse', 'x_Optic']
event_splits = ({'Go cue 1': 210, 'Go cue 2': 211}, {'first activation': 150})
tmin, tmax = (-2.5, -2), (0.5, 0)
l_freq, h_freq = (1, 0.3), (48, 3)
decim = 5
low_pct, high_pct = 50, 50

# Functions that do not always have to be run
# Creating the epoch data, train ICA model, and plotting the data for visual inspection.
create_data, create_ica, check_data = False, False, False
if create_data:
    load_save_epochs(tmin=tmin, tmax=tmax, l_freq=l_freq, h_freq=h_freq, 
                     decim=decim, event_splits=event_splits, non_eeg_channels=non_eeg_channels)
if create_ica:
    load_save_ica('train', 'ps_gocue', info, h_freq[0], par_list=[7], max_iter=2000)
if check_data:
    for par in [participants[3]]:
        epochs_ps = load_epochs(f'saved-data/{par}/train', 'ps_gocue')
        epochs_ps.pick('eeg')

        print(f'{par}: {bad_channels(epochs_ps)}')
        epochs_ps.drop_channels(info[par]['bad_chn'])
        epochs_ps.plot(scalings='auto')
        input("Press Enter To Contingue:") 

# The variable gocue determens if the analyisis is run for the go cue marker as end point
# or that the first action is taken as the end point of the epochs.
gocue = False 
if gocue:
    file_name = "gocue"
else:
    file_name = "action"

# Run analysis for train data
for par in participants:
    data_path = f'saved-data/{par}'

    events = load_events(f'{data_path}/train')
    epochs_ps = load_epochs(f'{data_path}/train', f'ps_{file_name}')
    epochs_fs = load_epochs(f'{data_path}/train', f'fs_{file_name}')

    epochs_fs_eeg = epochs_fs.copy().pick('eeg')
    epochs_fs_eeg.drop_channels(info[par]['bad_chn'])

    epochs_ps_eeg = epochs_ps.copy().pick('eeg')
    epochs_ps_eeg.drop_channels(info[par]['bad_chn'])

    RTs, bad_rts = reaction_times(events, gocue)

    ica = load_ica(data_path)

    # Select the unwanted ICA componets
    ica.exclude = info[par]['exclude']

    # Apply the ICA filters on to the data used to train the model. 
    ica_epoches = ica.apply(epochs_ps_eeg.copy(), exclude=ica.exclude, n_pca_components=ica.n_components_)

    # Find the outlier epochs
    outliers = epoch_outlier_indices(ica_epoches.get_data(), events, par, plot=True, rt_thresh=info[par]['rt_thresh'],
                                     ptp_thresh=info[par]['ptp_thresh'], nstd=info[par]['ptp_nstd'], gocue=gocue)

    if outliers.size < ica_epoches.get_data().shape[0] * .5:
         # Invervals we want to avrage to create the features
        if gocue:
            clf_ival_boundaries = np.array([-2, -1.85, -1.25, -1.1, -0.5, -0.25])
        else: 
            clf_ival_boundaries = info[par]['clf_ival_boundaries']

        # Apply the special and temperal filters to the raw data. 
        # Then create the epochs and remove the outliers from those aswell as the reaction times.
        epochs, RTs = preprocess_epochs(epochs_fs_eeg, ica, events, outliers, par, clf_ival_boundaries, gocue=gocue)

        # Create features
        X, y = extract_features(epochs.pick(info[par]['ch_picks']), RTs, clf_ival_boundaries, per_two=True)

        lin_reg = Lasso(alpha=0.000009)
        lin_class = LDA(solver='lsqr', shrinkage='auto')

        y_split, indices = assign_percentiles(y, low_pct=low_pct, high_pct=high_pct)
        X_split = np.delete(X, indices, axis=0)

        plot_scatter_lin_reg(X, y, par, lin_reg, test_size=0.2, gocue=gocue)
        plot_time_lin_reg(X, y, par, lin_reg, test_size=0.2, gocue=gocue)
        plot_lin_class(X_split, y_split, par, lin_class, gocue=gocue)

        trained_models[par] = {
            'lin_reg': lin_reg.fit(X, y), 
            'lin_class': lin_class.fit(X_split, y_split)
        }

# Run parameter from above analysis on test data
for par in participants:
    if par in trained_models: 
        data_path = f'saved-data/{par}'

        events = load_events(f'{data_path}/test')
        epochs_fs = load_epochs(f'{data_path}/test', f'fs_{file_name}')

        # Select only eeg channels and the the bads
        epochs_fs_eeg = epochs_fs.copy().pick('eeg')
        epochs_fs_eeg.drop_channels(info[par]['bad_chn'])

        # Load the ica object related to this participant
        ica = load_ica(data_path)

        # Invervals we want to avrage to create the features
        if gocue:
            clf_ival_boundaries = np.array([-2, -1.85, -1.25, -1.1, -0.5, -0.25])
        else: 
            clf_ival_boundaries = info[par]['clf_ival_boundaries']

        # Apply the special and temperal filters to the raw data. 
        # Then create the epochs and remove the outliers from those aswell as the reaction times.
        epochs, RTs = preprocess_epochs(epochs_fs_eeg, ica, events, [], par, clf_ival_boundaries, gocue=gocue, train=False)

        # Create features
        X, y = extract_features(epochs.pick(info[par]['ch_picks']), RTs, clf_ival_boundaries, per_two=True)

        # Load the already traind models
        lin_reg = trained_models[par]['lin_reg']
        lin_class = trained_models[par]['lin_class']
        
        y_split, indices = assign_percentiles(y, low_pct=low_pct, high_pct=high_pct)
        X_split = np.delete(X, indices, axis=0)

        # Make some plots using the models
        plot_scatter_lin_reg(X, y, par, lin_reg, train=False, gocue=gocue)
        plot_time_lin_reg(X, y, par, lin_reg, train=False, gocue=gocue)
        plot_lin_class(X_split, y_split, par, lin_class, train=False, gocue=gocue)

