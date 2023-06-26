# erp_pipeline_slow_potentials

This repository contains the code used for analysis in the Artificial Intelligence Bachelor Thesis of Mats Robben. Additionally, the repository contains all the figures that were generated during the analysis, which can be found in the `appendix` folder. 

## Installation
This project was written in Python, using the MNE library. Therefore, it is necessary to install both before attempting to run the code. The project assumes the EEG data is stored in a folder named `Data`, where every participant has its own folder containing the EEG recordings. In addition, the assumption is made that the markers s150 (movement onset), s210 (go-cue condition 1), s211 (go-cue condition 2) are present in the marker files. Furthermore, the analysis requires the `participants_config` to be filled in properly, which stores all the hyperparameters for the participants. The template of this file can be found in the repository.

## Appendix
The appendix folder contains two sub-directories, one for the pre-trial figures and one for the pre-movement figures. Both sub-directories contain the same 9 files, which in turn contain the figures of the participants. Here follows a short description of the 9 types of plots:

- avg_signal_train: The averaged activation over the channels after pre-processing and signal cleaning, on the training data.
- avg_signal_holdout: The averaged activation over the channels after pre-processing and signal cleaning, on the holdout data.
- class_train: The results of the classification model on the training data. The figures contain the confusion matrix and ROC curve, both averaged over the folds of the cross-validation.
- class_holdout: The results of the classification model on the holdout data. The figures contain the confusion matrix and ROC curve. As Well as the calculated p-value.
- reg_train: The results of the regression model on the train data. The figures plot the real reaction times and the predicted ones. The data is split up into a train and validation set, which are both shown as sub-plots.
- reg_holdout: The results of the regression model on the holdout data. The figures plot the real reaction times and the predicted ones.
- reg_scatter_train: Scatter plot of the real reaction times against the predicted reaction times for the train data. These figures show the correlation between the real and predicted reaction times.
- reg_scatter_holdout: Scatter plot of the real reaction times against the predicted reaction times for the holdout data.
- feature_importance: Bar graph of the importance of the selected features, where importance is chosen to be the coefficients of the features.