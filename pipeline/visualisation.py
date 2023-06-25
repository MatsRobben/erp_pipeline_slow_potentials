from pipeline.models import chronological_train_test_split, assign_percentiles, calculate_p_values
from pipeline.utilities import save_plot, get_participant_nr

from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, mean_squared_error, auc
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
    
def visualize_average_feature_importance(models, feature_names, num_time_windows=3):
    """
    Visualizes the average feature importances per unique feature for a list of regression models.

    Args:
        models (list): List of regression models.
        feature_names (list): List of feature names for each model.

    Returns:
        None (displays the plot)
    """
    unique_features = np.unique(np.concatenate(feature_names))
    sorted_features = np.sort(unique_features)  # Sort the unique features

    colors = cm.tab10(np.linspace(0, 1, num_time_windows))  # Color cycling using tab10 colormap

    plt.figure(figsize=(10, 6))
    for idx, feature in enumerate(sorted_features):
        feature_importances = []
        for model, model_feature_names in zip(models, feature_names):
            if feature in model_feature_names:
                feature_importances.append(np.abs(model.coef_[model_feature_names.index(feature)]))
        average_importance = np.mean(feature_importances)

        plt.bar(idx, average_importance, color=colors[idx % num_time_windows])

    plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)  # Rotate x-axis labels for better visibility
    plt.xlabel('Feature')
    plt.ylabel('Average Feature Importance')
    plt.title('Average Feature Importances per Unique Feature')

    # Create custom legend for time windows
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(num_time_windows)]
    labels = [f'Time Window {i+1}' for i in range(num_time_windows)]
    plt.legend(handles, labels)

    plt.show()

def visualize_feature_importance(model, feature_names, par, gocue):
    """
    Visualizes the predictive power of features from a Lasso regression model.

    Args:
        model: Trained Lasso regression model with nonzero coefficients.
        feature_names (list): List of feature names corresponding to the model's nonzero coefficients.

    Returns:
        None (displays the plot)
    """
    importances = np.abs(model.coef_)

    # Sort the feature importances in descending order
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_feature_names = [feature_names[i] for i in indices]

    # Create horizontal bar plot
    fig = plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
    plt.yticks(range(len(sorted_importances)), sorted_feature_names)
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.title(f'Feature Importances participant {get_participant_nr(par)}')
    
    name = 'gocue_feature_importance' if gocue else 'activation_feature_importance'
    save_plot(fig, 'plots', name, '.svg', par=par)

def plot_p_values(regression_p_values, classification_p_values, participant_numbers):
    """
    Plots the p-values of regression and classification models for each participant.

    Args:
        regression_p_values (list): List of regression model p-values.
        classification_p_values (list): List of classification model p-values.
        participant_numbers (list): List of participant numbers.

    Returns:
        None (displays the plot)
    """
    num_participants = len(participant_numbers)

    # Set the x-axis range based on the number of participants
    x = np.arange(num_participants)

    # Set the width of each bar
    bar_width = 0.35

    # Plot regression p-values
    plt.bar(x, regression_p_values, width=bar_width, color='b', alpha=0.5, label='Regression')

    # Plot classification p-values with adjusted bar position
    plt.bar(x + bar_width, classification_p_values, width=bar_width, color='r', alpha=0.5, label='Classification')

    # Add participant numbers to x-axis
    plt.xticks(x + bar_width / 2, participant_numbers)

    # Draw a horizontal line at the p-value threshold of 0.05
    plt.axhline(y=0.05, color='k', linestyle='--', label='Threshold')

    plt.xlabel('Participant Number')
    plt.ylabel('p-value')
    plt.title('p-values of Regression and Classification Models')
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_scatter_lin_reg(X, y, par, reg_model, train=True, test_size=0.2, gocue=True):
    """
    Plot scatter plot and linear regression line for the given data.

    Args:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target values.
        par (int): Participant number.
        reg_model (sklearn.linear_model): Regression model object with `fit` and `predict` methods.
        train (bool, optional): If True, perform plotting for training data.
        test_size (float, optional): Test data size for train-test split.
        gocue (bool, optional): If True, plot for gocue data; otherwise, plot for activation data.
    """
    name = 'gocue_scatter_lin_reg' if gocue else 'activation_scatter_lin_reg'

    if train:
        X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, test_size=test_size)

        model = reg_model.fit(X_train, y_train)
        prediction_train = model.predict(X_train)
        prediction_test = model.predict(X_test)

        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_train, prediction_train, label="Train")
        plt.scatter(y_test, prediction_test, label="Test")

       
        xlim = plt.gca().get_xlim()
        plt.plot(xlim, xlim, 'k--')
        plt.axis('scaled')
        
        plt.title(f"Real and predicted reaction times for participant {get_participant_nr(par)}")
        plt.xlabel("Real Reaction Times (ms)")
        plt.ylabel("Predicted Reaction Times (ms)")
        plt.legend()

        save_plot(fig, 'plots/train', name, '.svg', par=par)
        
    else:
        prediction = reg_model.predict(X)

        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y, prediction)
        
        
        xlim = plt.gca().get_xlim()
        plt.plot(xlim, xlim, 'k--')
        plt.axis('scaled')
        
        plt.title(f"Real and predicted reaction times of holdout data for participant {get_participant_nr(par)}")
        plt.xlabel("Real Reaction Times (ms)")
        plt.ylabel("Predicted Reaction Times (ms)")

        save_plot(fig, 'plots/test', name, '.svg', par=par)

def plot_time_lin_reg(X, y, par, reg_model, train=True, test_size=0.2, gocue=True):
    """
    Plot the real and predicted reaction times over time using linear regression.

    Args:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target values.
        par (int): Participant number.
        reg_model (sklearn.linear_model): Regression model object with `fit`, `predict`, and `score` methods.
        train (bool, optional): If True, perform plotting for training data.
        test_size (float, optional): Test data size for train-test split.
        gocue (bool, optional): If True, plot for gocue data; otherwise, plot for activation data.
    """

    name = 'gocue_time_lin_reg' if gocue else 'activation_time_lin_reg'
 
    if train:
        X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, test_size=test_size)

        model = reg_model.fit(X_train, y_train)
        prediction_train = model.predict(X_train)
        prediction_test = model.predict(X_test)

        r2_train = model.score(X_train, y_train)
        r2_test = model.score(X_test, y_test)

        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        ax[0].plot(y_train, label='Real')
        ax[0].plot(prediction_train, label=f'Predict, $R^2$: {r2_train:.3f}')
        ax[0].set_xlabel('Trials')
        ax[0].set_ylabel('Reaction times (ms)')
        ax[0].set_title('Predicted reaction times train data')
        ax[0].legend()

        ax[1].plot(y_test, label='Real')
        ax[1].plot(prediction_test, label=f'Predict, $R^2$: {r2_test:.3f}')
        ax[1].set_xlabel('Trials')
        ax[1].set_ylabel('Reaction times (ms)')
        ax[1].set_title('Predicted reaction times test data')
        ax[1].legend()

        fig.suptitle(f'Reaction times for participant {get_participant_nr(par)}')

        save_plot(fig, 'plots/train', name, '.svg', par=par)

    else:
        prediction = reg_model.predict(X)
        mse = mean_squared_error(y, prediction)

        p_value = calculate_p_values(reg_model, X, y, alternative='greater')

        fig = plt.figure(figsize=(10, 6))
        plt.plot(y, label='Real')
        plt.plot(prediction, label=f'Predict, MSE: {mse:.2f}')
        plt.xlabel('Trials')
        plt.ylabel('Reaction times (ms)')
        plt.title(f'Predicted reaction times holdout data for participant {get_participant_nr(par)}, p-value={p_value:.5f}')
        plt.legend()

        save_plot(fig, 'plots/test', name, '.svg', par=par)

def lda_chron(X, y, class_model, par, plot_cm=False, plot_roc=False):
    """
    Perform LDA with chronological split cross-validation and plot results.

    Args:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target values.
        class_model (sklearn.discriminant_analysis.LinearDiscriminantAnalysis): LDA model object with `predict`, `predict_proba`, and `score` methods.
        par (int): Participant number.
        plot_cm (bool, optional): If True, plot confusion matrix for each split.
        plot_roc (bool, optional): If True, plot ROC curve for each split.

    Returns:
        matplotlib.figure.Figure: A figure with two subplots, one for the confusion matrix and one for the ROC curve.
    """

    y_pred = class_model.predict(X)
    
    p_value = calculate_p_values(class_model, X, y, alternative='greater')

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    if plot_cm:
        cm = confusion_matrix(y, y_pred)
        classes = np.unique(y)
        cmap = plt.cm.Blues
        axs[0].imshow(cm, interpolation='nearest', cmap=cmap)
        axs[0].set_title(f"Confusion matrix for test data")
        axs[0].set_xticks(np.arange(len(classes)))
        axs[0].set_yticks(np.arange(len(classes)))
        axs[0].set_xticklabels(classes)
        axs[0].set_yticklabels(classes)

        for j, k in product(range(len(classes)), range(len(classes))):
            axs[0].text(k, j, format(cm[j, k], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[j, k] > cm.max() / 2 else "black")

    if plot_roc:
        fpr, tpr, _ = roc_curve(y, class_model.predict_proba(X)[:, 1])
        roc_auc = roc_auc_score(y, class_model.predict_proba(X)[:, 1])
        axs[1].plot(fpr, tpr, label=f"ROC curve for test data (AUC = {roc_auc:.2f})")
        axs[1].plot([0, 1], [0, 1], 'k--')
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive Rate')
        axs[1].set_title(f"ROC Curve for test data")
        axs[1].legend(loc="lower right")

    fig.suptitle(f'LDA on holdout data of participant {get_participant_nr(par)}, p-value={p_value:.5f}')
    plt.tight_layout()

    return fig

def lda_chron_split_avg(X, y, n_splits, class_model, tscv, par, plot_cm=False, plot_roc=False):
    """
    Perform LDA with chronological split cross-validation and plot average results.

    Args:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target values.
        n_splits (int): Number of splits for time series cross-validation.
        class_model (sklearn.discriminant_analysis.LinearDiscriminantAnalysis): LDA model object with `fit`, `predict`, and `score` methods.
        tscv (sklearn.model_selection.TimeSeriesSplit): Time series cross-validation object.
        par (int): Participant number.
        plot_cm (bool, optional): If True, plot confusion matrix for each split.
        plot_roc (bool, optional): If True, plot ROC curve for each split.
    """
    acc_scores = []
    y_true_list, y_pred_list, y_prob_list = [], [], []

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    classes = np.unique(y)

    total_cm = np.zeros((len(classes), len(classes)))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        class_model.fit(X_train, y_train)
        y_pred = class_model.predict(X_test)
        acc = class_model.score(X_test, y_test)

        acc_scores.append(acc)
        y_true_list.append(y_test)
        y_pred_list.append(y_pred)
        y_prob_list.append(class_model.predict_proba(X_test)[:, 1])

        if plot_cm:
            cm = confusion_matrix(y_test, y_pred)
            total_cm += cm

        if plot_roc:
            fpr, tpr, _ = roc_curve(y_test, class_model.predict_proba(X_test)[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = roc_auc_score(y_test, class_model.predict_proba(X_test)[:, 1])
            aucs.append(roc_auc)

    avg_cm = total_cm / n_splits
    cmap = plt.cm.Blues
    axs[0].imshow(avg_cm, interpolation='nearest', cmap=cmap)
    axs[0].set_title("Average Confusion Matrix")
    axs[0].set_xticks(np.arange(len(classes)))
    axs[0].set_yticks(np.arange(len(classes)))
    axs[0].set_xticklabels(classes)
    axs[0].set_yticklabels(classes)

    for j, k in product(range(len(classes)), range(len(classes))):
        axs[0].text(k, j, format(int(avg_cm[j, k]), 'd'),
                    horizontalalignment="center",
                    color="white" if avg_cm[j, k] > avg_cm.max() / 2 else "black")

    mean_tpr = np.mean(tprs, axis=0)
    mean_acc = np.mean(acc_scores)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    axs[1].plot(mean_fpr, mean_tpr, label=f"Average ROC curve (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})")
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title("Average ROC Curve")
    axs[1].legend(loc="lower right")

    fig.suptitle(f'Average over cross-validation folds of LDA, for participant {get_participant_nr(par)} \n Accuracy: {mean_acc:.3f}')
    plt.tight_layout()


    return fig

def lda_chron_split(X, y, n_splits, class_model, tscv, par, plot_cm=False, plot_roc=False):
    """
    Perform LDA with chronological split cross-validation.

    Args:
        X (numpy.ndarray): Input features.
            Input data of shape (n_samples, n_features).
        y (numpy.ndarray): Target values.
            Target data of shape (n_samples,).
        n_splits (int): Number of splits for time series cross-validation.
        class_model (sklearn.discriminant_analysis.LinearDiscriminantAnalysis): LDA model object.
            An instance of LDA model with `fit`, `predict`, and `score` methods.
        tscv (sklearn.model_selection.TimeSeriesSplit): Time series cross-validation object.
            An instance of TimeSeriesSplit for splitting the data.
        par (int): Participant number.
            The participant number for identification.
        plot_cm (bool, optional): Whether to plot the confusion matrix for each split. Defaults to False.
        plot_roc (bool, optional): Whether to plot the ROC curve for each split. Defaults to False.

    Returns:
        acc_scores (list): The accuracy scores obtained for the different splits.
            A list containing the accuracy scores for each split.
    """
    acc_scores = []
    y_true_list, y_pred_list, y_prob_list = [], [], []

    fig, axs = plt.subplots(n_splits, 2, figsize=(10, 10))

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        class_model.fit(X_train, y_train)
        y_pred = class_model.predict(X_test)
        acc = class_model.score(X_test, y_test)

        acc_scores.append(acc)
        y_true_list.append(y_test)
        y_pred_list.append(y_pred)
        y_prob_list.append(class_model.predict_proba(X_test)[:, 1])

        if plot_cm:
            cm = confusion_matrix(y_test, y_pred)
            classes = np.unique(y)
            cmap = plt.cm.Blues
            axs[i, 0].imshow(cm, interpolation='nearest', cmap=cmap)
            axs[i, 0].set_title(f"Confusion matrix for split {i}")
            axs[i, 0].set_xticks(np.arange(len(classes)))
            axs[i, 0].set_yticks(np.arange(len(classes)))
            axs[i, 0].set_xticklabels(classes)
            axs[i, 0].set_yticklabels(classes)

            for j, k in product(range(len(classes)), range(len(classes))):
                axs[i, 0].text(k, j, format(cm[j, k], 'd'),
                               horizontalalignment="center",
                               color="white" if cm[j, k] > cm.max() / 2 else "black")

        if plot_roc:
            fpr, tpr, _ = roc_curve(y_test, class_model.predict_proba(X_test)[:, 1])
            roc_auc = roc_auc_score(y_test, class_model.predict_proba(X_test)[:, 1])
            axs[i, 1].plot(fpr, tpr, label=f"ROC curve for split {i} (AUC = {roc_auc:.2f})")
            axs[i, 1].plot([0, 1], [0, 1], 'k--')
            axs[i, 1].set_xlabel('False Positive Rate')
            axs[i, 1].set_ylabel('True Positive Rate')
            axs[i, 1].set_title(f"ROC Curve for split {i}")
            axs[i, 1].legend(loc="lower right")

    fig.suptitle(f'{n_splits} fold cross-validation of LDA for participant {get_participant_nr(par)}')
    plt.tight_layout()

    return fig

def plot_lin_class(X, y, par, class_model, n_splits=5, train=True, gocue=True):
    """
    Plot linear classification results using LDA.

    Args:
        X (numpy.ndarray): Input features.
            Input data of shape (n_samples, n_features).
        y (numpy.ndarray): Target values.
            Target data of shape (n_samples,).
        par (int): Participant number.
            The participant number for identification.
        class_model (sklearn.discriminant_analysis.LinearDiscriminantAnalysis): LDA model object.
            An instance of LDA model with `fit`, `predict`, and `score` methods.
        n_splits (int, optional): Number of splits for time series cross-validation. Defaults to 5.
        train (bool, optional): Whether to perform plotting for training data. Defaults to True.
        gocue (bool, optional): Whether to plot for gocue data. If False, plot for activation data. Defaults to True.

    Returns:
        None
    """
    name = 'gocue_lin_class_scores' if gocue else 'activation_lin_class_scores'

    if train:
        fig = lda_chron_split(X, y, n_splits=n_splits, class_model=class_model, par=par,
                          tscv=KFold(n_splits=n_splits), plot_cm=True, plot_roc=True)

        save_plot(fig, 'plots/train', name, '.svg', par=par)

        fig_avg = lda_chron_split_avg(X, y, n_splits=n_splits, class_model=class_model, par=par,
                          tscv=KFold(n_splits=n_splits), plot_cm=True, plot_roc=True)
        
        save_plot(fig_avg, 'plots/train', f'{name}_avg', '.svg', par=par)
    else:
        fig = lda_chron(X, y, class_model, par, plot_cm=True, plot_roc=True)

        save_plot(fig, 'plots/test', name, '.svg', par=par)