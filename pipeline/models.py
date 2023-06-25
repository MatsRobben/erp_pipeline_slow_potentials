import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def visualize_significant_distribution(samples, p_value, observed_value):
    """
    Visualizes a significant distribution with observed value and p-value shading.

    Args:
        samples (array-like): Array of samples forming the distribution.
        p_value (float): The p-value for the observed value.
        observed_value (float): The observed value to be highlighted.

    Returns:
        None (displays the plot)
    """
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(samples, bins=10, density=True, alpha=0.5, label='Distribution', edgecolor='black', linewidth=1.2, histtype='step')

    # Create a density line plot
    x = np.linspace(min(samples), max(samples), 100)
    kernel = stats.gaussian_kde(samples)
    plt.plot(x, kernel(x), color='black', linewidth=2, label='Distribution Line')

    plt.axvline(observed_value, color='r', linestyle='--', linewidth=2, label='Observed Value')

    shaded_samples = samples[samples >= observed_value]
    plt.fill_between(shaded_samples, np.zeros_like(shaded_samples),
                     alpha=0.3, color='g', label='p-value Area')

    # Scale the histogram counts to align with the density line
    max_count = max(counts)
    plt.ylim(0, max_count + max_count * 0.2)

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Significant Distribution with Observed Value and p-value Area')
    plt.legend()
    plt.show()

def chronological_train_test_split(X, y, test_size):
    """
    Split the data into training and testing sets while keeping the data in chronological order.

    Args:
        X (array-like): The input samples.
            Input data of shape (n_samples, n_features).
        y (array-like): The target values.
            Target data of shape (n_samples,).
        test_size (float or int): The size of the test set.
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.

    Returns:
        X_train (array-like): The training input samples.
            Training data of shape (n_train_samples, n_features).
        X_test (array-like): The testing input samples.
            Testing data of shape (n_test_samples, n_features).
        y_train (array-like): The training target values.
            Training target data of shape (n_train_samples,).
        y_test (array-like): The testing target values.
            Testing target data of shape (n_test_samples,).
    """

    # Determine the number of test samples based on the test_size parameter
    if isinstance(test_size, float):
        n_test_samples = int(len(X) * test_size)
    elif isinstance(test_size, int):
        n_test_samples = test_size
    else:
        raise ValueError("Invalid value for test_size parameter")

    # Determine the number of training samples
    n_train_samples = len(X) - n_test_samples

    # Split the data into training and testing sets
    X_train = X[:n_train_samples, :]
    X_test = X[n_train_samples:, :]
    y_train = y[:n_train_samples]
    y_test = y[n_train_samples:]

    return X_train, X_test, y_train, y_test


def assign_percentiles(array, low_pct, high_pct):
    """
    Assigns a class to each element in an array based on percentiles.

    Args:
        array (numpy.ndarray): The array of data to classify.
        low_pct (float): The percentile below which data should be classified as 0.
        high_pct (float): The percentile above which data should be classified as 1.

    Returns:
        tuple: A tuple containing:
            - classes (numpy.ndarray): An array of assigned classes (0 or 1).
            - indices (list): A list of indices that were not included in the classification
              (i.e., between the low and high percentiles).
    """
    low_val = np.percentile(array, low_pct)
    high_val = np.percentile(array, high_pct)
    classes = np.where(array < low_val, 0, np.where(array > high_val, 1, -1))
    indices = np.where(classes == -1)[0]
    
    return np.delete(classes, indices), indices

def calculate_p_values(model, X, y, n_permutations=10000, alternative='two-sided'):
    """
    Calculate p-values using permutation testing.

    Args:
        model: The model object with a `score` method.
            The model used to calculate the test statistic.
        X (array-like): Input features.
            Input data of shape (n_samples, n_features).
        y (array-like): Target values.
            Target data of shape (n_samples,).
        n_permutations (int, optional): The number of permutations. Defaults to 10000.
        alternative (str, optional): The alternative hypothesis for the test. 
            Should be one of 'two-sided', 'less', or 'greater'. Defaults to 'two-sided'.

    Returns:
        p_value (float): The calculated p-value based on the chosen alternative.
            The p-value for the hypothesis test.
    """

    # Calculate the observed test statistic
    score = model.score(X, y)
    
    # Initialize an array to store permuted test statistics
    permutation_scores = np.zeros(n_permutations)
    
    # Permutation testing
    for i in range(n_permutations):
        # Permute the response variable
        permuted_y = np.random.permutation(y)
        
        # Calculate the test statistic for permuted data
        permuted_score = model.score(X, permuted_y)
        permutation_scores[i] = permuted_score

    # Calculate the mean of the permuted test statistics
    # permuted_mean = np.mean(permutation_scores)
    
    # # Center the test statistics around zero
    # permutation_scores -= permuted_mean

    # score -= permuted_mean
    
    # Calculate the p-value based on the chosen alternative
    if alternative == 'two-sided':
        # Calculate the p-value for two-sided test
        p_value = (len(np.where(np.abs(permutation_scores) >= np.abs(score))[0]) + 1) / (n_permutations + 1)
    elif alternative == 'less':
        # Calculate the p-value for less than test
        p_value = (len(np.where(permutation_scores <= score)[0]) + 1) / (n_permutations + 1)
    elif alternative == 'greater':
        # Calculate the p-value for greater than test
        p_value = (len(np.where(permutation_scores >= score)[0]) + 1) / (n_permutations + 1)
    else:
        raise ValueError("Invalid alternative. Please choose 'two-sided', 'less', or 'greater'.")
    
    # Print the observed test statistic and p-value
    print("Observed Test Statistic:", score)
    print(f"{alternative.capitalize()} p-value:", p_value)

    # Plot the histogram of permuted test statistics
    # plt.hist(permutation_scores, bins=50)
    # plt.axvline(x = score)
    # plt.show()
    # visualize_significant_distribution(permutation_scores, p_value, score)

    return p_value