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

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    test_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.

    Returns
    -------
    X_train : array-like of shape (n_train_samples, n_features)
        The training input samples.
    X_test : array-like of shape (n_test_samples, n_features)
        The testing input samples.
    y_train : array-like of shape (n_train_samples,)
        The training target values.
    y_test : array-like of shape (n_test_samples,)
        The testing target values.
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
    Takes an array and assigns a 0 to each element that is below the low percentile 
    and a 1 to each element that is above the high percentile.
    
    Arguments:
    array -- a NumPy array of data to classify
    low_pct -- the percentile below which data should be classified as 0
    high_pct -- the percentile above which data should be classified as 1
    
    Returns:
    A tuple containing:
    - A NumPy array of the assigned classes (0 or 1)
    - A list of indices that were not included in the classification 
    (i.e., between the low and high percentiles)
    """
    low_val = np.percentile(array, low_pct)
    high_val = np.percentile(array, high_pct)
    classes = np.where(array < low_val, 0, np.where(array > high_val, 1, -1))
    indices = np.where(classes == -1)[0]
    
    return np.delete(classes, indices), indices

def calculate_p_values(model, X, y, n_permutations=10000, alternative='two-sided'):
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


def calculate_p_values_classification(model, X, y, n_permutations=10000):
    coef = model.coef_[0]
    n_features = X.shape[1]
    
    # Calculate the observed test statistic
    lda_score = model.score(X, y)
    
    # Permutation testing
    permutation_scores = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Permute the response variable
        permuted_y = np.random.permutation(y)
        
        # Calculate the test statistic for permuted data
        permuted_score = model.score(X, permuted_y)
        permutation_scores[i] = permuted_score
    
    plt.hist(permutation_scores)
    plt.title('Classification')
    plt.show()

    # Calculate the p-values
    p_values = (len(np.where(permutation_scores >= lda_score)[0]) + 1) / (n_permutations + 1) 
    print(p_values)

    return p_values
