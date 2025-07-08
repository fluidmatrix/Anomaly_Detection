# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset from CSV
df = pd.read_csv('diabetes.csv')

# Filter out invalid data: remove rows where Glucose or Insulin are zero or less
df = df[(df['Glucose'] > 0) & (df['Insulin'] > 0)]

# Split the data into training and validation sets (60% training, 40% validation)
split_index = int(len(df) * 0.6)

X_train_pd = df[:split_index][['Glucose', 'Insulin']]  # Training features
x_val_pd = df[split_index:][['Glucose', 'Insulin']]     # Validation features
y_val_pd = df[split_index:]['Outcome']                 # Validation labels

# Convert from pandas to NumPy arrays
X_train = X_train_pd.values
x_val = x_val_pd.values
y_val = y_val_pd.values

# Feature scaling for better numerical stability
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
x_val = scaler.transform(x_val)

# Function to estimate the Gaussian parameters (mean and variance) for each feature
def estimate_gaussian(X):
    """
    Calculates mean and variance of all features in the dataset.
    """
    m, n = X.shape
    mu = 1/m * (np.sum(X, axis=0))
    var = 1/m * (np.sum((X - mu)**2, axis=0))
    return mu, var

# Estimate Gaussian parameters on the training data
mu, var = estimate_gaussian(X_train)

# Function to compute the multivariate Gaussian probability for each sample
def multivariate_gaussian(X, mu, var):
    """
    Computes the probability density function of the examples X under
    the multivariate Gaussian distribution.
    """
    k = len(mu)
    sigma2 = np.diag(var)  # Construct diagonal covariance matrix
    X = X - mu             # Center the data

    denom = np.sqrt((2 * np.pi) ** k * np.linalg.det(sigma2))
    num = np.exp(-0.5 * np.sum(X @ np.linalg.inv(sigma2) * X, axis=1))

    return num / denom

# Compute the anomaly probabilities for validation set
p_val = multivariate_gaussian(x_val, mu, var)

# (Optional): Function to select best epsilon threshold using F1 score
def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers using validation data.
    """
    F1 = 0
    best_epsilon = 0
    best_F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000
    p_val = p_val.flatten()

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        prediction = p_val < epsilon

        tp = np.sum((prediction == 1) & (y_val == 1))  # True Positives
        fp = np.sum((prediction == 1) & (y_val == 0))  # False Positives
        fn = np.sum((prediction == 0) & (y_val == 1))  # False Negatives

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        F1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1

# Manually selected threshold (or use select_threshold to find the best one)
epsilon = 0.06  # Can tune or use select_threshold()
print(f"Best epsilon: {epsilon}")

# Classify each point in validation set as anomaly or not
predictions = p_val < epsilon

# Report number of anomalies found
print(f"Found {np.sum(predictions)} anomalies in validation set out of {len(x_val)} samples")

# Identify indices of normal and anomalous points
normal_indices = np.where(~predictions)[0]
anomaly_indices = np.where(predictions)[0]

# Plot normal and anomalous points in validation set
plt.scatter(x_val[~predictions][:, 0], x_val[~predictions][:, 1], c='b', label='Normal')
plt.scatter(x_val[predictions][:, 0], x_val[predictions][:, 1], c='r', marker='x', label='Anomaly')

plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.title("Anomaly Detection (Glucose vs Insulin)")
plt.legend()
plt.show()

# Final prediction using tighter threshold for evaluation
predictions = p_val < 0.005
print(classification_report(y_val, predictions, target_names=["Normal", "Anomaly"]))
