import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('diabetes.csv')

# Filter FIRST
df = df[(df['Glucose'] > 0) & (df['Insulin'] > 0)]

# Then split
split_index = int(len(df) * 0.6)

X_train_pd = df[:split_index][['Glucose', 'Insulin']]
x_val_pd = df[split_index:][['Glucose', 'Insulin']]
y_val_pd = df[split_index:]['Outcome']

X_train = X_train_pd.values
x_val = x_val_pd.values
y_val = y_val_pd.values


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
x_val = scaler.transform(x_val)

def estimate_gaussian(X):
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m,n = X.shape

    mu = 1/m * (np.sum(X, axis=0))
    var = 1/m * (np.sum((X - mu)**2, axis=0))

    return mu,var

mu, var = estimate_gaussian(X_train)

def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 
    F1 = 0
    best_epsilon = 0
    best_F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000
    p_val = p_val.flatten()
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        prediction = p_val < epsilon

        tp = np.sum((prediction == 1) & (y_val == 1))
        fp = np.sum((prediction == 1) & (y_val == 0))
        fn = np.sum((prediction == 0) & (y_val == 1))

        #Included if statement as prec and rec were being 0 and calculating as 0/0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        F1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1

def multivariate_gaussian(X, mu, var):
    """
    Computes the probability density function of the examples X under 
    the multivariate gaussian distribution with parameters mu and var.

    Args:
        X (ndarray): (m, n) matrix where each row is a sample
        mu (ndarray): (n,) mean
        var (ndarray): (n,) variance (diagonal covariance matrix)

    Returns:
        p (ndarray): (m,) probabilities for each sample
    """
    k = len(mu)
    sigma2 = np.diag(var)
    X = X - mu

    denom = np.sqrt((2 * np.pi) ** k * np.linalg.det(sigma2))
    num = np.exp(-0.5 * np.sum(X @ np.linalg.inv(sigma2) * X, axis=1))

    return num / denom

# Step 1: Compute probability for validation set
p_val = multivariate_gaussian(x_val, mu, var)

# Step 2: Select the best threshold using y_val
epsilon = 0.06

print(f"Best epsilon: {epsilon}")

# Step 3: Predict anomalies
predictions = p_val < epsilon

# Optional: Print indices or count of anomalies
print(f"Found {np.sum(predictions)} anomalies in validation set out of {len(x_val)} samples")


# Normal points
normal_indices = np.where(~predictions)[0]
anomaly_indices = np.where(predictions)[0]

# Plot
normal_indices = np.where(~(p_val < 0.005))[0]
anomaly_indices = np.where(p_val < 0.005)[0]

plt.scatter(x_val[~predictions][:, 0], x_val[~predictions][:, 1], c='b', label='Normal')
plt.scatter(x_val[predictions][:, 0], x_val[predictions][:, 1], c='r', marker='x', label='Anomaly')

plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.title("Anomaly Detection (Glucose vs Insulin)")
plt.legend()
plt.show()

predictions = p_val < 0.005
print(classification_report(y_val, predictions, target_names=["Normal", "Anomaly"]))