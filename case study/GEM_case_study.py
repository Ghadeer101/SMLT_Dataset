
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# ------------------------------
# Utility Function: Compute kNN Distance
# ------------------------------
def compute_kNN_distance(point, dataset, k):
    '''
    Compute the average distance from a given point to its k-nearest neighbors in the dataset.

    Parameters:
    point: A 1D NumPy array representing the data point.
    dataset: A 2D NumPy array of historical/baseline data.
    k: Number of nearest neighbors.

    Returns:
    Mean distance to the k-nearest neighbors.
    '''
    nbrs = NearestNeighbors(n_neighbors=k).fit(dataset)
    distances, _ = nbrs.kneighbors([point])
    return np.mean(distances)

# ------------------------------
# Offline Phase: Prepare kNN Baseline
# ------------------------------
def offline_phase(data, k):
    '''
    Create two subsets from the baseline data, and compute distances from one subset to another.

    Parameters:
    data: 2D NumPy array (samples x features).
    k: Number of neighbors to consider.

    Returns:
    sorted_distances: Sorted list of distances (used for thresholding).
    S1: Subset used for nearest neighbor reference.
    '''
    N = len(data)
    S1 = data[:N // 2]  # Reference subset
    S2 = data[N // 2:]  # Evaluation subset

    distances = [compute_kNN_distance(x_j, S1, k) for x_j in S2]
    return sorted(distances), S1

# ------------------------------
# Online Detection Phase (GEM-based Anomaly Detection)
# ------------------------------
def detection_code(df_test):
    '''
    Main anomaly detection logic using statistical scoring of nearest-neighbor distances.

    Parameters:
    df_test: Pandas DataFrame containing bus-wise LMP values and labels.

    Outputs:
    Evaluation metrics including precision, recall, F1 score, AUC, and false alarm rate.
    '''
    # Detection parameters
    window = 336           # Initial window size (2 week)
    k = 2                  # Number of neighbors
    h = 2                  # Detection threshold
    alpha = 0.05           # Sensitivity parameter
    epsilon = 1e-2         # To avoid division by zero
    decay = 0.98           # Score decay factor

    # Prepare test data
    actual_label = df_test['Label'].iloc[window:].tolist()
    df = df_test.drop(columns=['Week', 'Label'])
    data = df.to_numpy()

    # Load clean LMP data for training baseline
    original_df = pd.read_csv('LMP_Normal_cleaned.csv')
    original_df.drop(columns=['Week', 'Label'], inplace=True)
    training_data = original_df.to_numpy()
    baseline_data = training_data[:window]

    # Compute reference distances from clean baseline
    sorted_distances, S1 = offline_phase(baseline_data, k)

    # Begin online detection
    g_t = 0
    all_g_t = []
    detection_label = []

    for index in range(len(data)):
        if index < window:
            continue  # Skip warm-up period

        x_t = data[index]
        d_t = compute_kNN_distance(x_t, S1, k)
        p_hat_t = np.sum([1 if d_j > d_t else 0 for d_j in sorted_distances]) / len(sorted_distances)
        s_hat_t = np.log(alpha / (p_hat_t + epsilon))
        g_t = max(0, decay * g_t + s_hat_t)
        all_g_t.append(g_t)

        label = 1 if g_t >= h else 0
        detection_label.append(label)

    # Compute and print evaluation metrics
    tn, fp, fn, tp = confusion_matrix(actual_label, detection_label).ravel()
    print(f'Detection Rate (Recall): {recall_score(actual_label, detection_label):.3f}')
    print(f'Precision: {precision_score(actual_label, detection_label):.3f}')
    print(f'F1 Score: {f1_score(actual_label, detection_label):.3f}')
    print(f'AUC: {roc_auc_score(actual_label, detection_label):.3f}')
    print(f'False Alarm Rate: {fp / (fp + tn):.3f}')

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    # Load case data (replace with correct file path)
    df = pd.read_csv('cases_for_analysis/case1.csv')

    # Use df.iloc[:700] to test GEM before drift
    # Use df.iloc[700:] to test GEM after drift
    detection_code(df.iloc[700:])
