
# -------------------------------------------
# Analysis 2: Spatial Spreadability of Attack
# (Related to Section 3.2)
# -------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------------
# STEP 1: Parse MATPOWER File
# ------------------------------
def parse_matpower_case(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    def extract_matrix(lines, start_key):
        data = []
        collecting = False
        for line in lines:
            if start_key in line:
                collecting = True
                continue
            if collecting:
                if '];' in line:
                    break
                parts = line.strip().strip(';').split()
                row = [float(p) for p in parts if re.match(r'^-?\d+(\.\d+)?$', p)]
                if row:
                    data.append(row)
        return np.array(data)

    bus_data = extract_matrix(lines, 'mpc.bus')
    branch_data = extract_matrix(lines, 'mpc.branch')
    return bus_data, branch_data

# ------------------------------
# STEP 2: Build Electrical Distance Matrix
# ------------------------------
def compute_electrical_distance_matrix(bus_data, branch_data):
    buses = bus_data[:, 0].astype(int)
    unique_buses = sorted(set(buses))
    G = nx.Graph()

    for row in branch_data:
        if len(row) < 4:
            continue
        fbus, tbus, r, x = int(row[0]), int(row[1]), row[2], row[3]
        z = complex(r, x)
        if abs(z) > 0:
            G.add_edge(fbus, tbus, weight=abs(z))

    nb = len(unique_buses)
    bus_index = {bus: idx for idx, bus in enumerate(unique_buses)}
    distance_matrix = np.zeros((nb, nb))

    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    for i, bi in enumerate(unique_buses):
        for j, bj in enumerate(unique_buses):
            if bj in lengths.get(bi, {}):
                distance_matrix[i, j] = lengths[bi][bj]
            else:
                distance_matrix[i, j] = np.inf

    np.fill_diagonal(distance_matrix, 0)
    finite_mask = np.isfinite(distance_matrix)
    max_dist = np.max(distance_matrix[finite_mask])
    distance_matrix[~finite_mask] = max_dist * 1.1

    return unique_buses, G, distance_matrix

# ------------------------------
# STEP 3: Load Data and Compute ΔLMP
# ------------------------------
# Load LMP time series
df1 = pd.read_csv('data/Normal_LMP.csv')
df2 = pd.read_csv('data/case1.csv')

# Drop unused columns if present
cols_to_drop = ['Week', 'Label']
df1_clean = df1.drop(columns=[col for col in cols_to_drop if col in df1.columns], errors='ignore')
df2_clean = df2.drop(columns=[col for col in cols_to_drop if col in df2.columns], errors='ignore')

# Select time step to analyze
time_step = 400
delta_lmp_at_t = df2_clean.iloc[time_step] - df1_clean.iloc[time_step]
abs_delta_lmp_at_t = np.abs(delta_lmp_at_t)

# ------------------------------
# STEP 4: Spreadability Computation
# ------------------------------
filepath = "NPCC.m"  # Replace with your own path
bus_data, branch_data = parse_matpower_case(filepath)
buses, G, distance_matrix = compute_electrical_distance_matrix(bus_data, branch_data)

target_bus = 115  # Change based on the selected case, as described in the paper
try:
    target_idx = buses.index(target_bus)
except ValueError:
    print(f"Bus {target_bus} not found.")
    target_idx = None

if target_idx is not None:
    distances = distance_matrix[target_idx]

    # Combine ΔLMP and distance
    data = [{
        'Bus': bus,
        'Distance_from_target': dist,
        'Abs_Delta_LMP': abs_delta_lmp_at_t.get(bus, np.nan)
    } for bus, dist in zip(buses, distances)]

    df_distance_lmp = pd.DataFrame(data).dropna()

    # Bin by distance
    bin_width = 0.05
    bins = np.arange(0, df_distance_lmp['Distance_from_target'].max() + bin_width, bin_width)
    df_distance_lmp['Distance_Bin'] = pd.cut(df_distance_lmp['Distance_from_target'], bins)

    # Group by bin
    df_avg = df_distance_lmp.groupby('Distance_Bin').agg({
        'Distance_from_target': 'mean',
        'Abs_Delta_LMP': 'mean'
    }).reset_index()

    # Smooth using LOWESS
    smoothed = lowess(df_avg['Abs_Delta_LMP'], df_avg['Distance_from_target'], frac=0.5)

    # Plot
    plt.figure(figsize=(5, 3))
    plt.scatter(df_avg['Distance_from_target'], df_avg['Abs_Delta_LMP'], label='Averaged LMP')
    plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2, label='Trend (LOWESS)')
    plt.xlabel('Electrical Distance')
    plt.ylabel('avg |ΔLMP|')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Spreadability (AUC)
    auc_spreadability = np.trapezoid(df_avg['Abs_Delta_LMP'], df_avg['Distance_from_target'])
    print(f"AUC (Spreadability): {auc_spreadability:.4f}")
