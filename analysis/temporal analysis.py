# This code represents the temporal analysis in Section 3.1 of the paper

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ------------------------------
# CONFIGURATION
# ------------------------------
# Set the target bus and specefic day for the static analysis for analysis
bus = 'Bus115'
day = slice(360, 384)  # Change as needed for different windows

# ------------------------------
# LOAD DATA
# ------------------------------
# Replace with actual file paths if running locally
df_normal = pd.read_csv('LMP_Normal_cleaned.csv')
df_manipulated = pd.read_csv('cases_for_analysis/case1.csv')

# ------------------------------
# FUNCTION: Calculate CV for daily blocks
# ------------------------------
def calculate_cv(data_subset):
    mean_value = np.mean(data_subset)
    std_dev = np.std(data_subset)
    return (std_dev / mean_value) if mean_value != 0 else 0

def cv_cal(df):
    '''
    Compute Coefficient of Variation (CV) over 24-hour blocks for each column (bus).

    Returns:
    Dictionary with daily CV values for each bus.
    '''
    cv_results = {}
    for column_name in df.columns:
        data = df[column_name]
        grouped_data = [data[i:i+24] for i in range(0, len(data), 24)]
        cv_values = [calculate_cv(group) for group in grouped_data]
        cv_results[column_name] = cv_values
    return cv_results

# ------------------------------
# SECTION 1: VISIBILITY ANALYSIS (STATIC DISTRIBUTIONS)
# ------------------------------
normal_lmp = df_normal[bus].iloc[day]
manipulated_lmp = df_manipulated[bus].iloc[day]

# Calculate means, stds, medians
mean_normal = normal_lmp.mean()
mean_manipulated = manipulated_lmp.mean()
std_normal = normal_lmp.std()
std_manipulated = manipulated_lmp.std()
cv_normal = std_normal / mean_normal
cv_manipulated = std_manipulated / mean_manipulated
visibility = abs(cv_normal - cv_manipulated) / cv_normal

print(f"Visibility = {visibility:.4f}")

# Plot KDE distribution comparison
plt.figure(figsize=(6, 5))
sns.kdeplot(normal_lmp, label='Normal LMP', fill=True, alpha=0.4)
sns.kdeplot(manipulated_lmp, label='Manipulated LMP', fill=True, alpha=0.4)
plt.xlabel('LMP ($/MWh)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=12)
plt.grid()
plt.title(f'Static Distribution Comparison - {bus}')
plt.show()

# ------------------------------
# SECTION 2: DETECTABILITY ANALYSIS (DYNAMIC CV SHIFTS)
# ------------------------------
df_normal_cv = pd.DataFrame(cv_cal(df_normal))
df_manipulated_cv = pd.DataFrame(cv_cal(df_manipulated))

# Plot CV over time
plt.figure(figsize=(6, 4))
plt.plot(df_normal_cv[bus][:-1], label='Normal LMP')
plt.plot(df_manipulated_cv[bus][:-1], label='Manipulated LMP')
plt.xlabel('Time (Days)', fontsize=14)
plt.ylabel('Coefficient of Variation (CV)', fontsize=14)
plt.title(f'Temporal CV Comparison - {bus}')
plt.legend(fontsize=10)
plt.grid()
plt.tight_layout()
plt.show()

# ------------------------------
# SECTION 3: CALCULATE AVERAGE DELTA CV (∆CV)
# ------------------------------
def calculate_detectability(cv_series):
    '''
    Compute the average absolute difference between consecutive CV values.

    Returns:
    Float value indicating temporal detectability.
    '''
    deltas = [abs(cv_series[i] - cv_series[i - 1]) for i in range(1, len(cv_series))]
    return np.mean(deltas) if deltas else 0

detectability_normal = calculate_detectability(df_normal_cv[bus])
detectability_manipulated = calculate_detectability(df_manipulated_cv[bus])
difference = abs(detectability_normal - detectability_manipulated)

print(f"avg(∆CV)_Normal: {detectability_normal:.4f}")
print(f"avg(∆CV)_Manipulated: {detectability_manipulated:.4f}")
print(f"Detectability Difference: {difference:.4f}")
