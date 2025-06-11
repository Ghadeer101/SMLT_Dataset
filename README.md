
# SMLT: Stealthy Manipulated LMP Timeseries Dataset

The **SMLT Dataset** provides time-series data and simulation scripts for evaluating detection models with stealthy cyber-physical attacks on electricity markets. It focuses on how different false data injection attacks (FDIAs) can influence **Locational Marginal Prices (LMPs)** in power systems. 

---

##  Repository Structure

```
SMLT_Dataset/
├── metadata.json                  # Metadata describing dataset and case types
├── data/                          # Raw LMP time series (normal + 8 attack cases)
│   ├── Normal_LMP.csv
│   ├── case1.csv  ...  case8.csv
├── analysis/                      # Python scripts for statistical visibility & spreadability
│   ├── temporal analysis.py       # Implements ∆CV detectability (Section 3.1)
│   └── spatial analysis.py        # Implements spreadability metric (Section 3.2)
├── case study/
│   └── GEM_case_study.py          # GEM-based online anomaly detection (Section 4)
├── scripts/
│   ├── FDIA simulation.m          # MATLAB simulation code to generate attack cases
│   └── preprocessing.py           # Cleaning pipeline for LMP data
```

---

## 📦 Dataset Description

Each case file contains hourly LMP values over 20 weeks across 140 buses. The `Normal_LMP.csv` represents the baseline, while `caseX.csv` files contain specific FDIA scenarios.

- Rows: Time (hourly)
- Columns: `Timestamp`, `Label`, `Bus1`, `Bus2`, ..., `Bus140`
- Label: 1 if under attack, 0 otherwise

Attack cases included:
- `case1.csv`: Line rating (RATE_A) reduction
- `case2.csv`: XR parameter manipulation
- `case3.csv`: Breaker status manipulation
- `case4.csv`: Load injection
- `case5–8.csv`: Advanced stealthy combinations (as described in paper)

---

## 🧪 Analysis Scripts

- `temporal analysis.py`: Computes statistical detectability via coefficient of variation drift.
- `spatial analysis.py`: Calculates spreadability using electrical distances (AUC metric).
- `GEM_case_study.py`: Online anomaly detection using nearest-neighbor scoring.

---


