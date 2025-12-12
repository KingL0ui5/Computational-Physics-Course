import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('dark')
sns.set_context('paper')
sns.set_palette("colorblind")
size = 13

plt.rc('font', size=size)
plt.rc('axes', titlesize=size)
plt.rc('axes', labelsize=size)
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('legend', fontsize=size-2)
plt.rc('figure', titlesize=size)

FILE_PATH = "Project/convergence data/convergence_summary_alpha_0.1.json"
WINDOW_SIZE = 5
SA_CUTOFF = 120


with open(FILE_PATH, 'r') as f:
    data = json.load(f)

    plt.figure(figsize=(10, 6))

    colors = {
        "Stochastic Reconfiguration": "#e41a1c",
        "Gradient Descent": "#377eb8",
        "RMSProp_GD": "#4daf4a",
        "Quasi-Newton DFP": "#984ea3",
        "Quasi-Newton BFGS": "#ff7f00",
        "Simulated Annealing": "#000000"
    }

    for method, details in data.items():
        history = details.get("Es_history", [])
        arr = np.array(history, dtype=np.float64)

        if np.isnan(arr).any():
            valid_indices = np.where(~np.isnan(arr))[0]
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1]
                arr = arr[:last_valid_idx+1]
            else:
                continue

        if method == "Simulated Annealing":
            arr = arr[:SA_CUTOFF]

        series = pd.Series(arr)
        moving_avg = series.rolling(window=WINDOW_SIZE, min_periods=1).mean()

        plt.plot(moving_avg, label=method, color=colors.get(
            method, 'gray'), linewidth=2)
        plt.plot(arr, color=colors.get(method, 'gray'),
                 linewidth=0.5, alpha=0.25)

        if method == "RMSProp_GD":
            residuals = np.abs(arr - moving_avg)
            std_dev = np.nanstd(residuals)
            threshold = 2.5 * std_dev
            anomalies_mask = residuals > threshold

            anomalous_x = np.where(anomalies_mask)[0]
            anomalous_y = arr[anomalies_mask]

            plt.scatter(anomalous_x, anomalous_y, color='red',
                        marker='x', s=60, zorder=5, label='RMSProp Anomalies')

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(f"Energy (Ha)", fontsize=12)
    plt.title("Convergence of H$_2$ Minimisers", fontsize=14)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
