# analysis_utils.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1,1)
    y_pred = np.asarray(y_pred).reshape(-1,1)
    return float(np.mean((y_true - y_pred)**2))

def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1,1)
    y_pred = np.asarray(y_pred).reshape(-1,1)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    denom = np.max(y_true) - np.min(y_true)
    return float(rmse / denom) if denom != 0 else float('nan')

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1,1)
    y_pred = np.asarray(y_pred).reshape(-1,1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res/ss_tot) if ss_tot != 0 else float('nan')

def parity_plot(y_true, y_pred, out_png, title="y vs y_hat"):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], linestyle='--')
    plt.xlabel("y (real)")
    plt.ylabel("ŷ (predicho)")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_summary_csv_row(csv_path, row_dict, header_order):
    import csv
    file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row_dict.get(k, "") for k in header_order})

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
