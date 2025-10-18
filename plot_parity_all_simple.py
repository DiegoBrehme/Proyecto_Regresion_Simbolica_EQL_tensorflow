import numpy as np
import matplotlib.pyplot as plt
import os

def plot_parity(file_path, title):
    if not os.path.exists(file_path):
        print(f"⚠ No existe {file_path}, se omite.")
        return
    data = np.load(file_path)
    if data.shape[1] < 3:
        print(f" Formato inesperado en {file_path}, se omite.")
        return
    X, y_true, y_pred = data[:, 0], data[:, 1], data[:, 2]

    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("y real")
    plt.ylabel("y predicha")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_parity("results/101/preds_val.npy", "Paridad F1 - Modelo 101")
plot_parity("results/102/preds_val.npy", "Paridad F2 - Modelo 102")
plot_parity("results/103/preds_val.npy", "Paridad F3 - Modelo 103")
plot_parity("results/104/preds_val.npy", "Paridad F4 - Modelo 104")
plot_parity("results/205/preds_val.npy", "Paridad F5 - Modelo 205")
plot_parity("results/206/preds_val.npy", "Paridad F6 - Modelo 206")
plot_parity("results/305/preds_val.npy", "Paridad F7 - Modelo 305")
plot_parity("results/306/preds_val.npy", "Paridad F8 - Modelo 306")
plot_parity("results/401/preds_val.npy", "Paridad F9 - Modelo 401")
plot_parity("results/402/preds_val.npy", "Paridad F10 - Modelo 402")
