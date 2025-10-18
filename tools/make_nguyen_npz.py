# tools/make_nguyen_npz.py
import os
import argparse
import numpy as np
import pandas as pd

def load_csv(path):
    df = pd.read_csv(path)
    xcols = [c for c in df.columns if c.startswith("x")]
    X = df[xcols].values.astype(np.float32)
    y = df["y"].values.astype(np.float32).reshape(-1,1)
    return X, y

def save_npz(out_base, Xtr, ytr, Xval, yval, Xte, yte):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    # Formato que leeremos con data_utils/get_input_fns
    np.savez(out_base + "_train_val.npz",
             train_inputs=Xtr, train_targets=ytr,
             val_inputs=Xval,   val_targets=yval)
    np.savez(out_base + "_test.npz",
             test_inputs=Xte, test_targets=yte)

def main(base_folder, out_dir, F):
    Fdir = os.path.join(base_folder, f"F{F}")
    Xtr, ytr = load_csv(os.path.join(Fdir, f"F{F}_test_extrap.csv"))  # ENTRENAR con extrap
    Xall, yall = load_csv(os.path.join(Fdir, f"F{F}_train.csv"))      # VALIDAR/TEST con train
    n = len(Xall)
    n_val = max(1, int(0.8*n))
    Xval, yval = Xall[:n_val], yall[:n_val]
    Xte,  yte  = Xall[n_val:], yall[n_val:]

    out_base = os.path.join(out_dir, f"F{F}data")
    save_npz(out_base, Xtr, ytr, Xval, yval, Xte, yte)
    print(f"[OK] {out_base}_train_val.npz  |  {out_base}_test.npz")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_folder", required=True)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--F", type=int, required=True)
    args = ap.parse_args()
    main(args.base_folder, args.out_dir, args.F)
