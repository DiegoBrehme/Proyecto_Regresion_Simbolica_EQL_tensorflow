# tools/npz_to_repo_data.py
import os, gzip, pickle, argparse, numpy as np

def save_gz_pickle(obj, out_path_no_ext):
    os.makedirs(os.path.dirname(out_path_no_ext), exist_ok=True)
    with gzip.open(out_path_no_ext, "wb") as f:
        # protocolo alto OK en Py3.7/TF1
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def convert_one(base_no_ext):
    """
    Lee:
      base_train_val.npz  con claves: train_inputs, train_targets, val_inputs, val_targets
      base_test.npz       con claves: test_inputs,  test_targets
    Escribe (SIN extensión, formato gzip+pickle):
      base_train_val  -> (Xtr, ytr, Xva, yva)
      base_test       -> (Xte, yte)
    """
    tv_npz = base_no_ext + "_train_val.npz"
    te_npz = base_no_ext + "_test.npz"
    if not os.path.isfile(tv_npz) or not os.path.isfile(te_npz):
        raise FileNotFoundError(f"Falta npz: {tv_npz} o {te_npz}")

    tv = np.load(tv_npz)
    te = np.load(te_npz)

    Xtr, ytr = tv["train_inputs"], tv["train_targets"]
    Xva, yva = tv["val_inputs"],   tv["val_targets"]
    Xte, yte = te["test_inputs"],  te["test_targets"]

    # === Formatos esperados por data_utils.py ===
    train_val_tuple = (Xtr, ytr, Xva, yva)
    test_tuple      = (Xte, yte)

    save_gz_pickle(train_val_tuple, base_no_ext + "_train_val")
    save_gz_pickle(test_tuple,      base_no_ext + "_test")
    print(f"[OK] {base_no_ext}_train_val  y  {base_no_ext}_test  (tuplas pickle+gzip)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="p.ej. data\\F1data")
    args = ap.parse_args()
    convert_one(args.base)
