# tools/dump_preds.py
import os, sys, argparse, numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_utils import data_from_file
from evaluation import set_evaluation_hook

def pick_train_module(base_path: str):
    base = os.path.basename(base_path).lower()
    if   "f1" in base or "f2" in base or "f3" in base or "f4" in base:
        import train_F1_F4 as mod
        return mod
    elif "f5" in base or "f6" in base:
        import train_F5_F6 as mod
        return mod
    elif "f7" in base or "f8" in base:
        import train_F7_F8 as mod
        return mod
    elif "f9" in base or "f10" in base:
        import train_F9_F10 as mod
        return mod
    else:
        raise ValueError(f"No pude inferir el módulo de entrenamiento desde '{base_path}'.")

def _predict_all(estimator, X):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
        return ds.batch(max(1, min(len(X), 1024)))
    preds = list(estimator.predict(input_fn=input_fn))
    return np.vstack([np.atleast_1d(p['yhat']) for p in preds])

def save_csv(y, yhat, out_path):
    import csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y", "yhat"])
        for a, b in zip(y.reshape(-1), yhat.reshape(-1)):
            w.writerow([float(a), float(b)])
    print("→ guardado", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Carpeta del experimento (results/xxx)")
    ap.add_argument("--base", required=True, help="Base de datos, p.ej. data/F1data (sin sufijos)")
    args = ap.parse_args()

    train_val_file = args.base + "_train_val"
    test_file      = args.base + "_test"

    train_mod = pick_train_module(args.base)

    # Construir params a partir de los defaults del módulo + overrides mínimos
    runtime_params = dict(train_mod.default_params)
    runtime_params.update({
        "model_dir": args.model_dir,
        "train_val_file": train_val_file,
        "test_file": test_file,
        "generate_symbolic_expr": False,  # ¡solo predecimos!
    })

    run_config = tf.compat.v1.estimator.RunConfig(
        save_summary_steps=1000,
        keep_checkpoint_max=1,
        session_config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1)
    )
    evaluation_hook = set_evaluation_hook(**runtime_params)

    est = tf.estimator.Estimator(
        model_fn=train_mod.model_fn,
        config=run_config,
        model_dir=runtime_params["model_dir"],
        params=runtime_params
    )

    # Cargar datasets (formato repo: tuples pickle+gzip "sin extensión")
    Xtr, ytr, Xva, yva = data_from_file(train_val_file)
    if os.path.exists(test_file):
        Xte, yte = data_from_file(test_file)
    else:
        Xte = yte = np.zeros((0, ytr.shape[1]), dtype=np.float32)

    # Predecir
    print("→ Prediciendo TRAIN…")
    ytr_hat = _predict_all(est, Xtr)
    print("→ Prediciendo VAL…")
    yva_hat = _predict_all(est, Xva)
    if len(Xte) > 0:
        print("→ Prediciendo TEST…")
        yte_hat = _predict_all(est, Xte)
    else:
        yte_hat = np.zeros_like(yte)

    # Guardar CSVs en la carpeta del experimento
    outdir = runtime_params["model_dir"]
    save_csv(ytr, ytr_hat, os.path.join(outdir, "preds_train.csv"))
    save_csv(yva, yva_hat, os.path.join(outdir, "preds_val.csv"))
    if len(Xte) > 0:
        save_csv(yte, yte_hat, os.path.join(outdir, "preds_test.csv"))

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
