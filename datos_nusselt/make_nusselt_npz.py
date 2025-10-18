# make_nusselt_npz.py
import os, csv, numpy as np
np.random.seed(0)

ROOT = os.path.join("datos_nusselt")
FILES = {
    "train_val": ["df_n_25.txt", "df_n_53.txt"],
    "test":      ["df_n_102.txt"],
}
OUT_TV = os.path.join(ROOT, "nusselt_train_val.npz")
OUT_TE = os.path.join(ROOT, "nusselt_test.npz")

def load_txt(path):
    X, y = [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                k = float(r["K"])
            except KeyError:
                # Por si el encabezado viene en minúsculas o con espacios
                r = {k.strip().lower(): v for k, v in r.items()}
                k = float(r["k"])
            if abs(k - 0.6) > 1e-9:
                continue
            re_  = float(r.get("Rem", r.get("rem")))
            pr   = float(r.get("prandtl", r.get("Prandtl", r.get("pr"))))
            nu   = float(r.get("nusselt", r.get("Nu", r.get("nu"))))
            X.append([re_, pr])
            y.append([nu])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def join(files):
    Xs, ys = [], []
    for fn in files:
        X, y = load_txt(os.path.join(ROOT, fn))
        if len(X) == 0:
            continue
        Xs.append(X); ys.append(y)
    if not Xs:
        raise RuntimeError("No se leyeron filas con K=0.6. Revisa rutas/encabezados.")
    X = np.vstack(Xs); y = np.vstack(ys)
    return X, y

def save_train_val(X, y, out_path, val_frac=0.2):
    n = X.shape[0]
    idx = np.arange(n); np.random.shuffle(idx)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    np.savez_compressed(out_path,
        train_inputs=X[tr_idx], train_targets=y[tr_idx],
        val_inputs=X[val_idx],  val_targets=y[val_idx],
        num_inputs=2, num_outputs=1,
        train_val_examples=n
    )

def save_test(X, y, out_path):
    np.savez_compressed(out_path,
        test_inputs=X, test_targets=y,
        num_inputs=2, num_outputs=1
    )

if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    # train+val
    X_tv, y_tv = join(FILES["train_val"])
    save_train_val(X_tv, y_tv, OUT_TV, val_frac=0.2)
    # test
    X_te, y_te = join(FILES["test"])
    save_test(X_te, y_te, OUT_TE)
    print(f"OK ✅  Guardados:\n - {OUT_TV}\n - {OUT_TE}\nFilas K=0.6: train+val={len(X_tv)}, test={len(X_te)}")
