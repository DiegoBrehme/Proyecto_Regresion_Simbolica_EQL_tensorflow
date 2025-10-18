# tools/dump_preds_fixed.py
import os, sys, argparse, importlib, json, glob, numpy as np
import tensorflow as tf

from data_utils import extract_metadata
from evaluation import set_evaluation_hook
from utils import get_run_config

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

LIKELY_PARAM_FILES = [
    "runtime_params.json", "run_params.json", "params.json",
    "results.json", "analysis_summary.json"
]

def read_ckpt_shapes(model_dir):
    reader = tf.compat.v1.train.NewCheckpointReader(tf.train.latest_checkpoint(model_dir))
    var_to_shape = reader.get_variable_to_shape_map()
    # Filtramos kernels/bias para inspección
    k = {n: s for n, s in var_to_shape.items() if "kernel" in n}
    b = {n: s for n, s in var_to_shape.items() if "bias" in n}
    return k, b

def try_load_params_from_dir(model_dir):
    for name in LIKELY_PARAM_FILES:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    js = json.load(f)
                # algunos archivos (analysis_summary.json) no tienen params directos, los ignoramos
                if isinstance(js, dict) and any(k in js for k in [
                    "num_h_layers", "layer_width", "train_val_file", "test_file", "learning_rate"
                ]):
                    print(f"✓ Usando params desde {name}")
                    return js
            except Exception:
                pass
    return {}

def merge_params(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = v
    return out

def load_split_npz(base_no_ext):
    out = {}
    tv = base_no_ext + "_train_val.npz"
    if os.path.isfile(tv):
        tvnpz = np.load(tv)
        out["train"] = (tvnpz["train_inputs"], tvnpz["train_targets"])
        out["val"]   = (tvnpz["val_inputs"],   tvnpz["val_targets"])
    te = base_no_ext + "_test.npz"
    if os.path.isfile(te):
        tenpz = np.load(te)
        out["test"] = (tenpz["test_inputs"], tenpz["test_targets"])
    return out

def predict_block(estimator, X):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
        return ds.batch(max(1, min(len(X), 4096)))
    preds = list(estimator.predict(input_fn=input_fn))
    return np.vstack([np.atleast_1d(p["yhat"]) for p in preds])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--base", required=True, help="p.ej. data/F1data (sin sufijos)")
    ap.add_argument("--script", required=True, help="módulo de entrenamiento: train_F1_F4, train_F5_F6, etc.")
    ap.add_argument("--params_json", default="", help="ruta a JSON con params originales (opcional)")
    ap.add_argument("--override", default="", help="dict en JSON para overrides (opcional)")
    args = ap.parse_args()

    model_dir = args.model_dir.rstrip("/\\")
    base_no_ext = args.base

    # 1) Importa módulo entrenamiento
    m = importlib.import_module(args.script)

    # 2) Parte de defaults del módulo
    params = dict(getattr(m, "default_params", {}))

    # 3) Intenta leer params reales desde la carpeta del run
    found = try_load_params_from_dir(model_dir)
    params = merge_params(params, found)

    # 4) Si pasan un JSON explícito, usa ese
    if args.params_json:
        with open(args.params_json, "r", encoding="utf-8") as f:
            pj = json.load(f)
        params = merge_params(params, pj)

    # 5) Overrides manuales por CLI (cadena JSON)
    if args.override:
        ov = json.loads(args.override)
        params = merge_params(params, ov)

    # 6) Ajusta rutas de datos a formato base + sufijo sin extensión (.npz existe aparte)
    params["model_dir"]      = model_dir
    params["model_base_dir"] = os.path.dirname(model_dir) or "results"
    params["train_val_file"] = base_no_ext + "_train_val"
    params["test_file"]      = base_no_ext + "_test"

    # Asegura hyperparámetros mínimos si no están
    params.setdefault("learning_rate", 1e-3)
    params.setdefault("beta1", 0.9)
    params.setdefault("batch_size", 128)
    params.setdefault("reg_sched", [0.2, 0.8])
    params.setdefault("reg_scale", 1e-4)
    params.setdefault("l0_threshold", 1e-3)
    params.setdefault("train_val_split", 0.8)
    params.setdefault("penalty_every", 50)
    params.setdefault("output_bound", None)
    params.setdefault("weight_init_param", 1.0)
    params.setdefault("test_div_threshold", 1e6)
    params.setdefault("network_init_seed", 123)

    # 7) Muestra info útil
    print("\n=== PARAMS EFECTIVOS PARA RECONSTRUIR EL GRAFO ===")
    for k in ["num_h_layers", "layer_width", "train_val_file", "test_file"]:
        if k in params: print(f"{k}: {params[k]}")
    print("==================================================\n")

    # 8) Inspecciona shapes del checkpoint (guía para detectar mismatch)
    try:
        ksh, bsh = read_ckpt_shapes(model_dir)
        if ksh:
            print("Shapes en checkpoint (kernels):")
            for n, s in sorted(ksh.items()):
                print("  ", n, s)
        if bsh:
            print("Shapes en checkpoint (biases):")
            for n, s in sorted(bsh.items()):
                print("  ", n, s)
        print("")
    except Exception as e:
        print(f"⚠ No se pudo leer shapes del checkpoint: {e}")

    # 9) Inyecta globales esperadas por tus train_*.py
    metadata = extract_metadata(params["train_val_file"], params["test_file"])
    setattr(m, "metadata", metadata)
    evaluation_hook = set_evaluation_hook(**params)
    setattr(m, "evaluation_hook", evaluation_hook)
    setattr(m, "penalty_flag", False)

    # 10) Construye Estimator con esos params
    run_config = get_run_config(kill_summaries=True)
    est = tf.estimator.Estimator(model_fn=m.model_fn, config=run_config,
                                 model_dir=model_dir, params=params)

    # 11) Carga splits y predice
    splits = load_split_npz(base_no_ext)
    if not splits:
        print(f"⚠ No encontré {base_no_ext}_train_val.npz ni {base_no_ext}_test.npz")
        sys.exit(1)

    os.makedirs(model_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        if split not in splits: 
            print(f"→ {split} no disponible; omito.")
            continue
        X, y = splits[split]
        print(f"→ Prediciendo {split.upper()}… ({X.shape[0]} ejemplos)")
        yhat = predict_block(est, X)
        arr = np.hstack([X, y, yhat])
        out = os.path.join(model_dir, f"preds_{split}.npy")
        np.save(out, arr)
        print("   Guardado:", out)

    print("\n✔ dump_preds_fixed: terminado OK.")

if __name__ == "__main__":
    main()
