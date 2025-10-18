# train_nusselt.py
"""
EQL para predecir Nu(Re, Pr) usando datos reales (K=0.6).
Arquitectura con neuronas {id, multiply} y v = [2,2,1].
Incluye opción de entrenar en log-espacio para estabilizar leyes de potencia.
Genera métricas, plots de paridad y ecuación simbólica (si está activado).
"""
import os
# Ajusta las rutas si tu instalación es distinta:
miktex_bin   = r"C:\Program Files\MiKTeX\miktex\bin\x64"
graphviz_bin = r"C:\Program Files\Graphviz\bin\bin"
gs_bin       = r"C:\Program Files\gs\gs10.06.0\bin"  # o la versión que tengas

for p in (miktex_bin, graphviz_bin, gs_bin):
    if os.path.isdir(p) and p not in os.environ["PATH"]:
        os.environ["PATH"] = p + ";" + os.environ["PATH"]



import os, math, sys
import numpy as np
import tensorflow as tf
from collections import namedtuple

import EQL_Layer_tf as eql
from data_utils import get_input_fns, extract_metadata
from evaluation import set_evaluation_hook, evaluate_estimator
from analysis_utils import mse, nrmse, r2_score, parity_plot, write_json, save_summary_csv_row
from utils import step_to_epochs, get_run_config, save_results, update_runtime_params, \
                  get_div_thresh_fn, get_max_episode

# -------------------------
# Parámetros por defecto
# -------------------------
default_params = {
    "model_base_dir": "results",
    "id": 301,
    "train_val_file": "datos_nusselt/nusselt_train_val",
    "test_file":      "datos_nusselt/nusselt_test",

    # Entrenamiento / arquitectura
    "epoch_factor": 1000,
    "num_h_layers": 3,
    "layer_width": [2, 2, 1],            # v = [2,2,1]
    "generate_symbolic_expr": True,
    "kill_summaries": False,

    # Optimizador / regularización
    # (más conservadores por defecto para evitar NaNs)
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "batch_size": 128,
    "reg_sched": [0.2, 0.8],
    "reg_scale": 1e-5,
    "l0_threshold": 1e-3,
    "train_val_split": 0.8,
    "penalty_every": 50,

    # Control de rangos
    "output_bound": 1e3,                 # límite de salida para contener explosiones, antes era None pero daba Naan
    "weight_init_param": 0.2,            # init más chico = arranque estable antes era 1.0 pero daba Naan
    "test_div_threshold": 1e6,
    "network_init_seed": 123,


    "default_to_log_domain": True
}

# -------------------------
# Modelo: solo id + multiply
# -------------------------
class Model(object):
    def __init__(self, mode, layer_width, num_h_layers, reg_sched, output_bound,
                 weight_init_param, epoch_factor, batch_size, test_div_threshold,
                 reg_scale, l0_threshold, train_val_split, network_init_seed=None, **_):
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        self.train_data_size = int(train_val_split * metadata["train_val_examples"])
        widths = layer_width if isinstance(layer_width, (list, tuple)) else [layer_width] * num_h_layers
        assert len(widths) == num_h_layers, "layer_width debe tener 3 entradas (v=[2,2,1])."

        self.num_h_layers = num_h_layers
        self.layer_widths = widths
        self.weight_init_scale = weight_init_param / math.sqrt(metadata["num_inputs"] + num_h_layers)
        self.seed = network_init_seed

        self.reg_start = math.floor(num_h_layers * epoch_factor * reg_sched[0])
        self.reg_end   = math.floor(num_h_layers * epoch_factor * reg_sched[1])
        self.output_bound = output_bound or metadata["extracted_output_bound"]
        self.reg_scale = reg_scale
        self.batch_size = batch_size
        self.l0_threshold = l0_threshold

        div_thresh_fn = get_div_thresh_fn(self.is_training, self.batch_size, test_div_threshold,
                                          train_examples=self.train_data_size)
        reg_div = namedtuple("reg_div", ["repeats", "div_thresh_fn"])

        # EQL: solo id y multiply
        self.eql_layers = []
        for lw in self.layer_widths:
            self.eql_layers.append(
                eql.EQL_Layer(id=lw, multiply=lw,
                              weight_init_scale=self.weight_init_scale, seed=self.seed)
            )
        # Capa final (lineal)
        self.eql_layers.append(
            eql.EQL_Layer(reg_div=reg_div(repeats=metadata["num_outputs"], div_thresh_fn=div_thresh_fn),
                          weight_init_scale=self.weight_init_scale, seed=self.seed)
        )

    def __call__(self, inputs):
        global_step = tf.train.get_or_create_global_step()
        num_epochs = step_to_epochs(global_step, self.batch_size, self.train_data_size)

        l1_reg_sched = tf.multiply(
            tf.cast(tf.less(num_epochs, self.reg_end), tf.float32),
            tf.cast(tf.greater(num_epochs, self.reg_start), tf.float32)
        ) * self.reg_scale

        l0_threshold = tf.cond(tf.less(num_epochs, self.reg_end),
                               lambda: tf.zeros(1), lambda: self.l0_threshold)

        out = inputs
        for layer in self.eql_layers:
            out = layer(out, l1_reg_sched=l1_reg_sched, l0_threshold=l0_threshold)

        # Penalización por salirse del rango
        P_bound = (tf.abs(out) - self.output_bound) * tf.cast((tf.abs(out) > self.output_bound), tf.float32)
        tf.add_to_collection("Bound_penalties", P_bound)
        return out


# -------------------------
# model_fn (Estimator)
# -------------------------
def model_fn(features, labels, mode, params):
    # --- Log-espacio opcional ---
    use_log = params.get("default_to_log_domain", False)
    if use_log:
        eps = 1e-8
        features = tf.log(tf.maximum(features, eps))  # log(Re), log(Pr)
        if mode != tf.estimator.ModeKeys.PREDICT and labels is not None:
            labels = tf.log(tf.maximum(labels, eps))  # log(Nu)

    model = Model(mode=mode, **params)

    # Hook de evaluación necesita conocer la estructura
    evaluation_hook.init_network_structure(model, params)

    preds = model(features)  # si use_log=True, esto es y_log

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"yhat": preds})

    if mode == tf.estimator.ModeKeys.TRAIN:
        reg_losses = tf.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum([tf.reduce_mean(x) for x in reg_losses], name="reg_loss_mean_sum")
        bound_penalty = tf.reduce_sum(tf.get_collection("Bound_penalties"))
        P_theta = tf.reduce_sum(tf.get_collection("Threshold_penalties"))

        # La loss se calcula en el espacio actual (lineal o log)
        mse_loss = tf.compat.v1.losses.mean_squared_error(labels, preds)
        normal_loss = tf.compat.v1.losses.get_total_loss() + P_theta
        loss = (P_theta + bound_penalty) if penalty_flag else normal_loss

        train_accuracy = tf.identity(
            tf.compat.v1.metrics.percentage_below(values=tf.abs(labels - preds), threshold=0.02)[1],
            name="train_accuracy"
        )

        opt = tf.compat.v1.train.AdamOptimizer(params["learning_rate"], beta1=params["beta1"])
        op = opt.minimize(loss, tf.compat.v1.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=op)

    if mode == tf.estimator.ModeKeys.EVAL:
        rmse = tf.sqrt(tf.compat.v1.losses.mean_squared_error(labels, preds))
        eval_acc = tf.compat.v1.metrics.percentage_below(values=tf.abs(labels - preds), threshold=0.02)
        return tf.estimator.EstimatorSpec(mode=mode, loss=rmse, eval_metric_ops={"eval_accuracy": eval_acc})


# -------------------------
# Predicción completa
# -------------------------
def _predict_all(estimator, X):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
        return ds.batch(max(1, min(len(X), 1024)))
    preds = list(estimator.predict(input_fn=input_fn))
    return np.vstack([np.atleast_1d(p["yhat"]) for p in preds])


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    runtime_params = update_runtime_params(sys.argv, default_params)
    metadata = extract_metadata(runtime_params["train_val_file"], runtime_params["test_file"])

    run_config = get_run_config(runtime_params["kill_summaries"])
    eqlearner = tf.estimator.Estimator(
        model_fn=model_fn, config=run_config,
        model_dir=runtime_params["model_dir"], params=runtime_params
    )
    logging_hook = tf.estimator.LoggingTensorHook(tensors={"train_accuracy": "train_accuracy"}, every_n_iter=1000)
    evaluation_hook = set_evaluation_hook(**runtime_params)
    max_episode = get_max_episode(**runtime_params)

    train_input, penalty_train_input, val_input, test_input = get_input_fns(**runtime_params, **metadata)

    print("One train episode equals %d normal epochs and 1 penalty epoch." % runtime_params["penalty_every"])
    for ep in range(1, max_episode + 1):
        print("Train episode: %d / %d" % (ep, max_episode))
        penalty_flag = True
        eqlearner.train(input_fn=penalty_train_input)
        penalty_flag = False
        eqlearner.train(input_fn=train_input, hooks=[logging_hook])

    print("✅ Training complete. Starting evaluation...")

    # ==== Evaluación ====
    tv = np.load(runtime_params["train_val_file"] + ".npz")
    Xva, yva = tv["val_inputs"], tv["val_targets"]
    n_val = int(Xva.shape[0]); bs = int(runtime_params["batch_size"])
    steps_val = max(1, int(np.ceil(n_val / float(bs))))
    print("📊 Evaluando validación...")
    val_res = eqlearner.evaluate(input_fn=val_input, name="validation", hooks=[evaluation_hook], steps=steps_val)

    results = dict(val_error=float(val_res["loss"]), complexity=int(evaluation_hook.get_complexity()))

    extr = None
    if test_input is not None and os.path.exists(runtime_params["test_file"] + ".npz"):
        te = np.load(runtime_params["test_file"] + ".npz")
        if te["test_inputs"].shape[0] > 0:
            print("📊 Evaluando test...")
            extr = evaluate_estimator(eqlearner, runtime_params["test_file"], batch_size=bs,
                                      logger=tf.compat.v1.logging)
            results["extr_error"] = float(extr["loss"])

    print("📈 Generando métricas y gráficos...")
    Xtr, ytr = tv["train_inputs"], tv["train_targets"]
    ytr_hat = _predict_all(eqlearner, Xtr)
    yva_hat = _predict_all(eqlearner, Xva)

    # Si entrenamos en log-espacio, convertimos predicciones a escala original para los plots
    if runtime_params.get("default_to_log_domain", False):
        ytr_hat = np.exp(ytr_hat)
        yva_hat = np.exp(yva_hat)

    # === Guardado y gráficos ===
    outdir = runtime_params["model_dir"]
    os.makedirs(outdir, exist_ok=True)

    parity_plot(ytr, ytr_hat, os.path.join(outdir, "parity_train.png"), "Train: y vs ŷ")
    parity_plot(yva, yva_hat, os.path.join(outdir, "parity_val.png"),   "Val: y vs ŷ")

    print("🧠 Intentando generar ecuaciones simbólicas (si está activado)...")
    if runtime_params.get("generate_symbolic_expr", False):
        print("   ⏳ Generando archivos LaTeX/Graphviz (puede tardar unos minutos)...")

    # === Gráfico adicional: Nu vs Re (Pr fijo) ===
    import matplotlib.pyplot as plt
    Pr_mean = float(np.mean(Xtr[:, 1]))
    Re_vals = np.linspace(np.min(Xtr[:, 0]), np.max(Xtr[:, 0]), 200)

    if runtime_params.get("default_to_log_domain", False):
        # El modelo espera log-features si use_log=True
        X_plot = np.stack([np.log(Re_vals), np.full_like(Re_vals, np.log(Pr_mean))], axis=1)
        y_pred_plot = np.exp(_predict_all(eqlearner, X_plot).flatten())  # vuelvo a escala original
    else:
        X_plot = np.stack([Re_vals, np.full_like(Re_vals, Pr_mean)], axis=1)
        y_pred_plot = _predict_all(eqlearner, X_plot).flatten()

    plt.figure(figsize=(6, 4))
    plt.plot(Re_vals, y_pred_plot, 'r-', lw=2, label=f'Pr fijo = {Pr_mean:.3f}')
    plt.xlabel('Número de Reynolds (Re)')
    plt.ylabel('Número de Nusselt (Nu predicho)')
    plt.title('Predicción EQL: Nu vs Re (Pr constante)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_graph = os.path.join(outdir, "Nu_vs_Re_fixedPr.png")
    plt.savefig(out_graph, dpi=150)
    plt.close()
    print(f"📊 Guardado gráfico: {out_graph}")

    # === Resumen en JSON + CSV (opcional) ===
    metrics = {
        "train": {"MSE": float(mse(ytr, ytr_hat)), "NRMSE": float(nrmse(ytr, ytr_hat)), "R2": float(r2_score(ytr, ytr_hat))},
        "val":   {"MSE": float(mse(yva, yva_hat)), "NRMSE": float(nrmse(yva, yva_hat)), "R2": float(r2_score(yva, yva_hat))},
    }
    results.update({"metrics": metrics})
    results["equation"] = "Revisa latex_y*.png / graph_y*.png del EvaluationHook en la carpeta del experimento."

    write_json(os.path.join(outdir, "analysis_summary.json"), results)

    csv_row = {
        "model_dir": outdir,
        "val_RMSE": float(val_res["loss"]),
        "extr_RMSE": float(results.get("extr_error", np.nan)),
        "train_MSE": metrics["train"]["MSE"], "train_R2": metrics["train"]["R2"],
        "val_MSE": metrics["val"]["MSE"],     "val_R2": metrics["val"]["R2"],
    }
    save_summary_csv_row(os.path.join(outdir, "..", "all_runs_summary.csv"),
                         csv_row, header_order=list(csv_row.keys()))

    save_results({"val_error": float(val_res["loss"]),
                  "extr_error": float(results.get("extr_error", np.nan)),
                  "complexity": int(results.get("complexity", np.nan))},
                 runtime_params)

    print("🎉 Model evaluated successfully. Results saved in", outdir)
