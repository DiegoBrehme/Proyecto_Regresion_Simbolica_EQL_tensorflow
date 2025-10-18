# train_F5_F6.py
"""EQL - entrenamiento F5..F6 (sin, cos, id, multiply), v=[3,3,2] + evaluación finita + prints de humo."""
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

# =========================
#   Parámetros por defecto
# =========================
default_params = {
    'model_base_dir': 'results',
    'id': 205,
    'train_val_file': 'data/F5data_train_val',
    'test_file':      'data/F5data_test',
    'epoch_factor': 1000,
    'num_h_layers': 3,
    'layer_width': [3, 3, 2],   # v=[3,3,2] pensado para F5-F6
    'generate_symbolic_expr': False,  # True si ya tienes LaTeX/Graphviz OK
    'kill_summaries': False,

    # Optimizador / regularización (igual que en tus otros scripts)
    'learning_rate': 1e-3,
    'beta1': 0.9,
    'batch_size': 128,
    'reg_sched': [0.2, 0.8],
    'reg_scale': 1e-4,
    'l0_threshold': 1e-3,
    'train_val_split': 0.8,
    'penalty_every': 50,
    'output_bound': None,
    'weight_init_param': 1.0,
    'test_div_threshold': 1e6,
    'network_init_seed': 123,
}

# =========================
#   Modelo
# =========================
class Model(object):
    def __init__(self, mode, layer_width, num_h_layers, reg_sched, output_bound,
                 weight_init_param, epoch_factor, batch_size, test_div_threshold,
                 reg_scale, l0_threshold, train_val_split, network_init_seed=None, **_):
        # metadata es global; la define extract_metadata en main
        self.train_data_size = int(train_val_split * metadata['train_val_examples'])
        self.layer_widths = layer_width if isinstance(layer_width, (list, tuple)) else [layer_width] * num_h_layers
        assert len(self.layer_widths) == num_h_layers, "layer_width debe tener tantas entradas como capas ocultas."
        self.num_h_layers = num_h_layers
        self.weight_init_scale = weight_init_param / math.sqrt(metadata['num_inputs'] + num_h_layers)
        self.seed = network_init_seed
        self.reg_start = math.floor(num_h_layers * epoch_factor * reg_sched[0])
        self.reg_end   = math.floor(num_h_layers * epoch_factor * reg_sched[1])
        self.output_bound = output_bound or metadata['extracted_output_bound']
        self.reg_scale = reg_scale
        self.batch_size = batch_size
        self.l0_threshold = l0_threshold
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        div_thresh_fn = get_div_thresh_fn(self.is_training, self.batch_size, test_div_threshold,
                                          train_examples=self.train_data_size)
        reg_div = namedtuple('reg_div', ['repeats', 'div_thresh_fn'])

        # ---- sin, cos, id, multiply; sin división ni sigmoide ----
        self.eql_layers = []
        for lw in self.layer_widths:
            self.eql_layers.append(
                eql.EQL_Layer(sin=lw, cos=lw, id=lw, multiply=lw,
                              weight_init_scale=self.weight_init_scale, seed=self.seed)
            )
        # Capa final lineal con manejo de penalizaciones/div threshold por salida
        self.eql_layers.append(
            eql.EQL_Layer(reg_div=reg_div(repeats=metadata['num_outputs'], div_thresh_fn=div_thresh_fn),
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

        output = inputs
        for layer in self.eql_layers:
            output = layer(output, l1_reg_sched=l1_reg_sched, l0_threshold=l0_threshold)

        # Penalización por salirse del rango de salida
        P_bound = (tf.abs(output) - self.output_bound) * tf.cast((tf.abs(output) > self.output_bound), dtype=tf.float32)
        tf.add_to_collection('Bound_penalties', P_bound)
        return output

# =========================
#   model_fn (Estimator)
# =========================
def model_fn(features, labels, mode, params):
    # evaluation_hook es global; se inicializa en __main__
    model = Model(mode=mode, **params)
    evaluation_hook.init_network_structure(model, params)
    predictions = model(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions={'yhat': predictions}
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum([tf.reduce_mean(x) for x in reg_losses], name='reg_loss_mean_sum')
        bound_penalty = tf.reduce_sum(tf.get_collection('Bound_penalties'))
        P_theta = tf.reduce_sum(tf.get_collection('Threshold_penalties'))

        mse_loss = tf.losses.mean_squared_error(labels, predictions)
        normal_loss = tf.losses.get_total_loss() + P_theta
        # penalty_flag es global; se cambia en el loop de entrenamiento
        loss = (P_theta + bound_penalty) if penalty_flag else normal_loss

        train_accuracy = tf.identity(
            tf.metrics.percentage_below(values=tf.abs(labels - predictions), threshold=0.02)[1],
            name='train_accuracy'
        )
        tf.summary.scalar('total_loss', loss, family='losses')
        tf.summary.scalar('MSE_loss', mse_loss, family='losses')
        tf.summary.scalar('Penalty_Loss', P_theta + bound_penalty, family='losses')
        tf.summary.scalar('Regularization_loss', reg_loss, family='losses')
        tf.summary.scalar('train_acc', train_accuracy, family='accuracies')

        op = tf.train.AdamOptimizer(params['learning_rate'], beta1=params['beta1']).minimize(
            loss, tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=op)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.sqrt(tf.losses.mean_squared_error(labels, predictions))
        eval_acc = tf.metrics.percentage_below(values=tf.abs(labels - predictions), threshold=0.02)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={'eval_accuracy': eval_acc}
        )

# =========================
#   Predicción utilitaria
# =========================
def _predict_all(estimator, X):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
        ds = ds.batch(max(1, min(len(X), 1024)))
        return ds
    preds = list(estimator.predict(input_fn=input_fn))
    yh = np.vstack([np.atleast_1d(p['yhat']) for p in preds])
    return yh

# =========================
#   Main
# =========================
if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    # (opcional) log a archivo para revisar si se cuelga en algún punto:
    # sys.stdout = open("train_F5_F6_log.txt", "w")

    # Parámetros y metadata
    runtime_params = update_runtime_params(sys.argv, default_params)
    metadata = extract_metadata(runtime_params['train_val_file'], runtime_params['test_file'])

    # Estimator + hooks
    run_config = get_run_config(runtime_params['kill_summaries'])
    eqlearner = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        model_dir=runtime_params['model_dir'],
        params=runtime_params
    )
    logging_hook = tf.train.LoggingTensorHook(tensors={'train_accuracy': 'train_accuracy'}, every_n_iter=1000)
    evaluation_hook = set_evaluation_hook(**runtime_params)
    max_episode = get_max_episode(**runtime_params)

    # Input fns
    train_input, penalty_train_input, val_input, test_input = get_input_fns(**runtime_params, **metadata)

    print('One train episode equals %d normal epochs and 1 penalty epoch.' % runtime_params['penalty_every'])
    for ep in range(1, max_episode + 1):
        print('Train episode: %d / %d' % (ep, max_episode))
        penalty_flag = True
        eqlearner.train(input_fn=penalty_train_input)                     # penalty epoch (1 pasada)
        penalty_flag = False
        eqlearner.train(input_fn=train_input, hooks=[logging_hook])       # normal epochs (penalty_every pasadas)

    print('Training complete. Evaluating...')

    # =========================
    #   Evaluación FINITA + PRINTS DE HUMO
    # =========================
    print(">> Eval de validación iniciando...")
    tv = np.load(runtime_params['train_val_file'] + ".npz")
    Xva, yva = tv['val_inputs'], tv['val_targets']
    n_val = int(Xva.shape[0])
    bs = int(runtime_params['batch_size'])
    steps_val = max(1, int(np.ceil(n_val / float(bs))))
    val_res = eqlearner.evaluate(input_fn=val_input, name='validation', hooks=[evaluation_hook], steps=steps_val)
    print(">> Eval de validación terminada.")

    results = dict(val_error=float(val_res['loss']), complexity=int(evaluation_hook.get_complexity()))

    print(">> Eval de test iniciando...")
    extr = None
    if test_input is not None and os.path.exists(runtime_params['test_file'] + ".npz"):
        te = np.load(runtime_params['test_file'] + ".npz")
        if te['test_inputs'].shape[0] > 0:
            extr = evaluate_estimator(eqlearner, runtime_params['test_file'], batch_size=bs,
                                      logger=tf.compat.v1.logging)
            results['extr_error'] = float(extr['loss'])
    print(">> Eval de test terminada.")

    print(">> Calculando métricas y generando plots...")
    # ========= Análisis adicional (métricas + plots + ecuación) =========
    Xtr, ytr = tv['train_inputs'], tv['train_targets']
    # (Xva,yva) ya cargados; Xte,yte si existe
    if test_input is not None and os.path.exists(runtime_params['test_file'] + ".npz"):
        Xte, yte = te['test_inputs'], te['test_targets']
    else:
        Xte, yte = np.zeros((0, Xtr.shape[1])), np.zeros((0, ytr.shape[1]))

    ytr_hat = _predict_all(eqlearner, Xtr)
    yva_hat = _predict_all(eqlearner, Xva)
    yte_hat = _predict_all(eqlearner, Xte) if len(Xte) > 0 else np.zeros_like(yte)

    metrics = {
        "train": {"MSE": float(mse(ytr, ytr_hat)), "NRMSE": float(nrmse(ytr, ytr_hat)), "R2": float(r2_score(ytr, ytr_hat))},
        "val":   {"MSE": float(mse(yva, yva_hat)), "NRMSE": float(nrmse(yva, yva_hat)), "R2": float(r2_score(yva, yva_hat))},
        "test":  ({"MSE": float(mse(yte, yte_hat)), "NRMSE": float(nrmse(yte, yte_hat)), "R2": float(r2_score(yte, yte_hat))}
                  if len(Xte) > 0 else {})
    }
    results.update({"metrics": metrics})

    # (El hook guarda PNGs/LaTeX si generate_symbolic_expr=True)
    results["equation"] = "Revisa los archivos (latex_y*.png / graph_y*.png) generados por EvaluationHook (si está activado)."

    print(">> Guardando resultados finales...")
    outdir = runtime_params['model_dir']
    os.makedirs(outdir, exist_ok=True)
    write_json(os.path.join(outdir, "analysis_summary.json"), results)
    parity_plot(ytr, ytr_hat, os.path.join(outdir, "parity_train.png"), "Train: y vs ŷ")
    parity_plot(yva, yva_hat, os.path.join(outdir, "parity_val.png"),   "Val: y vs ŷ")
    if len(Xte) > 0:
        parity_plot(yte, yte_hat, os.path.join(outdir, "parity_test.png"), "Test: y vs ŷ")

    csv_row = {
        "model_dir": outdir,
        "val_RMSE": float(val_res['loss']),
        "extr_RMSE": float(results.get('extr_error', np.nan)),
        "train_MSE": metrics["train"]["MSE"], "train_R2": metrics["train"]["R2"],
        "val_MSE": metrics["val"]["MSE"],     "val_R2": metrics["val"]["R2"],
        "test_MSE": metrics.get("test", {}).get("MSE", np.nan),
        "test_R2":  metrics.get("test", {}).get("R2",  np.nan),
    }
    save_summary_csv_row(os.path.join(outdir, "..", "all_runs_summary.csv"),
                         csv_row,
                         header_order=list(csv_row.keys()))

    save_results({"val_error": float(val_res['loss']),
                  "extr_error": float(results.get('extr_error', np.nan)),
                  "complexity": int(results['complexity'])},
                 runtime_params)

    print("✅ Script finalizado correctamente.")
