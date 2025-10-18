# train_F9_F10.py
"""EQL - entrenamiento F9..F10 (sin, cos, id, multiply – sin sigmoid), capas v=[3,3,3]."""
import os, math, sys
from collections import namedtuple
import numpy as np
import tensorflow as tf

import EQL_Layer_tf as eql
from data_utils import get_input_fns, extract_metadata
from evaluation import set_evaluation_hook, evaluate_estimator
from utils import step_to_epochs, get_run_config, save_results, update_runtime_params, \
                  get_div_thresh_fn, get_max_episode
from analysis_utils import mse, nrmse, r2_score, parity_plot, write_json, save_summary_csv_row

# =========================
#   Parámetros por defecto
# =========================
default_params = {
    'model_base_dir': 'results',
    'id': 401,
    'train_val_file': 'data/F9data_train_val',   # pasa F10 por CLI si corresponde
    'test_file':      'data/F9data_test',
    'epoch_factor': 1000,
    'num_h_layers': 3,
    'layer_width': [3, 3, 3],   # v = [3,3,3]
    'generate_symbolic_expr': False,  # para evitar cuelgues; cámbialo a True si ya tienes LaTeX/Graphviz ok
    'kill_summaries': False,
}

# =========================
#   Modelo
# =========================
class Model(object):
    def __init__(self, mode, layer_width, num_h_layers, reg_sched, output_bound,
                 weight_init_param, epoch_factor, batch_size, test_div_threshold,
                 reg_scale, l0_threshold, train_val_split, network_init_seed=None, **_):
        # metadata inyectada en __main__
        self.train_data_size = int(train_val_split * metadata['train_val_examples'])
        self.layer_widths = layer_width if isinstance(layer_width, (list, tuple)) else [layer_width]*num_h_layers
        assert len(self.layer_widths) == num_h_layers, "layer_width debe tener 3 entradas."
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

        # ---- sin, cos, id, multiply; v=[3,3,3] ----
        self.eql_layers = []
        for lw in self.layer_widths:
            self.eql_layers.append(
                eql.EQL_Layer(sin=lw, cos=lw, id=lw, multiply=lw,
                              weight_init_scale=self.weight_init_scale, seed=self.seed)
            )
        # Capa final (lineal) con manejo de penalizaciones/div-threshold
        self.eql_layers.append(
            eql.EQL_Layer(reg_div=reg_div(repeats=metadata['num_outputs'], div_thresh_fn=div_thresh_fn),
                          weight_init_scale=self.weight_init_scale, seed=self.seed)
        )

    def __call__(self, inputs):
        global_step = tf.train.get_or_create_global_step()
        num_epochs = step_to_epochs(global_step, self.batch_size, self.train_data_size)

        l1_reg_sched = (tf.cast(tf.less(num_epochs, self.reg_end), tf.float32) *
                        tf.cast(tf.greater(num_epochs, self.reg_start), tf.float32)) * self.reg_scale
        l0_threshold = tf.cond(tf.less(num_epochs, self.reg_end),
                               lambda: tf.zeros(1), lambda: self.l0_threshold)

        out = inputs
        for layer in self.eql_layers:
            out = layer(out, l1_reg_sched=l1_reg_sched, l0_threshold=l0_threshold)

        # Penalización por rango de salida
        P_bound = (tf.abs(out) - self.output_bound) * tf.cast((tf.abs(out) > self.output_bound), tf.float32)
        tf.add_to_collection('Bound_penalties', P_bound)
        return out

# =========================
#   model_fn (Estimator)
# =========================
def model_fn(features, labels, mode, params):
    model = Model(mode=mode, **params)
    evaluation_hook.init_network_structure(model, params)
    preds = model(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'yhat': preds})

    if mode == tf.estimator.ModeKeys.TRAIN:
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum([tf.reduce_mean(x) for x in reg_losses], name='reg_loss_mean_sum')
        bound_penalty = tf.reduce_sum(tf.get_collection('Bound_penalties'))
        P_theta = tf.reduce_sum(tf.get_collection('Threshold_penalties'))

        mse_loss = tf.losses.mean_squared_error(labels, preds)
        normal_loss = tf.losses.get_total_loss() + P_theta
        # penalty_flag se setea en el bucle de entrenamiento (global)
        loss = (P_theta + bound_penalty) if penalty_flag else normal_loss

        train_acc = tf.identity(tf.metrics.percentage_below(values=tf.abs(labels - preds), threshold=0.02)[1],
                                name='train_accuracy')
        tf.summary.scalar('total_loss', loss, family='losses')
        tf.summary.scalar('MSE_loss', mse_loss, family='losses')
        tf.summary.scalar('Penalty_Loss', P_theta + bound_penalty, family='losses')
        tf.summary.scalar('Regularization_loss', reg_loss, family='losses')
        tf.summary.scalar('train_acc', train_acc, family='accuracies')

        opt = tf.train.AdamOptimizer(params['learning_rate'], beta1=params['beta1']).minimize(
            loss, tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=opt)

    if mode == tf.estimator.ModeKeys.EVAL:
        rmse = tf.sqrt(tf.losses.mean_squared_error(labels, preds))
        eval_acc = tf.metrics.percentage_below(values=tf.abs(labels - preds), threshold=0.02)
        return tf.estimator.EstimatorSpec(mode=mode, loss=rmse, eval_metric_ops={'eval_accuracy': eval_acc})

# =========================
#   Utilidad: predicción en bloque
# =========================
def _predict_all(estimator, X):
    def input_fn():
        return tf.data.Dataset.from_tensor_slices(X.astype(np.float32)).batch(max(1, min(len(X), 1024)))
    return np.vstack([np.atleast_1d(p['yhat']) for p in estimator.predict(input_fn=input_fn)])

# =========================
#   Main
# =========================
if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    runtime_params = update_runtime_params(sys.argv, default_params)
    metadata = extract_metadata(runtime_params['train_val_file'], runtime_params['test_file'])

    run_config = get_run_config(runtime_params['kill_summaries'])
    eqlearner = tf.estimator.Estimator(model_fn=model_fn, config=run_config,
                                       model_dir=runtime_params['model_dir'], params=runtime_params)
    logging_hook = tf.train.LoggingTensorHook(tensors={'train_accuracy': 'train_accuracy'}, every_n_iter=1000)
    evaluation_hook = set_evaluation_hook(**runtime_params)
    max_episode = get_max_episode(**runtime_params)

    train_input, penalty_train_input, val_input, test_input = get_input_fns(**runtime_params, **metadata)

    print('One train episode equals %d normal epochs and 1 penalty epoch.' % runtime_params['penalty_every'])
    for ep in range(1, max_episode + 1):
        print('Train episode: %d / %d' % (ep, max_episode))
        penalty_flag = True;  eqlearner.train(input_fn=penalty_train_input)              # 1 pasada de “penalty”
        penalty_flag = False; eqlearner.train(input_fn=train_input, hooks=[logging_hook])# epochs normales

    print('Training complete. Evaluating...')
    # === Validación finita (a partir del .npz para pasos finitos) ===
    tv = np.load(runtime_params['train_val_file'] + ".npz")
    Xva, yva = tv['val_inputs'], tv['val_targets']
    steps_val = max(1, int(np.ceil(Xva.shape[0] / float(int(runtime_params['batch_size'])))))
    val_res = eqlearner.evaluate(input_fn=val_input, name='validation', hooks=[evaluation_hook], steps=steps_val)

    results = dict(val_error=float(val_res['loss']), complexity=int(evaluation_hook.get_complexity()))

    # === Test (si existe) usando evaluate_estimator (pasos finitos) ===
    if test_input is not None and os.path.exists(runtime_params['test_file'] + ".npz"):
        extr = evaluate_estimator(eqlearner, runtime_params['test_file'],
                                  batch_size=int(runtime_params['batch_size']),
                                  logger=tf.compat.v1.logging)
        results['extr_error'] = float(extr['loss'])

    # === Métricas + parity plots ===
    Xtr, ytr = tv['train_inputs'], tv['train_targets']
    ytr_hat = _predict_all(eqlearner, Xtr)
    yva_hat = _predict_all(eqlearner, Xva)

    te = np.load(runtime_params['test_file'] + ".npz") if os.path.exists(runtime_params['test_file'] + ".npz") else None
    if te is not None and 'test_inputs' in te:
        Xte, yte = te['test_inputs'], te['test_targets']
        yte_hat = _predict_all(eqlearner, Xte)
    else:
        Xte = yte = yte_hat = np.zeros((0, ytr.shape[1]))

    metrics = {
        "train": {"MSE": float(mse(ytr, ytr_hat)), "NRMSE": float(nrmse(ytr, ytr_hat)), "R2": float(r2_score(ytr, ytr_hat))},
        "val":   {"MSE": float(mse(yva, yva_hat)), "NRMSE": float(nrmse(yva, yva_hat)), "R2": float(r2_score(yva, yva_hat))},
        "test":  ({"MSE": float(mse(yte, yte_hat)), "NRMSE": float(nrmse(yte, yte_hat)), "R2": float(r2_score(yte, yte_hat))}
                  if Xte.shape[0] > 0 else {})
    }
    results.update({"metrics": metrics})
    results["equation"] = "Ver latex_y*.png / graph_y*.png en la carpeta del experimento (si generate_symbolic_expr=True)."

    outdir = runtime_params['model_dir']
    write_json(os.path.join(outdir, "analysis_summary.json"), results)
    parity_plot(ytr, ytr_hat, os.path.join(outdir, "parity_train.png"), "Train: y vs ŷ")
    parity_plot(yva, yva_hat, os.path.join(outdir, "parity_val.png"),   "Val: y vs ŷ")
    if Xte.shape[0] > 0:
        parity_plot(yte, yte_hat, os.path.join(outdir, "parity_test.png"), "Test: y vs ŷ")

    # CSV resumen
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
                         csv_row, header_order=list(csv_row.keys()))
    save_results({"val_error": float(val_res['loss']),
                  "extr_error": float(results.get('extr_error', np.nan)),
                  "complexity": int(results['complexity'])}, runtime_params)
    print('Model evaluated. Results:\n', results)
