# train_F1_F4.py
# EQL - entrenamiento F1..F4 SOLO con {id, multiply}, ancho v=[2,3,3] + análisis/plots (eval finita)
# Nota: aquí NO uso sen/cos/sigmoid ni división. La última capa es un "readout" lineal hecho con id.

import os, math, sys
import numpy as np
import tensorflow as tf

import EQL_Layer_tf as eql
from data_utils import get_input_fns, extract_metadata
from evaluation import set_evaluation_hook, evaluate_estimator
from utils import step_to_epochs, get_run_config, save_results, update_runtime_params, get_max_episode
from analysis_utils import mse, nrmse, r2_score, parity_plot, write_json, save_summary_csv_row

# =========================
#   Parámetros por defecto
# =========================
default_params = {
    'model_base_dir': 'results',
    'id': 1,
    # por defecto corro F1; desde CLI puedo pasar F2/F3/F4
    'train_val_file': 'data/F1data_train_val',
    'test_file':      'data/F1data_test',
    'epoch_factor': 1000,
    'num_h_layers': 3,           # voy con 3 capas ocultas
    'layer_width': [2, 3, 3],    # v = [2,3,3] (esto me da margen para construir hasta x^6)
    'generate_symbolic_expr': True,
    'kill_summaries': False,
}

# =========================
#   Modelo (solo id y multiply)
# =========================
class Model(object):
    def __init__(self, mode, layer_width, num_h_layers, reg_sched, output_bound,
                 weight_init_param, epoch_factor, batch_size,
                 reg_scale, l0_threshold, train_val_split, network_init_seed=None, **_):
        # metadata llega desde __main__ (ya cargada)
        self.train_data_size = int(train_val_split * metadata['train_val_examples'])
        self.layer_widths = layer_width if isinstance(layer_width, (list, tuple)) else [layer_width] * num_h_layers
        assert len(self.layer_widths) == num_h_layers, "layer_width debe tener 3 entradas (v=[2,3,3])."

        self.num_h_layers = num_h_layers
        # normalizo la escala inicial con algo simple (me ha funcionado estable así)
        self.weight_init_scale = weight_init_param / math.sqrt(metadata['num_inputs'] + num_h_layers)
        self.seed = network_init_seed

        # programo cuándo prendo el L1 (al medio del entrenamiento básicamente)
        self.reg_start = math.floor(num_h_layers * epoch_factor * reg_sched[0])
        self.reg_end   = math.floor(num_h_layers * epoch_factor * reg_sched[1])

        self.output_bound = output_bound or metadata['extracted_output_bound']
        self.reg_scale = reg_scale
        self.batch_size = batch_size
        self.l0_threshold = l0_threshold
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # ---- SOLO id y multiply en capas ocultas ----
        self.eql_layers = []
        for lw in self.layer_widths:
            # cada capa mezcla linealmente -> aplica bloques {id, multiply}
            self.eql_layers.append(
                eql.EQL_Layer(id=lw, multiply=lw,
                              weight_init_scale=self.weight_init_scale, seed=self.seed)
            )

        # readout final: otra EQL_Layer solo con 'id' para hacer combinación lineal hacia #salidas
        # (prefiero esto a una capa "div" o cosas raras; mantengo todo consistente)
        self.eql_layers.append(
            eql.EQL_Layer(id=metadata['num_outputs'],
                          weight_init_scale=self.weight_init_scale, seed=self.seed)
        )

    def __call__(self, inputs):
        global_step = tf.train.get_or_create_global_step()
        num_epochs = step_to_epochs(global_step, self.batch_size, self.train_data_size)

        # enciendo L1 entre reg_start y reg_end
        l1_reg_sched = tf.multiply(
            tf.cast(tf.less(num_epochs, self.reg_end), tf.float32),
            tf.cast(tf.greater(num_epochs, self.reg_start), tf.float32)
        ) * self.reg_scale

        # L0-threshold lo dejo en 0 hasta terminar la etapa con L1 (esto evita “apagar” muy temprano)
        l0_threshold = tf.cond(tf.less(num_epochs, self.reg_end),
                               lambda: tf.zeros(1), lambda: self.l0_threshold)

        # paso hacia delante por todas las capas EQL
        output = inputs
        for layer in self.eql_layers:
            output = layer(output, l1_reg_sched=l1_reg_sched, l0_threshold=l0_threshold)

        # penalización suave si me salgo del rango esperado de salida (esto me estabiliza bastante)
        P_bound = (tf.abs(output) - self.output_bound) * tf.cast((tf.abs(output) > self.output_bound), dtype=tf.float32)
        tf.add_to_collection('Bound_penalties', P_bound)
        return output


# =========================
#   model_fn (Estimator)
# =========================
def model_fn(features, labels, mode, params):
    # evaluation_hook es global; lo inicializo acá con la estructura real del modelo
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
        P_theta = tf.reduce_sum(tf.get_collection('Threshold_penalties'))  # L0 proxy del EQL

        mse_loss = tf.losses.mean_squared_error(labels, predictions)
        normal_loss = tf.losses.get_total_loss() + P_theta  # (incluye L1 si corresponde)
        # durante el "penalty epoch" fuerzo a optimizar la penalización
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
    # ojo: para no quedarme corto con el batch en valid/test
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

    # cargo parámetros (puedo sobreescribir por CLI) + metadata del dataset
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

    # input functions para train/penalty/val/test
    train_input, penalty_train_input, val_input, test_input = get_input_fns(**runtime_params, **metadata)

    print('One train episode equals %d normal epochs and 1 penalty epoch.' % runtime_params['penalty_every'])
    for ep in range(1, max_episode + 1):
        print('Train episode: %d / %d' % (ep, max_episode))
        penalty_flag = True
        eqlearner.train(input_fn=penalty_train_input)               # 1 pasada dedicada a “penalty”
        penalty_flag = False
        eqlearner.train(input_fn=train_input, hooks=[logging_hook]) # luego las pasadas normales

    print('Training complete. Evaluating...')

    # =========================
    #   Evaluación FINITA (para que no se quede corriendo)
    # =========================
    tv = np.load(runtime_params['train_val_file'] + ".npz")
    Xva, yva = tv['val_inputs'], tv['val_targets']
    n_val = int(Xva.shape[0])
    bs = int(runtime_params['batch_size'])
    steps_val = max(1, int(np.ceil(n_val / float(bs))))
    val_res = eqlearner.evaluate(input_fn=val_input, name='validation', hooks=[evaluation_hook], steps=steps_val)

    results = dict(val_error=float(val_res['loss']), complexity=int(evaluation_hook.get_complexity()))

    # test/extrapolación: uso evaluate_estimator (también finito)
    extr = None
    if test_input is not None and os.path.exists(runtime_params['test_file'] + ".npz"):
        te = np.load(runtime_params['test_file'] + ".npz")
        Xte = te['test_inputs']
        if Xte.shape[0] > 0:
            extr = evaluate_estimator(eqlearner, runtime_params['test_file'], batch_size=bs,
                                      logger=tf.compat.v1.logging)
            results['extr_error'] = float(extr['loss'])

    # ========= Métricas + plots =========
    Xtr, ytr = tv['train_inputs'], tv['train_targets']
    if test_input is not None and os.path.exists(runtime_params['test_file'] + ".npz"):
        yte = te['test_targets']
    else:
        Xte, yte = np.zeros((0, Xtr.shape[1])), np.zeros((0, ytr.shape[1]))

    ytr_hat = _predict_all(eqlearner, Xtr)
    yva_hat = _predict_all(eqlearner, Xva)
    yte_hat = _predict_all(eqlearner, Xte) if 'Xte' in locals() and len(Xte) > 0 else np.zeros_like(yte)

    metrics = {
        "train": {"MSE": float(mse(ytr, ytr_hat)), "NRMSE": float(nrmse(ytr, ytr_hat)), "R2": float(r2_score(ytr, ytr_hat))},
        "val":   {"MSE": float(mse(yva, yva_hat)), "NRMSE": float(nrmse(yva, yva_hat)), "R2": float(r2_score(yva, yva_hat))},
        "test":  ({"MSE": float(mse(yte, yte_hat)), "NRMSE": float(nrmse(yte, yte_hat)), "R2": float(r2_score(yte, yte_hat))}
                  if 'Xte' in locals() and len(Xte) > 0 else {})
    }
    results.update({"metrics": metrics})

    # el hook ya guarda los PNG (latex_y*.png / graph_y*.png); acá solo apunto a eso
    results["equation"] = "Las ecuaciones simbólicas están en la carpeta del experimento (latex_y*.png / graph_y*.png)."

    outdir = runtime_params['model_dir']
    write_json(os.path.join(outdir, "analysis_summary.json"), results)
    parity_plot(ytr, ytr_hat, os.path.join(outdir, "parity_train.png"), "Train: y vs ŷ")
    parity_plot(yva, yva_hat, os.path.join(outdir, "parity_val.png"),   "Val: y vs ŷ")
    if 'Xte' in locals() and len(Xte) > 0:
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

    print('Model evaluated. Results:\n', results)
