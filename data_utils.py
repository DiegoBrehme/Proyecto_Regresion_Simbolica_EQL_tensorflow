
# -*- coding: utf-8 -*-
"""
Utilidades de datos para EQL (TF1):
- Lectura robusta de archivos .gz (pickle) con distintos formatos.
- Generación de datasets tf.data en pares (X, y).
- Construcción de input_fns para train / val / penalty / test.
- Extracción robusta de metadatos.

Soporta estos formatos al cargar sets:
  ((Xtr,ytr),(Xva,yva))
  (Xtr,ytr,Xva,yva)
  {'train':(X,y), 'val':(X,y)}
  {'train':{'X':..,'y':..}, 'val':{'X':..,'y':..}}
  {'Xtr':..,'ytr':..,'Xva':..,'yva':..} y variantes
Para test:
  (X,y)  |  {'X':..,'y':..}  |  {'test':(X,y)}  |  (Xtr,ytr,Xva,yva)->usa (Xva,yva)
"""

import gzip
import os
import os.path
import pickle
from ast import literal_eval
from sys import argv

import numpy as np
import tensorflow as tf

from utils import to_float32, number_of_positional_arguments


# ============================
# Helpers de normalización
# ============================

def _to_py(obj):
    """Convierte ndarrays dtype=object a objetos Python (lista/tupla/escalares)."""
    import numpy as _np
    if isinstance(obj, _np.ndarray) and obj.dtype == object:
        try:
            return obj.item()
        except Exception:
            return obj.tolist()
    return obj

def _pair_like(x):
    return isinstance(x, (list, tuple)) and len(x) == 2

def _quad_like(x):
    return isinstance(x, (list, tuple)) and len(x) == 4

def _as_xy_pair(raw):
    """Normaliza cualquier estructura a un par (X, y) para test/penalty."""
    raw = _to_py(raw)

    # (X, y)
    if _pair_like(raw):
        return raw[0], raw[1]

    # (Xtr,ytr,Xva,yva) -> usa el par final por conveniencia (p.ej. test)
    if _quad_like(raw):
        return raw[2], raw[3]

    # Diccionarios varias formas
    if isinstance(raw, dict):
        # claves directas
        for kx, ky in (("X", "y"),
                       ("Xte", "yte"), ("Xtest", "ytest"),
                       ("X_val", "y_val"), ("Xva", "yva")):
            if kx in raw and ky in raw:
                return raw[kx], raw[ky]
        # pares anidados
        for k in ("test", "val", "data"):
            if k in raw and _pair_like(raw[k]):
                return raw[k][0], raw[k][1]

    raise ValueError("Formato de datos no reconocido para (X, y).")

def _split_train_val(raw):
    """Normaliza train/val a ((Xtr,ytr),(Xva,yva)) desde varios formatos."""
    raw = _to_py(raw)

    # ((Xtr,ytr),(Xva,yva))
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and all(_pair_like(e) for e in raw):
        return raw[0], raw[1]

    # (Xtr,ytr,Xva,yva)
    if _quad_like(raw):
        Xtr, ytr, Xva, yva = raw
        return (Xtr, ytr), (Xva, yva)

    # Diccionarios
    if isinstance(raw, dict):
        # {'train':(Xtr,ytr), 'val':(Xva,yva)}
        if "train" in raw and "val" in raw and _pair_like(raw["train"]) and _pair_like(raw["val"]):
            tr, va = raw["train"], raw["val"]
            return (tr[0], tr[1]), (va[0], va[1])

        # {'train':{'X':..,'y':..}, 'val':{'X':..,'y':..}}
        if "train" in raw and "val" in raw and isinstance(raw["train"], dict) and isinstance(raw["val"], dict):
            tr, va = raw["train"], raw["val"]
            # train
            Xtr = tr.get("X", tr.get("Xtr", tr.get("X_train")))
            ytr = tr.get("y", tr.get("ytr", tr.get("y_train")))
            # val
            Xva = va.get("X", va.get("Xva", va.get("X_val")))
            yva = va.get("y", va.get("yva", va.get("y_val")))
            if Xtr is not None and ytr is not None and Xva is not None and yva is not None:
                return (Xtr, ytr), (Xva, yva)

        # {'Xtr':..,'ytr':..,'Xva':..,'yva':..} o variantes
        key_sets = [
            ("Xtr", "ytr", "Xva", "yva"),
            ("X_train", "y_train", "X_val", "y_val"),
            ("Xtr", "ytr", "Xv", "yv"),
        ]
        for kXtr, kytr, kXva, kyva in key_sets:
            if all(k in raw for k in (kXtr, kytr, kXva, kyva)):
                return (raw[kXtr], raw[kytr]), (raw[kXva], raw[kyva])

    raise ValueError("Formato de train_val_file no reconocido. "
                     "Acepto ((Xtr,ytr),(Xva,yva)), (Xtr,ytr,Xva,yva), "
                     "{'train':(X,y),'val':(X,y)} o diccionarios equivalentes.")

# ============================
# Funciones del paper (F1-F5)
# ============================

def F1(x1, x2, x3, x4):
    """Requires 1 hidden layer."""
    y0 = (np.sin(np.pi * x1) + np.sin(2 * np.pi * x2 + np.pi / 8.0) + x2 - x3 * x4) / 3.0
    return y0,

def F2(x1, x2, x3, x4):
    """Requires 2 hidden layers."""
    y0 = (np.sin(np.pi * x1) + x2 * np.cos(2 * np.pi * x1 + np.pi / 4.0) + x3 - x4 * x4) / 3.0
    return y0,

def F3(x1, x2, x3, x4):
    """Requires 2 hidden layers."""
    y0 = ((1.0 + x2) * np.sin(np.pi * x1) + x2 * x3 * x4) / 3.0
    return y0,

def F4(x1, x2, x3, x4):
    """Requires 4 hidden layers."""
    y0 = 0.5 * (np.sin(np.pi * x1) + np.cos(2.0 * x2 * np.sin(np.pi * x1)) + x2 * x3 * x4)
    return y0,

def F5(x1, x2, x3, x4):
    """Equation for cart pendulum. Requires 4 hidden layers."""
    y1 = x3
    y2 = x4
    y3 = (-x1 - 0.01 * x3 + x4 ** 2 * np.sin(x2) + 0.1 * x4 * np.cos(x2) + 9.81 * np.sin(x2) * np.cos(x2)) \
         / (np.sin(x2) ** 2 + 1)
    y4 = -0.2 * x4 - 19.62 * np.sin(x2) + x1 * np.cos(x2) + 0.01 * x3 * np.cos(x2) - x4 ** 2 * np.sin(x2) * np.cos(x2) \
         / (np.sin(x2) ** 2 + 1)
    return y1, y2, y3, y4,

# ============================
# Generación de datos sintéticos
# ============================

data_gen_params = {
    'file_name': 'F1data',              # nombre base (se guardará en data/<file_name>_train_val / _test)
    'fn_to_learn': 'F1',                # función definida arriba
    'train_val_examples': 10000,        # total train+val
    'train_val_bounds': (-1.0, 1.0),    # dominio para train/val
    'test_examples': 5000,              # si None, no crea test
    'test_bounds': (-2.0, 2.0),         # dominio test
    'noise': 0.01,
    'seed': None
}

def generate_data(fn, num_examples, bounds, noise, seed=None):
    np.random.seed(seed)
    lower, upper = bounds
    input_dim = number_of_positional_arguments(fn)
    xs = np.random.uniform(lower, upper, (num_examples, input_dim)).astype(np.float32)
    xs_as_list = np.split(xs, input_dim, axis=1)
    ys = fn(*xs_as_list)
    ys = np.concatenate(ys, axis=1)
    ys = ys + np.random.uniform(-noise, noise, ys.shape).astype(np.float32)
    return xs, ys

# ============================
# Lectura de archivos .gz
# ============================

def data_from_file(filename, split=None):
    """
    Lee y normaliza el archivo 'filename' (pickle.gz).
    Si split is None -> retorna lo que haya (sin tocar), útil para test.
    Si split no es None y el archivo trae (X,y) -> parte en ((Xtr,ytr),(Xva,yva)).
    Si ya trae train/val en cualquier formato soportado -> lo normaliza igual.
    """
    raw = to_float32(pickle.load(gzip.open(filename, "rb"), encoding='latin1'))

    # Si piden split explícito y el archivo es un par (X,y), lo partimos aquí.
    if split is not None:
        try:
            # si ya viene como estructura train/val, solo normalizamos
            return _split_train_val(raw)
        except Exception:
            # si raw es un par simple, lo partimos manualmente
            if _pair_like(raw):
                X, y = raw
                n = X.shape[0]
                cut = int(n * split)
                return (X[:cut], y[:cut]), (X[cut:], y[cut:])
            # si no, relanzar el error anterior
            raise

    # Sin split, devolvemos tal cual (sirve para test/otros usos)
    return raw

# ============================
# tf.data input
# ============================

def input_from_data(data, batch_size, repeats=1, shuffle=True):
    """
    Convierte un par (X, y) en batches usando tf.data.
    Siempre retorna exactamente 2 tensores (x_batch, y_batch).
    """
    # Enforce par (X, y)
    X, y = data

    # tf1 compat
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, X.shape[0]))
    ds = ds.repeat(repeats).batch(batch_size)

    it = ds.make_one_shot_iterator()
    x_batch, y_batch = it.get_next()
    return x_batch, y_batch

# ============================
# Penalty data
# ============================

def get_penalty_data(num_examples, penalty_bounds, num_inputs, num_outputs):
    """
    Genera datos para penalty: y = 0, X ~ U(bounds).
    penalty_bounds puede ser (low, high) global o lista de tuplas por dimensión.
    """
    if isinstance(penalty_bounds, tuple):
        lower, upper = penalty_bounds
    else:
        lower, upper = zip(*penalty_bounds)
    xs = np.random.uniform(lower, upper, (num_examples, num_inputs)).astype(np.float32)
    ys = np.zeros((num_examples, num_outputs), dtype=np.float32)
    return xs, ys

# ============================
# Input fns (train/val/penalty/test)
# ============================

def get_input_fns(train_val_split, batch_size, train_val_file, test_file, penalty_every,
                  num_inputs, num_outputs, train_val_examples, penalty_bounds,
                  extracted_penalty_bounds, **_):
    """
    Devuelve input_fn para train, penalty-train, val y test, normalizando SIEMPRE a pares (X, y).
    """

    # 1) límites para penalty
    penalty_bounds = penalty_bounds or extracted_penalty_bounds

    # 2) TRAIN/VAL (si el archivo trae un único par, lo partimos con 'split')
    raw_train_val = data_from_file(train_val_file, split=train_val_split)
    (Xtr, ytr), (Xva, yva) = _split_train_val(raw_train_val)
    train_data = (Xtr, ytr)
    val_data   = (Xva, yva)

    # 3) PENALTY
    penalty_raw = get_penalty_data(
        num_examples=int(train_val_split * train_val_examples),
        penalty_bounds=penalty_bounds,
        num_inputs=num_inputs,
        num_outputs=num_outputs
    )
    penalty_data = _as_xy_pair(penalty_raw)

    # 4) TEST (si existe)
    if test_file is not None:
        test_raw  = data_from_file(test_file, split=None)
        test_data = _as_xy_pair(test_raw)
        test_input = lambda: input_from_data(data=test_data, batch_size=batch_size, repeats=1, shuffle=False)
    else:
        test_input = None

    # 5) input_fns
    train_input   = lambda: input_from_data(data=train_data,   batch_size=batch_size, repeats=penalty_every, shuffle=True)
    val_input     = lambda: input_from_data(data=val_data,     batch_size=batch_size, repeats=1,             shuffle=False)
    penalty_input = lambda: input_from_data(data=penalty_data, batch_size=batch_size, repeats=1,             shuffle=False)

    return train_input, penalty_input, val_input, test_input

# ============================
# Metadatos
# ============================

def extract_metadata(train_val_file, test_file, domain_bound_factor=2, res_bound_factor=10):
    """
    Extrae: train_val_examples, num_inputs, num_outputs,
            extracted_output_bound, extracted_penalty_bounds
    de manera robusta.
    """
    # Normalizamos train/val: si el archivo trae un par, NO partimos aquí (solo leer);
    # pero si trae estructura train/val, la interpretamos.
    raw_tv = to_float32(pickle.load(gzip.open(train_val_file, "rb"), encoding='latin1'))

    # Intentar leer como par directo
    Xtv = ytv = None
    try:
        Xtv, ytv = _as_xy_pair(raw_tv)
    except Exception:
        # Entonces debe venir como train/val
        (Xtr, ytr), (Xva, yva) = _split_train_val(raw_tv)
        Xtv = Xtr
        ytv = ytr

    train_val_examples = int(Xtv.shape[0])
    num_inputs  = int(Xtv.shape[1])
    num_outputs = int(ytv.shape[1])
    extracted_output_bound = float(np.max(np.abs(ytv)) * res_bound_factor)

    # penalty bounds a partir de test si existe, si no desde train/val escalado
    if test_file is not None:
        raw_te = to_float32(pickle.load(gzip.open(test_file, "rb"), encoding='latin1'))
        Xte, _ = _as_xy_pair(raw_te)
        mins = np.min(Xte, axis=0)
        maxs = np.max(Xte, axis=0)
    else:
        mins = np.min(Xtv, axis=0) * domain_bound_factor
        maxs = np.max(Xtv, axis=0) * domain_bound_factor

    extracted_penalty_bounds = list(zip(mins, maxs))

    metadata = dict(train_val_examples=train_val_examples,
                    num_inputs=num_inputs,
                    num_outputs=num_outputs,
                    extracted_output_bound=extracted_output_bound,
                    extracted_penalty_bounds=extracted_penalty_bounds)
    return metadata

# ============================
# Crear archivos desde función
# ============================

def files_from_fn(file_name, fn_to_learn, train_val_examples, test_examples, train_val_bounds,
                  test_bounds, noise, seed=None):
    """
    Genera archivos .gz con train/val, test y meta-datos desde una función.
    """
    fn_to_learn = globals()[fn_to_learn]
    if not os.path.exists('data'):
        os.mkdir('data')

    train_val_set = generate_data(fn=fn_to_learn,
                                  num_examples=train_val_examples,
                                  bounds=train_val_bounds,
                                  noise=noise,
                                  seed=seed)

    train_val_data_file = os.path.join('data', file_name + '_train_val')
    pickle.dump(train_val_set, gzip.open(train_val_data_file, "wb"))
    print('Successfully created train/val data file in %s.' % train_val_data_file)

    if test_examples is not None:
        test_set = generate_data(fn=fn_to_learn,
                                 num_examples=test_examples,
                                 bounds=test_bounds,
                                 noise=noise,
                                 seed=seed)
        test_data_file = os.path.join('data', file_name + '_test')
        pickle.dump(test_set, gzip.open(test_data_file, "wb"))
        print('Successfully created test data file in %s.' % test_data_file)

# ============================
# CLI
# ============================

if __name__ == '__main__':
    if len(argv) > 1:
        print('Updating default parameters.')
        data_gen_params.update(literal_eval(argv[1]))
    else:
        print('Using default parameters.')
    files_from_fn(**data_gen_params)
