# evaluation.py
"""
Módulo de manipulación/evaluación simbólica para EQL (TF1).
- EvaluationHook: genera expresiones simbólicas (opcional) y calcula complejidad.
- Exporta SIEMPRE (si está activado) expresiones en texto (.txt) y LaTeX plano (.tex).
- Opcionalmente (si hay herramientas y symbolic_png=True) exporta PNG de LaTeX y del grafo (Graphviz).
- Complejidad: calculate_complexity -> complexity_of_layer -> complexity_of_node
"""

import os
from os import path
import math
from functools import reduce
import shutil

import numpy as np
import sympy
import tensorflow as tf
from sympy.printing.dot import dotprint

# TF1 compatibility
tf1 = tf.compat.v1
from tensorflow.python.training.session_run_hook import SessionRunHook

from timeout import time_limit, TimeoutException
from utils import generate_arguments, yield_with_repeats, weight_name_for_i
from data_utils import data_from_file, _as_xy_pair, input_from_data


# =========================
#   Utilidades de entorno
# =========================
def _have(cmd: str) -> bool:
    """True si el ejecutable está disponible en PATH."""
    return shutil.which(cmd) is not None


def _tools_for_latex_png_ok() -> bool:
    # SymPy->LaTeX->dvi/png suele requerir latex + dvipng; ghostscript a veces es usado por la ruta png/ps
    return _have("latex") and (_have("dvipng") or _have("gswin64c"))


def _tools_for_graphviz_png_ok() -> bool:
    return _have("dot")


# =========================
#   Simplificación SymPy
# =========================
@time_limit(60)
def proper_simplify(expr):
    """Combinación básica: simplify + trigsimp con timeout (60s)."""
    return sympy.simplify(sympy.trigsimp(expr))


def round_sympy_expr(expr, decimals):
    """Redondea constantes float dentro del árbol simbólico."""
    rounded_expr = expr
    for a in sympy.preorder_traversal(expr):
        if isinstance(a, sympy.Float):
            rounded_expr = rounded_expr.subs(a, round(a, decimals))
    return rounded_expr


# =========================
#   Rutas de salida
# =========================
def _save_text_variants(expr, save_dir, idx, latex_str=None):
    """Guarda y_i.txt (str) y y_i.tex (LaTeX crudo)."""
    os.makedirs(save_dir, exist_ok=True)
    txt_path = path.join(save_dir, f"y{idx}.txt")
    tex_path = path.join(save_dir, f"y{idx}.tex")

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(str(expr))
    except Exception as e:
        print(f"[evaluation] Advertencia: no pude escribir {txt_path}: {e}")

    try:
        if latex_str is None:
            latex_str = sympy.latex(expr)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
    except Exception as e:
        print(f"[evaluation] Advertencia: no pude escribir {tex_path}: {e}")


def _expr_to_latex_png(expr, output_file):
    """Render LaTeX -> PNG mediante SymPy preview (requiere latex + dvipng/gs)."""
    try:
        sympy.preview(expr, viewer='file', filename=output_file)
        return True
    except Exception as e:
        print(f"[evaluation] LaTeX->PNG falló para {output_file}: {e}")
        return False


def _expression_graph_as_png(expr, output_file):
    """Render del grafo (Graphviz) -> PNG usando dotprint."""
    assert output_file.endswith(".png")
    try:
        from graphviz import Source
        graph = Source(dotprint(expr))
        graph.format = 'png'
        graph.render(output_file.rpartition('.png')[0], view=False, cleanup=True)
        return True
    except Exception as e:
        print(f"[evaluation] Graphviz->PNG falló para {output_file}: {e}")
        return False


# =========================
#   Simbolización de capas
# =========================
def get_symbol_list(number_of_symbols):
    """Lista de símbolos de entrada x_i (reales)."""
    return sympy.symbols([f'x_{i + 1}' for i in range(number_of_symbols)], real=True)


def symbolic_matmul_and_bias(input_nodes_symbolic, weight_matrix, bias_vector):
    """Representación simbólica de y = W^T x + b (columna a columna)."""
    def output_for_index(i):
        return bias_vector[i] + sum([w * x for w, x in zip(weight_matrix[:, i], input_nodes_symbolic)])
    return [output_for_index(i) for i in range(weight_matrix.shape[1])]


def symbolic_eql_layer(input_nodes_symbolic, output_fn_group_list):
    """
    Aplica las funciones de una capa EQL a sus entradas simbólicas.
    output_fn_group_list: lista de tuplas (tf_fn, sp_fn, repeats, num_args)
    """
    _, output_fns, repeats, arg_nums = zip(*output_fn_group_list)
    arg_iterator = generate_arguments(input_nodes_symbolic, repeats, arg_nums)
    fn_iterator = yield_with_repeats(output_fns, repeats)
    return [fn(*items) for fn, items in zip(fn_iterator, arg_iterator)]


# =========================
#   Guardado de expresiones
# =========================
def save_symbolic_expression(kernels, biases, fns_list, save_path, round_decimals,
                             save_text=True, save_png=False):
    """
    Construye y guarda las expresiones simbólicas finales de la red.
    - Siempre puede guardar .txt y .tex si save_text=True (no requiere LaTeX/Graphviz).
    - Si save_png=True, intenta además latex_y*.png y graph_y*.png (solo si hay herramientas).

    :param kernels: lista de matrices (numpy)
    :param biases: lista de vectores (numpy)
    :param fns_list: lista de capas con (tf_fn, sp_fn, repeats, num_args)
    :param save_path: carpeta de salida
    :param round_decimals: redondeo de constantes
    :param save_text: bool
    :param save_png: bool
    """
    in_nodes = get_symbol_list(kernels[0].shape[0])
    res = in_nodes
    for kernel, bias, fns in zip(kernels, biases, fns_list):
        res = symbolic_matmul_and_bias(res, kernel, bias)
        res = symbolic_eql_layer(res, fns)

    have_latex_png = _tools_for_latex_png_ok() if save_png else False
    have_graphviz_png = _tools_for_graphviz_png_ok() if save_png else False
    if save_png and not (have_latex_png or have_graphviz_png):
        print("[evaluation] PNG deshabilitado: faltan herramientas (latex/dvipng/ghostscript y/o dot). "
              "Se guardarán solo .txt/.tex.")

    for i, result in enumerate(res):
        # simplificación robusta + redondeo
        result = round_sympy_expr(result, round_decimals)
        try:
            result_simpl = proper_simplify(result)
        except (TimeoutException, RecursionError):
            print(f"[evaluation] Simplificación de y{i} excedió tiempo/recursión. Se usa versión no simplificada.")
            result_simpl = result

        # salida en texto/latex plano
        if save_text:
            _save_text_variants(result_simpl, save_path, i)

        # salida PNG (si las herramientas y el flag lo permiten)
        if save_png and have_latex_png:
            _expr_to_latex_png(result_simpl, path.join(save_path, f'latex_y{i}.png'))
        if save_png and have_graphviz_png:
            _expression_graph_as_png(result_simpl, path.join(save_path, f'graph_y{i}.png'))


# =========================
#   Complejidad
# =========================
def calculate_complexity(kernels, biases, fns_list, thresh):
    """
    Cuenta nodos activos (no identidades) con pesos de entrada y salida significativos.
    """
    complexities = [
        complexity_of_layer(fns=fns, in_biases=in_biases, in_weights=in_weights, out_weights=out_weights, thresh=thresh)
        for fns, in_biases, in_weights, out_weights in zip(fns_list, biases[:-1], kernels[:-1], kernels[1:])
    ]
    return int(sum(complexities))


def complexity_of_layer(fns, in_biases, in_weights, out_weights, thresh):
    """
    Complejidad por capa (número de nodos activos).
    """
    in_weight_sum = np.sum(np.abs(in_weights), axis=0) + np.abs(in_biases)
    out_weight_sum = np.sum(np.abs(out_weights), axis=1)
    output_fns, _, repeats, arg_nums = zip(*fns)
    input_iterator = generate_arguments(all_args=in_weight_sum, repeats=repeats, arg_nums=arg_nums)
    fn_iterator = yield_with_repeats(output_fns, repeats)
    count = sum([complexity_of_node(out_w, in_w_tuple, fn, thresh)
                 for out_w, in_w_tuple, fn in zip(out_weight_sum, input_iterator, fn_iterator)])
    return int(count)


def complexity_of_node(out_weight, in_weights, fn, thresh):
    """Nodo activo = 1 si no es identidad y todos sus pesos superan umbrales."""
    if fn == tf.identity:
        return 0
    all_weights = [out_weight, *in_weights]
    weight_product = np.prod(all_weights)
    if all(abs(w) > thresh for w in all_weights) and (abs(weight_product) > (thresh ** len(all_weights))):
        return 1
    return 0


# =========================
#   Hook de evaluación
# =========================
class EvaluationHook(SessionRunHook):
    """Hook que extrae pesos, calcula complejidad y exporta ecuaciones simbólicas."""

    def __init__(self, list_of_vars, store_path=None):
        self.list_of_vars = list_of_vars
        self.weights = None
        self.store_path = store_path
        self.fns_list = None

        self.round_decimals = 3
        self.complexity = None
        self.iteration = 0
        self.thresh = 0.0

        # Flags de salida
        self.generate_symbolic_expr = False
        self.symbolic_text = True     # opción B por defecto
        self.symbolic_png = False     # PNG opcional (requiere herramientas)

    def begin(self):
        self.iteration = 0

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):
        # En la primera iteración pedimos los tensores de pesos/bias
        if self.iteration == 0:
            graph = tf1.get_default_graph()
            tens = {v: graph.get_tensor_by_name(v) for v in self.list_of_vars}
        else:
            tens = {}
        return tf1.train.SessionRunArgs(fetches=tens)

    def after_run(self, run_context, run_values):
        if self.iteration == 0:
            self.weights = run_values.results
        self.iteration += 1

    def end(self, session):
        if self.store_path is None:
            return
        if self.fns_list is None:
            raise ValueError("Network structure not provided. Llama a init_network_structure(model, params).")

        # separar kernels y biases por nombre para mantener el orden
        kernels = [value for key, value in self.weights.items() if 'kernel' in key.lower()]
        biases  = [value for key, value in self.weights.items() if 'bias'   in key.lower()]

        # complejidad
        self.complexity = calculate_complexity(kernels, biases, self.fns_list, self.thresh)

        # export simbólico (texto y/o PNG)
        if self.generate_symbolic_expr:
            save_symbolic_expression(
                kernels, biases, self.fns_list, self.store_path, self.round_decimals,
                save_text=self.symbolic_text, save_png=self.symbolic_png
            )

    def init_network_structure(self, model, params):
        self.fns_list = [layer.get_fns() for layer in model.eql_layers]
        self.thresh = params.get('complexity_threshold', 0.0)
        # Compatibilidad: el viejo flag sigue mandando
        self.generate_symbolic_expr = bool(params.get('generate_symbolic_expr', False))
        # Nuevos flags (opción B por defecto)
        self.symbolic_text = bool(params.get('symbolic_text', True))
        self.symbolic_png  = bool(params.get('symbolic_png', False))

    def get_complexity(self):
        if self.complexity is not None:
            return self.complexity
        raise ValueError('Complexity not yet evaluated.')


def set_evaluation_hook(num_h_layers, model_dir, **_):
    kernel_tensornames = [weight_name_for_i(i, 'kernel') for i in range(num_h_layers + 1)]
    bias_tensornames   = [weight_name_for_i(i, 'bias')   for i in range(num_h_layers + 1)]
    symbolic_hook = EvaluationHook([*kernel_tensornames, *bias_tensornames], store_path=model_dir)
    return symbolic_hook


# =========================
#   Evaluación finita (test)
# =========================
def evaluate_estimator(estimator, test_file, batch_size, logger=None):
    """
    Evalúa con un número FINITO de steps calculado desde |test|.
    """
    raw_test = data_from_file(test_file)
    Xte, yte = _as_xy_pair(raw_test)
    n_test = int(Xte.shape[0])

    if logger:
        logger.info("Evaluando sobre %d ejemplos de test.", n_test)
    else:
        print(f"Evaluando sobre {n_test} ejemplos de test.")

    test_input = lambda: input_from_data(data=(Xte, yte), batch_size=batch_size, repeats=1)
    steps = max(1, int(math.ceil(n_test / float(batch_size))))

    if logger:
        logger.info("Evaluación en %d steps (batch_size=%d).", steps, batch_size)
    else:
        print(f"Evaluación en {steps} steps (batch_size={batch_size}).")

    results = estimator.evaluate(input_fn=test_input, steps=steps)

    if logger:
        logger.info("Resultados de evaluación: %s", results)
    else:
        print("Resultados de evaluación:", results)

    return results


