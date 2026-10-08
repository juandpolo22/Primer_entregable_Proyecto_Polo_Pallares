"""Textos del dashboard (compartidos con la vista previa HTML y con los capítulos 13–15 del Jupyter Book).

Las cifras que aparecen en los textos NO se escriben a mano: se calculan aquí con los mismos artefactos
(``dashboard/datos/``) que alimentan las figuras. Si cambian los datos o los modelos, los textos cambian con ellos.
Las interpretaciones completas están en el Jupyter Book (capítulos 13–15).
"""
from __future__ import annotations

import numpy as np

import figuras as F

n_ = F.num
p_ = F.pct


def _miles(x) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Cifras derivadas de los artefactos
# ---------------------------------------------------------------------------
_c = F.contexto()
_prueba = F.tabla("modelos_prueba.csv").set_index("modelo")
_ref = F.tabla("referencia_empirica.csv").iloc[0]
_techo = F.tabla("techo_sin_fuga.csv").iloc[0]
_fuga = F.tabla("con_fuga_vs_sin_fuga.csv")
_celdas = F.tabla("probabilidad_por_celda.csv").set_index("mode_x_daypart")
_elegido = _prueba.loc[F.MODELO_ELEGIDO]
_dummy = _prueba.loc[F.MODELO_DUMMY]
_aditivo = _prueba.loc["Logística aditiva (modo + hora)"]
_moran = F.tabla("moran_global.csv").set_index("variable").loc["tasa de tardías"]
_n_puntajes = int(F.predicciones()[f"p::{F.MODELO_ELEGIDO}"].round(6).nunique())
_cal = F.calidad()


def _n_estable_aprendizaje(tol: float = 0.005, paso: int = 5000) -> int:
    """Primer tamaño de entrenamiento cuyo ROC-AUC de validación queda a menos de ``tol`` del máximo (redondeado a ``paso``)."""
    c = F.tabla("curva_aprendizaje.csv")
    cerca = c[c["AUC_valid"] >= c["AUC_valid"].max() - tol]
    return int(round(float(cerca["n_train"].min()) / paso) * paso)


def _desviacion_calibracion() -> float:
    """Mayor diferencia (en puntos de probabilidad) entre probabilidad media predicha y frecuencia observada, modelo elegido."""
    p = F.predicciones()
    g = F._bins_calibracion(p[f"p::{F.MODELO_ELEGIDO}"], p["y"])
    return float((g["pm"] - g["obs"]).abs().max())


LINEAS_POR_PEDIDO = _c["train_lineas"] / _c["pedidos_train"]
N_ESTABLE = _n_estable_aprendizaje()
MAX_DESV_CAL = _desviacion_calibracion()
EXCLUIDAS_TRAIN = _c["train_lineas"] - _c["modelado_train"]

# ---------------------------------------------------------------------------
# Pestaña 1
# ---------------------------------------------------------------------------
PROBLEMA = (
    f"DataCo Global vende artículos deportivos, ropa y electrónica a {_c['paises_destino']} países. Una parte importante de sus "
    "entregas llega después de la fecha prometida. Las entregas tardías deterioran la satisfacción y la "
    "retención de clientes, generan reclamos, devoluciones y penalizaciones, y obligan a reprogramar "
    "transporte e inventario con sobrecostos. Predecir el riesgo de retraso **en el momento de la compra** "
    "permite intervenir antes del despacho (cambiar el modo de envío, priorizar pedidos o ajustar la fecha "
    "prometida) en lugar de reaccionar cuando el daño ya ocurrió."
)
PREGUNTA = ("¿Es posible predecir, al crear un pedido y solo con información disponible en ese instante, "
            "si su entrega llegará tarde respecto al plazo prometido?")
UNIDAD = (
    "**Unidad de observación: la línea de pedido** (un producto dentro de un pedido), no el pedido completo. "
    f"Un pedido puede tener varias líneas ({n_(LINEAS_POR_PEDIDO, 1)} en promedio en el entrenamiento) y **todas las líneas de un pedido "
    "comparten la misma etiqueta**, así que la respuesta para un pedido es la de sus líneas. El modelado, la validación y las métricas "
    "son a nivel de línea; la partición cronológica deja cada pedido completo en un solo conjunto "
    f"(pedidos compartidos entre entrenamiento y prueba: {_cal['integridad']['pedidos_compartidos_train_test']}) y los intervalos de confianza "
    "se obtienen remuestreando **pedidos** (`order_id`), no líneas."
)
OBJETIVO = (
    "En la población de modelado, tras excluir los pedidos cancelados o sospechosos de fraude, `Late_delivery_risk` = 1 "
    "identifica una entrega tardía (días reales > días programados) y 0 una entrega no tardía (adelantada o a tiempo). "
    "En el dataset original, los pedidos cancelados o sospechosos de fraude tienen etiqueta 0 aunque **no** representan una entrega "
    f"puntual (nunca se entregaron): por eso se excluyen ({_miles(EXCLUIDAS_TRAIN)} líneas del entrenamiento) y no se interpretan como entregas a tiempo."
)
METRICA = (
    "**Métrica principal: ROC-AUC.** Mide la capacidad de ordenar las líneas por riesgo (el uso previsto), "
    f"no depende del umbral de decisión y las clases están casi balanceadas (≈ {p_(_c['tasa_tardia_modelado'], 0)} tardías en la población de modelado). "
    f"Sirve también para detectar fuga: el Dummy vale 0,5, la referencia empírica sin fuga ≈ {n_(_ref['ROC-AUC'])} y con fuga se llega a ≈ {n_(_fuga.iloc[0, 1])}. "
    "Accuracy, Precision, Recall, F1, Balanced Accuracy, Average Precision y Brier son métricas complementarias; las que dependen del umbral "
    f"se reportan con el umbral {n_(_c['umbral_final'], 2)} fijado de antemano."
)
SIN_FUGA = (
    "**Versión sin fuga.** Se excluyen las variables que solo se conocen después del despacho: días reales de "
    "envío, estado de la entrega, estado del pedido y fecha de despacho."
)
FUENTE = ("Constante, Silva y Pereira (2019). *DataCo Smart Supply Chain for Big Data Analysis*. Mendeley Data, "
          "DOI 10.17632/8gx2fvg2k6.5, licencia CC BY 4.0. Réplica en Kaggle (shashwatwork).")

DUMMY_NOTA = (
    f"**{F.MODELO_DUMMY}:** predice siempre la clase mayoritaria (tardía) y usa la prevalencia como probabilidad. "
    f"Por eso tiene Recall = {n_(_dummy['Recall'], 0)} y un F1 relativamente alto ({n_(_dummy['F1'])}) sin haber aprendido ningún patrón: "
    f"su ROC-AUC y su Balanced Accuracy valen {n_(_dummy['ROC-AUC'], 1)}."
)
IC_NOTA = "Los IC 95 % se obtienen mediante bootstrap agrupado por pedido."

TOOLTIPS = {
    "ROC-AUC": "Probabilidad de que una línea tardía reciba un puntaje mayor que una no tardía. No depende del umbral. 0,5 = azar.",
    "Accuracy": "Proporción de aciertos con el umbral fijado. Engañosa cuando una clase domina: el Dummy ya acierta la prevalencia.",
    "Precision": "De las líneas marcadas como tardías, ¿qué fracción realmente lo es? TP / (TP + FP). Depende del umbral.",
    "Recall": "De las líneas realmente tardías, ¿qué fracción se detecta? TP / (TP + FN). Depende del umbral.",
    "F1": "Media armónica de Precision y Recall. Depende del umbral y de la prevalencia.",
    "Balanced Accuracy": "Promedio de Recall y especificidad: el Dummy vale 0,5 sin importar la prevalencia.",
    "Average Precision": "Área bajo la curva Precision–Recall (resume Precision a distintos niveles de Recall). El Dummy vale la prevalencia.",
    "Brier": "Error cuadrático medio de las probabilidades (menor es mejor). Mide calibración y discriminación a la vez.",
    "ROC-AUC CV temporal": "ROC-AUC medio de la validación cruzada temporal del entrenamiento ± desviación entre pliegues.",
}

# ---------------------------------------------------------------------------
# Textos de «Objetivo y calidad» y de fuga
# ---------------------------------------------------------------------------
CALIDAD_NOTA = ("Resumen de los resultados ya calculados en los capítulos 2 y 9 (tablas faltantes_por_columna, auditoria_duplicados, "
                "valores_inconsistentes y auditoria_fuga). El detalle completo está en el Jupyter Book.")
FUGA_DISPONIBLES = ("Se conocen cuando el cliente compra (modo de envío, producto, cliente, destino, "
                    "fecha y hora). Son las únicas que pueden usarse como predictoras.")
FUGA_POSTERIORES = ("Solo se conocen cuando el pedido ya se despachó o entregó. "
                    "No se usan en ningún modelo presentado como válido; solo aparecen en la comparación con/sin fuga como demostración metodológica.")

# ---------------------------------------------------------------------------
# Lecturas cortas (interpretaciones completas en los capítulos 13–15)
# ---------------------------------------------------------------------------
LECTURAS = {
    "particion": "La prueba es el último 20 % del tiempo; los 7 días previos se descartan porque su etiqueta aún no se conocería.",
    "objetivo": "El objetivo sale del estado de entrega. Los pedidos cancelados quedan como 0 aunque nunca se entregaron: no son entregas a tiempo.",
    "tasa_categoria": "El modo de envío es la única variable que separa claramente las clases: First Class se retrasa casi siempre y Standard Class bastante menos. En las demás las tasas quedan cerca de la global.",
    "numerica": "Las distribuciones de tardías y no tardías se superponen: ninguna numérica de la línea separa las clases.",
    "hora_modo": "En Same Day, los pedidos hechos antes de las 12:00 prácticamente nunca llegan tarde y los de la tarde casi siempre: hay una hora de corte.",
    "serie": (f"La tasa semanal fluctúa alrededor de {p_(_c['tasa_tardia_train'], 0)}. En los análisis realizados no se observa una tendencia o "
              "estacionalidad relevante para explicar el objetivo (ADF/KPSS y Ljung-Box, capítulo 7)."),
    "mapa": ("No se detectó evidencia estadísticamente significativa de autocorrelación espacial con la prueba utilizada "
             f"(I de Moran global de la tasa por celda de 1°: I = {n_(_moran['I'])}, p = {n_(_moran['p_permutacion'], 2)})."),
    "fuga": (f"Las variables posteriores al pedido delatan la respuesta: con ellas el ROC-AUC sería ≈ {n_(_fuga.iloc[0, 1])}. Con las variables y relaciones "
             f"evaluadas, la referencia empírica sin fuga basada en modo × jornada alcanza ≈ {n_(_techo['ROC-AUC'])} (benchmark observado con la información "
             "disponible; no se afirma que sea un máximo demostrado)."),
    "tabla": (f"Con las variables y relaciones evaluadas, la referencia empírica basada en modo × jornada alcanza ROC-AUC ≈ {n_(_ref['ROC-AUC'])} en la prueba "
              f"y la regresión logística modo × jornada obtiene un resultado prácticamente equivalente ({n_(_elegido['ROC-AUC'])}). "
              f"Con umbral {n_(_c['umbral_final'], 2)} su Precision ({n_(_elegido['Precision'])}) es alta y su Recall ({n_(_elegido['Recall'])}) moderado."),
    "roc": f"La curva del modelo elegido tiene pocos vértices porque su puntaje solo toma {_n_puntajes} valores (4 modos × 2 jornadas).",
    "pr": (f"La línea punteada es la Precision de un clasificador sin información (la prevalencia, {n_(F.predicciones()['y'].mean())}). "
           f"Average Precision: {n_(_elegido['Average Precision'])} (modo × jornada) frente a {n_(_aditivo['Average Precision'])} (aditiva) "
           f"y {n_(_dummy['Average Precision'])} (Dummy)."),
    "confusion": "Mueve el umbral: el ROC-AUC no cambia, pero sí cuántas líneas se marcan como tardías, la Precision y el Recall.",
    "umbral_aviso": ("Exploración del umbral con predicciones out-of-fold del entrenamiento. "
                     "El conjunto de prueba no se utiliza para seleccionar el umbral."),
    "umbral_test": f"Evaluación final en TEST con umbral previamente fijado = {n_(_c['umbral_final'], 2)}",
    "calibracion": (f"Para el modelo modo × jornada cada probabilidad predicha distinta es un punto (los empates no se parten en deciles): "
                    f"la mayor diferencia entre probabilidad media predicha y frecuencia observada es de {F.num(100 * MAX_DESV_CAL, 1)} puntos porcentuales."),
    "celdas": ("El modelo reproduce las diferencias observadas en entrenamiento entre las combinaciones de modo de envío y jornada: "
               f"tasas altas en First Class y Same Day de la tarde ({p_(_celdas.loc['Same Day | PM', 'tasa_train'], 0)} en entrenamiento) y la más baja en Same Day de la mañana "
               f"({p_(_celdas.loc['Same Day | AM', 'tasa_train'], 0)})."),
    "subgrupos": ("El ROC-AUC observado es similar entre los subgrupos evaluados. Este análisis describe estabilidad del desempeño y no constituye "
                  "una auditoría completa de equidad o fairness."),
    "aprendizaje": (f"Las curvas de entrenamiento y validación son cercanas y la mejora se estabiliza aproximadamente desde {_miles(N_ESTABLE)} líneas; "
                    "para esta especificación no se observa evidencia importante de sobreajuste y el beneficio marginal de aumentar la muestra parece pequeño."),
    "seleccion": ("Añadir producto, cliente, destino o coordenadas no mejora el ROC-AUC de validación cruzada respecto de modo × jornada "
                  "(diferencias dentro de la variación entre pliegues); se elige el conjunto más simple."),
}
