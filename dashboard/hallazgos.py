"""Títulos que comunican el hallazgo de cada gráfico. Todos se calculan desde los artefactos (``datos/``) y los textos.

Se escribe un hallazgo solo cuando el resultado existente lo respalda; si el dato cambia, el título cambia con él.
"""
from __future__ import annotations

import figuras as F
import textos as T


def titulo_objetivo() -> str:
    tasa = F.contexto()["tasa_tardia_train"]
    return f"{F.pct(tasa)} de las líneas llegan tarde" + (": más de la mitad" if tasa > 0.5 else "")


def titulo_categoria(variable: str = "shipping_mode") -> str:
    t = F.tabla("tasas_por_categoria.csv")
    t = t[t["variable"] == variable].sort_values("tasa")
    hi, lo = t.iloc[-1], t.iloc[0]
    return f"{hi['categoria']} se retrasa en el {F.pct(hi['tasa'])} de las líneas; {lo['categoria']}, en el {F.pct(lo['tasa'])}"


TITULO_HORA = "Same Day: antes del mediodía casi nunca llega tarde; por la tarde, casi siempre"
TITULO_NUMERICA = "Las distribuciones de tardías y no tardías se superponen"


def titulo_serie() -> str:
    return f"La tasa semanal de tardías se mantiene alrededor de {F.pct(F.contexto()['tasa_tardia_train'], 0)}"


def titulo_moran() -> str:
    m = F.tabla("moran_global.csv").set_index("variable").loc["tasa de tardías"]
    return f"Sin evidencia de autocorrelación espacial: I de Moran = {F.num(m['I'])} (p = {F.num(m['p_permutacion'], 2)})"


def titulo_fuga_auc() -> str:
    auc = F.tabla("auc_univariado_todas.csv")
    post = auc[auc["variable"].isin(F._posteriores())].sort_values("AUC univariado", ascending=False).iloc[0]
    return f"Las variables posteriores delatan la respuesta: {post['variable']} alcanza un AUC univariado de {F.num(post['AUC univariado'])}"


def titulo_con_sin_fuga() -> str:
    t = F.tabla("con_fuga_vs_sin_fuga.csv")
    return f"Con fuga el ROC-AUC llegaría a {F.num(t.iloc[0, 1])}; sin fuga, {F.num(t.iloc[2, 1])}"


def titulo_particion() -> str:
    return T.LECTURAS["particion"].split(";")[0]


def dif_calibracion() -> float:
    """Mayor diferencia (en puntos porcentuales) entre probabilidad media predicha y frecuencia observada del modelo elegido."""
    p = F.predicciones()
    g = F._bins_calibracion(p[f"p::{F.MODELO_ELEGIDO}"], p["y"])
    return float((g["pm"] - g["obs"]).abs().max() * 100)


def titulo_calibracion() -> str:
    return f"La mayor diferencia entre probabilidad predicha y frecuencia observada es de {F.num(dif_calibracion(), 1)} puntos porcentuales"


TITULO_CELDAS = "Las probabilidades reproducen las tasas observadas por modo × jornada"
TITULO_SELECCION = "Más variables no mejoran el ROC-AUC de validación respecto de modo × jornada"
TITULO_APRENDIZAJE = "La mejora se estabiliza con ≈ 40.000 líneas de entrenamiento"
TITULO_SUBGRUPOS = "El ROC-AUC es similar entre los subgrupos evaluados"


def titulo_roc() -> str:
    p = F.tabla("modelos_prueba.csv").set_index("modelo")
    return (f"El modelo supera claramente la línea base: ROC-AUC {F.num(p.loc[F.MODELO_ELEGIDO, 'ROC-AUC'])} "
            f"frente a {F.num(p.loc[F.MODELO_DUMMY, 'ROC-AUC'], 1)}")


def titulo_matriz_test(m_test: dict, umbral: float) -> str:
    return f"Con umbral {F.num(umbral, 2)} el modelo marca el {F.pct(m_test['marcados_tardía'], 0)} de las líneas como tardías"


def titulo_matriz_oof(m_oof: dict) -> str:
    return f"Con predicciones fuera de pliegue se marca el {F.pct(m_oof['marcados_tardía'], 0)}: patrón similar"


def lectura_matriz(m_test: dict) -> str:
    cm = m_test["cm"]
    return ("Filas = realidad, columnas = predicción. El porcentaje de cada celda es sobre el total de su fila real; en la prueba acierta el "
            f"{F.pct(cm[0, 0] / cm[0].sum(), 0)} de las no tardías y el {F.pct(cm[1, 1] / cm[1].sum(), 0)} de las tardías.")


# Texto breve del umbral (solo habla del ROC-AUC, la métrica principal)
LECTURA_UMBRAL = ("Mueve el umbral: el ROC-AUC no cambia, pero sí cuántas líneas se marcan como tardías y cómo se reparten "
                  "aciertos y errores en la matriz.")
