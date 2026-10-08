"""Comprueba que el dashboard, los CSV de resultados y las tablas del Jupyter Book usan exactamente las mismas cifras.

    python dashboard/verificar_consistencia.py          # imprime el resultado y termina con código 1 si algo no coincide

También se usa desde el capítulo 13 (``verificar()`` devuelve una tabla con cada comprobación).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figuras as F  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
TABLAS = RAIZ / "tables"
DATOS = F.DATOS
TOL = 1e-9


def _cercano(a, b, tol: float = TOL) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=0, equal_nan=True))


def verificar() -> pd.DataFrame:
    filas = []

    def comprobar(nombre: str, ok: bool, detalle: str = "") -> None:
        filas.append({"comprobación": nombre, "resultado": "OK" if ok else "FALLA", "detalle": detalle})

    prueba = F.tabla("modelos_prueba.csv").set_index("modelo")
    ic = F.tabla("modelos_ic.csv")
    cv = F.tabla("modelos_cv.csv")
    ctx = F.contexto()
    cal = F.calidad()

    # 1. Métricas en la prueba: dashboard vs capítulo 11 (tables/resultados_prueba.csv)
    res11 = pd.read_csv(TABLAS / "resultados_prueba.csv", index_col=0)
    previas = ["Accuracy", "Recall", "F1", "ROC-AUC", "Average Precision", "Brier"]
    for nombre11, nombre in [("Regresión logística", F.MODELO_ELEGIDO), ("Dummy (prior)", F.MODELO_DUMMY)]:
        comprobar(f"Métricas en la prueba de «{nombre}» = capítulo 11", _cercano(res11.loc[nombre11, previas], prueba.loc[nombre, previas]))
    ref = F.tabla("referencia_empirica.csv").iloc[0]
    ref11 = res11.iloc[2]
    comprobar("Referencia empírica (prueba) = capítulo 11 (Accuracy, Recall, F1, ROC-AUC, AP)", _cercano(ref11[previas[:-1]], ref[previas[:-1]]))

    # 2. Intervalos de confianza: dashboard vs capítulo 11
    for nombre11, nombre, archivo in [("Regresión logística", F.MODELO_ELEGIDO, "intervalos_bootstrap_logistica.csv"),
                                      ("Dummy", F.MODELO_DUMMY, "intervalos_bootstrap_dummy.csv")]:
        a = pd.read_csv(TABLAS / archivo, index_col=0)
        b = ic[ic["modelo"] == nombre].set_index("Métrica")[a.columns]
        comunes = [m for m in a.index if m in b.index]
        comprobar(f"IC 95 % de «{nombre11}» = capítulo 11 ({len(comunes)} métricas)", _cercano(a.loc[comunes], b.loc[comunes]))

    # 3. Validación cruzada temporal
    cv11 = pd.read_csv(TABLAS / "cv_resumen.csv")
    fila = cv11[(cv11["esquema"].str.startswith("Temporal")) & (cv11["modelo"] == "Regresión logística")].iloc[0]
    comprobar("ROC-AUC CV temporal del modelo elegido = capítulo 11",
              _cercano(fila["_roc_auc_mean"], cv[(cv["modelo"] == F.MODELO_ELEGIDO) & (cv["métrica"] == "ROC-AUC")]["media"].iloc[0]))

    # 4. Estimaciones puntuales de la tabla de IC = tabla de la prueba (todas las métricas, incluidas Precision y Balanced Accuracy)
    ok = True
    for m in F.modelos():
        e = ic[ic["modelo"] == m].set_index("Métrica")["Estimación"]
        ok &= _cercano(e[F.METRICAS_TABLA], prueba.loc[m, F.METRICAS_TABLA])
    comprobar("Estimaciones de modelos_ic.csv = modelos_prueba.csv (8 métricas, incluye Precision y Balanced Accuracy)", bool(ok))

    # 5. Precision y Balanced Accuracy recalculadas desde las predicciones de la prueba
    p = F.predicciones()
    ok = True
    for m in F.modelos():
        s_ = p[f"p::{m}"].to_numpy()
        pred = (s_ >= F.umbral_final()).astype(int) if not F.es_dummy(m) else np.ones(len(p), dtype=int)
        ok &= _cercano(skm.precision_score(p["y"], pred, zero_division=0), prueba.loc[m, "Precision"])
        ok &= _cercano(skm.balanced_accuracy_score(p["y"], pred), prueba.loc[m, "Balanced Accuracy"])
    comprobar("Precision y Balanced Accuracy de la tabla = recalculadas desde predicciones_prueba", bool(ok))

    # 6. Tamaños y nivel de análisis (línea de pedido)
    tam = pd.read_csv(TABLAS / "tamano_muestra.csv").set_index("medida")["valor"]
    comprobar("Líneas y pedidos de entrenamiento = capítulo 2 (tamano_muestra.csv)",
              str(tam["Líneas (filas) de entrenamiento"]).replace(".", "") == str(ctx["train_lineas"])
              and str(tam["Pedidos distintos"]).replace(".", "") == str(ctx["pedidos_train"]))
    comprobar("Variables originales = 53 (diccionario) y candidatas = auditoría de fuga",
              cal["variables"]["originales"] == 53 and cal["variables"]["candidatas_auditoria"]
              == int(pd.read_csv(TABLAS / "auditoria_fuga.csv")["decisión"].str.startswith("Candidata").sum()))
    comprobar("Pedidos compartidos entre entrenamiento y prueba = 0", cal["integridad"]["pedidos_compartidos_train_test"] == 0)
    n_pedidos_test = p["order_id"].nunique()
    comprobar("Pedidos de la prueba (predicciones) = contexto.json", n_pedidos_test == ctx["modelado_pedidos_test"])

    # 7. Tablas copiadas del Jupyter Book sin cambios
    ok = True
    for nombre in ["con_fuga_vs_sin_fuga.csv", "auc_univariado_todas.csv", "techo_sin_fuga.csv", "moran_global.csv", "curva_aprendizaje.csv",
                   "seleccion_conjuntos.csv", "diagnostico_residuos.csv"]:
        ok &= pd.read_csv(TABLAS / nombre).equals(pd.read_csv(DATOS / nombre))
    comprobar("Tablas copiadas del libro (fuga, curva de aprendizaje, selección, Moran...) idénticas", bool(ok))

    # 8. El umbral se explora SOLO con OOF y nunca con la prueba
    oof = F.predicciones_oof()
    comprobar("OOF: todas las fechas son anteriores al gap (no se mezcla con la prueba)",
              bool(oof["order_date"].max() < pd.Timestamp(ctx["inicio_gap"])) and not set(oof["order_id"]) & set(p["order_id"]))
    m_oof = F.metricas_umbral(F.MODELO_ELEGIDO, 0.5)                    # fuente por defecto
    comprobar("metricas_umbral() usa OOF por defecto (n = líneas OOF, no las de la prueba)", m_oof["n"] == ctx["oof_lineas"] != len(p))
    a, b = F.metricas_umbral(F.MODELO_ELEGIDO, 0.3, "test"), F.metricas_umbral(F.MODELO_ELEGIDO, 0.9, "test")
    comprobar("ROC-AUC no depende del umbral", _cercano(a["ROC-AUC"], b["ROC-AUC"]))
    comprobar("Umbral final fijado de antemano = 0,50", F.umbral_final() == 0.5)

    # 9. Calibración sin partir empates
    ok, detalle = True, []
    for m in [x for x in F.modelos() if not F.es_dummy(x)]:
        g = F._bins_calibracion(p[f"p::{m}"], p["y"])
        unicos = p[f"p::{m}"].round(6).nunique()
        if unicos <= 20:
            ok &= len(g) == unicos
        ok &= int(g["n"].sum()) == len(p)
        detalle.append(f"{m}: {unicos} probabilidades únicas → {len(g)} grupos")
    comprobar("Calibración: una probabilidad única = un grupo (no se parten empates); n suma el total", bool(ok), "; ".join(detalle))

    return pd.DataFrame(filas)


if __name__ == "__main__":
    tabla = verificar()
    print(tabla.to_string(index=False))
    sys.exit(0 if (tabla["resultado"] == "OK").all() else 1)
