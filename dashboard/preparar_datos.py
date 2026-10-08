"""Prepara los datos agregados que usa el dashboard (Entregable 2).

Se ejecuta UNA vez (≈ 3 min) desde la raíz del proyecto:

    python dashboard/preparar_datos.py

Lee el CSV original y las tablas del Jupyter Book (carpeta ``tables/``), ajusta los
modelos base y guarda en ``dashboard/datos/`` archivos pequeños (CSV y JSON). El
dashboard solo lee esos archivos, así que arranca rápido y con poca memoria.

Reglas que se respetan (las mismas del Entregable 1):
- el EDA usa SOLO el entrenamiento; la prueba solo aparece en la pestaña de modelos;
- partición cronológica con gap de 7 días y validación cruzada temporal;
- población de modelado sin pedidos cancelados ni sospechosos de fraude;
- unidad de observación: la LÍNEA de pedido (cap. 1–2); los IC se remuestrean por ``order_id``;
- el umbral NO se explora con la prueba: ``predicciones_oof_train.csv.gz`` guarda las predicciones
  fuera de pliegue (OOF) del entrenamiento con los mismos pliegues temporales de la validación cruzada,
  y la prueba solo se evalúa con el umbral fijado de antemano (``UMBRAL_FINAL``).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project_utils import (CUSTOMER_COL, DATE_COL, GAP_DAYS, LAT, LON, ORDER_COL, RANDOM_STATE, TARGET,  # noqa: E402
                               classification_metrics, cluster_bootstrap, load_test, load_train,
                               make_model_pipeline, rate_table, spatial_cells, split_boundaries,
                               temporal_folds, load_gap)

warnings.filterwarnings("ignore")
OUT = ROOT / "dashboard" / "datos"
OUT.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / "tables"

CAT_EDA = ["shipping_mode", "payment_type", "market", "order_region", "customer_segment", "department_name",
           "category_name", "order_weekend", "order_afternoon"]
NUM_EDA = ["sales", "order_item_quantity", "order_item_discount_rate", "order_item_profit_ratio", "items_in_order",
           "orders_prev_7d", "order_hour", "days_shipping_scheduled"]
MODELO_DUMMY = "Dummy — clase mayoritaria / prior"      # predice la clase mayoritaria; su probabilidad es la prevalencia
MODELOS = {
    MODELO_DUMMY: None,
    "Logística aditiva (modo + hora)": {"num_lin": ["order_hour_sin", "order_hour_cos"], "num_log": [], "cat": ["shipping_mode"]},
    "Logística modo × jornada": {"num_lin": [], "num_log": [], "cat": ["mode_x_daypart"]},
}
GRID_C = [0.01, 0.1, 1.0, 10.0]
UMBRAL_FINAL = 0.5          # umbral fijado ANTES de mirar la prueba (el mismo del capítulo 11)
B_BOOTSTRAP = 1000          # remuestreos de pedidos: el mismo B del capítulo 11, para que los IC coincidan con los del Jupyter Book


def guardar(df: pd.DataFrame, nombre: str) -> None:
    df.to_csv(OUT / nombre, index=False)


def resumen_calidad(train_all: pd.DataFrame, test_all: pd.DataFrame, gap: pd.DataFrame) -> dict:
    """Resumen de calidad de datos para la subpestaña «Objetivo y calidad».

    Reutiliza las tablas YA calculadas en los capítulos 2 y 9 (faltantes, duplicados, inconsistencias,
    auditoría de fuga) y vuelve a comprobar con los datos que no hay pedidos compartidos entre train y test.
    """
    falt = pd.read_csv(TABLES / "faltantes_por_columna.csv", index_col=0)
    dup = pd.read_csv(TABLES / "auditoria_duplicados.csv").set_index("comprobación")["resultado"]
    inc = pd.read_csv(TABLES / "valores_inconsistentes.csv")
    fuga = pd.read_csv(TABLES / "auditoria_fuga.csv")
    cols_dup = pd.read_csv(TABLES / "columnas_duplicadas.csv")
    tam = pd.read_csv(TABLES / "tamano_muestra.csv").set_index("medida")["valor"]

    compartidos = len(set(train_all[ORDER_COL]) & set(test_all[ORDER_COL]))
    compartidos_gap = len(set(train_all[ORDER_COL]) & set(gap[ORDER_COL]))
    lineas_comp = len(set(train_all["order_item_id"]) & set(test_all["order_item_id"]))
    previo = float(dup["pedidos en entrenamiento y prueba"])
    if int(previo) != compartidos:
        print(f"AVISO: auditoria_duplicados.csv dice {previo:.0f} pedidos compartidos y ahora se obtienen {compartidos}")
    hallazgos = inc[inc["casos"] > 0]
    sospecha = inc[inc["comprobación"].str.startswith("Tardía según días reales")].iloc[0]
    alto = fuga[fuga["riesgo_de_fuga"] == "Alto"]
    posteriores = fuga[fuga["disponible_al_predecir"] == "No"]
    return {
        "faltantes": {"n_variables": int(len(falt)), "variable_max": str(falt["faltantes_%"].idxmax()),
                      "pct_max": float(falt["faltantes_%"].max()),
                      "detalle": {str(k): float(v) for k, v in falt["faltantes_%"].items()}},
        "integridad": {"pedidos_compartidos_train_test": int(compartidos), "pedidos_compartidos_train_gap": int(compartidos_gap),
                       "lineas_compartidas_train_test": int(lineas_comp),
                       "dias_entre_train_y_test": float(dup["pedido más reciente de train vs más antiguo de test (días)"]),
                       "columnas_duplicadas": int(len(cols_dup)),
                       "clientes_en_ambos_conjuntos": int(dup["clientes con pedidos en ambos (esperado, no se usan como predictor)"]),
                       "pct_validacion_con_pedido_en_train_si_fuera_aleatoria": float(
                           dup["% de líneas de validación cuyo pedido estaría en train con partición ALEATORIA por filas"])},
        "inconsistencias": {"comprobaciones": int(len(inc)), "con_hallazgos": int(len(hallazgos)),
                            "lineas_tardias_por_dias_con_etiqueta_0": int(sospecha["casos"]),
                            "detalle": {str(r["comprobación"]): int(r["casos"]) for _, r in hallazgos.iterrows()}},
        "fuga": {"alto_riesgo": [str(v) for v in alto[alto["variable"] != TARGET]["variable"]],
                 "posteriores_al_pedido": [str(v) for v in posteriores[posteriores["variable"] != TARGET]["variable"]],
                 "n_variables_auditadas": int(len(fuga)),
                 "disponibles_al_pedido": [str(v) for v in fuga[(fuga["disponible_al_predecir"].str.startswith("Sí"))
                                                              & (fuga["decisión"].str.startswith("Candidata"))]["variable"]]},
        "variables": {"originales": int(len(pd.read_csv(TABLES / "diccionario_datos.csv").query("nombre_original != '(derivada)'"))),
                      "candidatas_originales": int(str(tam["Predictoras candidatas originales (sin objetivo, fuga, ids ni datos personales)"]).replace(".", "")),
                      "candidatas_auditoria": int(fuga["decisión"].str.startswith("Candidata").sum())},
    }


def main() -> None:
    train_all = load_train(include_canceled=True)          # EDA: entrenamiento completo
    gap = load_gap()
    test_all = load_test(include_canceled=True)
    gap_start, cutoff = split_boundaries()
    y_all = train_all[TARGET]

    # ------------------------------------------------------------------ contexto
    contexto = {
        "lineas": int(len(train_all) + len(gap) + len(test_all)),
        "pedidos": int(pd.concat([train_all[ORDER_COL], gap[ORDER_COL], test_all[ORDER_COL]]).nunique()),
        "clientes": int(pd.concat([train_all[CUSTOMER_COL], gap[CUSTOMER_COL], test_all[CUSTOMER_COL]]).nunique()),
        "desde": f"{train_all[DATE_COL].min():%Y-%m-%d}", "hasta": f"{test_all[DATE_COL].max():%Y-%m-%d}",
        "train_lineas": int(len(train_all)), "gap_lineas": int(len(gap)), "test_lineas": int(len(test_all)),
        "inicio_gap": f"{gap_start:%Y-%m-%d}", "inicio_test": f"{cutoff:%Y-%m-%d}",
        "tasa_tardia_train": float(y_all.mean()),
        "paises_destino": int(train_all["order_country"].nunique()),
    }
    contexto["gap_dias"] = int(GAP_DAYS)
    contexto["pedidos_train"] = int(train_all[ORDER_COL].nunique())
    contexto["pedidos_test"] = int(test_all[ORDER_COL].nunique())
    contexto["umbral_final"] = UMBRAL_FINAL
    calidad = resumen_calidad(train_all, test_all, gap)
    (OUT / "calidad_datos.json").write_text(json.dumps(calidad, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Pedidos compartidos train/test:", calidad["integridad"]["pedidos_compartidos_train_test"])
    # Volumen semanal en todo el periodo (sin etiquetas): ilustra la partición sin mirar la prueba.
    todo = pd.concat([train_all.assign(conjunto="Entrenamiento"), gap.assign(conjunto="Gap (descartado)"),
                      test_all.assign(conjunto="Prueba")])
    vol = (todo.groupby([todo[DATE_COL].dt.to_period("W").dt.start_time, "conjunto"]).size()
           .rename("lineas").reset_index().rename(columns={DATE_COL: "semana"}))
    guardar(vol, "volumen_semanal.csv")
    del todo

    # ------------------------------------------------------------------ EDA (solo entrenamiento)
    dist = (train_all.groupby(["delivery_status", TARGET]).size().rename("lineas").reset_index())
    guardar(dist, "objetivo_por_estado.csv")
    tasas = []
    for col in CAT_EDA:
        t = rate_table(train_all[col].astype(str), y_all, min_n=200)
        tasas.append(t.assign(variable=col))
    guardar(pd.concat(tasas, ignore_index=True), "tasas_por_categoria.csv")
    muestra = train_all.sample(15000, random_state=RANDOM_STATE)[NUM_EDA + [TARGET]]
    muestra.to_csv(OUT / "muestra_numericas.csv.gz", index=False, compression="gzip")
    hm = train_all.pivot_table(index="shipping_mode", columns="order_hour", values=TARGET, aggfunc="mean")
    hm.to_csv(OUT / "hora_por_modo.csv")
    sem = (train_all.groupby([train_all[DATE_COL].dt.to_period("W").dt.start_time, "market"])[TARGET]
           .agg(["mean", "size"]).reset_index().rename(columns={DATE_COL: "semana", "mean": "tasa", "size": "n"}))
    sem_tot = (train_all.groupby(train_all[DATE_COL].dt.to_period("W").dt.start_time)[TARGET]
               .agg(["mean", "size"]).reset_index().rename(columns={DATE_COL: "semana", "mean": "tasa", "size": "n"})
               .assign(market="Todos"))
    guardar(pd.concat([sem_tot, sem], ignore_index=True), "serie_semanal_mercado.csv")
    zona = train_all[(train_all[LAT].between(17.8, 50)) & (train_all[LON].between(-125, -65))].copy()
    zona["celda"] = spatial_cells(zona[LAT], zona[LON]).to_numpy()
    cel = zona.groupby("celda").agg(n=(TARGET, "size"), tasa=(TARGET, "mean"), lat=(LAT, "mean"), lon=(LON, "mean"))
    guardar(cel[cel["n"] >= 30].reset_index(), "mapa_celdas.csv")
    for nombre in ["con_fuga_vs_sin_fuga.csv", "auc_univariado_todas.csv", "techo_sin_fuga.csv", "moran_global.csv",
                   "estacionariedad.csv", "ljung_box_tasa.csv", "mercado_por_trimestre.csv", "seleccion_conjuntos.csv",
                   "decisiones_preprocesamiento.csv", "diccionario_datos.csv", "curva_aprendizaje.csv",
                   "diagnostico_residuos.csv"]:
        pd.read_csv(TABLES / nombre).to_csv(OUT / nombre, index=False)

    # ------------------------------------------------------------------ modelos
    train = load_train(include_canceled=False)
    test = load_test(include_canceled=False)
    ytr, yte = train[TARGET], test[TARGET]
    folds = temporal_folds(train[DATE_COL])
    filas_cv, filas_test, ic_rows, coef_rows = [], [], [], []
    # Predicciones fuera de pliegue (OOF) del entrenamiento: se usan para explorar el umbral SIN mirar la prueba.
    oof = train[[ORDER_COL, DATE_COL]].copy()
    oof["y"] = ytr.to_numpy()
    oof["pliegue"] = 0
    for k, (_, va) in enumerate(folds, start=1):
        oof.iloc[va, oof.columns.get_loc("pliegue")] = k
    preds = test[[ORDER_COL, DATE_COL, "shipping_mode", "mode_x_daypart", "market", "customer_segment", "order_hour"]].copy()
    preds["periodo"] = np.where(test[DATE_COL] >= "2017-10-01", "desde oct-2017", "abr–sep 2017")
    preds["y"] = yte.to_numpy()
    for nombre, cols in MODELOS.items():
        if cols is None:
            fabrica, X_tr, X_te, C = (lambda: DummyClassifier(strategy="prior")), train[["shipping_mode"]], test[["shipping_mode"]], None
        else:
            todas = cols["num_lin"] + cols["num_log"] + cols["cat"]
            X_tr, X_te = train[todas], test[todas]
            res_c = {}
            for c in GRID_C:
                s = [roc_auc_score(ytr.iloc[va], make_model_pipeline(cols["num_lin"], cols["num_log"], cols["cat"], C=c)
                                   .fit(X_tr.iloc[tr], ytr.iloc[tr]).predict_proba(X_tr.iloc[va])[:, 1]) for tr, va in folds]
                res_c[c] = float(np.mean(s))
            C = max(res_c, key=res_c.get)
            fabrica = (lambda cols=cols, C=C: make_model_pipeline(cols["num_lin"], cols["num_log"], cols["cat"], C=C))
        cv_m = []
        p_oof = np.full(len(train), np.nan)
        for tr, va in folds:
            m = fabrica().fit(X_tr.iloc[tr], ytr.iloc[tr])
            p = m.predict_proba(X_tr.iloc[va])[:, 1]
            p_oof[va] = p
            cv_m.append(classification_metrics(ytr.iloc[va], (p >= UMBRAL_FINAL).astype(int), p))
        oof[f"p::{nombre}"] = p_oof
        cv_m = pd.DataFrame(cv_m)
        for met in cv_m.columns:
            filas_cv.append({"modelo": nombre, "métrica": met, "media": cv_m[met].mean(), "desv": cv_m[met].std()})
        modelo = fabrica().fit(X_tr, ytr)
        p_te = modelo.predict_proba(X_te)[:, 1]
        pred = (p_te >= UMBRAL_FINAL).astype(int) if cols is not None else modelo.predict(X_te)
        preds[f"p::{nombre}"] = p_te
        filas_test.append({"modelo": nombre, "C": C, **classification_metrics(yte, pred, p_te)})
        ic = cluster_bootstrap(yte, pred, p_te, test[ORDER_COL], B=B_BOOTSTRAP)
        ic_rows.append(ic.reset_index().assign(modelo=nombre))
        if cols is not None:
            names = modelo.named_steps["prep"].get_feature_names_out()
            coef = modelo.named_steps["clf"].coef_.ravel()
            coef_rows.append(pd.DataFrame({"modelo": nombre, "variable": names, "coeficiente": coef,
                                           "odds_ratio": np.exp(coef)}))
            print(f"{nombre}: C = {C}, ROC-AUC prueba = {filas_test[-1]['ROC-AUC']:.4f}")
    guardar(pd.DataFrame(filas_cv), "modelos_cv.csv")
    guardar(pd.DataFrame(filas_test), "modelos_prueba.csv")
    guardar(pd.concat(ic_rows, ignore_index=True), "modelos_ic.csv")
    guardar(pd.concat(coef_rows, ignore_index=True), "modelos_coeficientes.csv")
    preds.to_csv(OUT / "predicciones_prueba.csv.gz", index=False, compression="gzip")
    oof = oof[oof["pliegue"] > 0].reset_index(drop=True)          # el primer 30 % del entrenamiento nunca se valida
    oof.to_csv(OUT / "predicciones_oof_train.csv.gz", index=False, compression="gzip")
    print(f"OOF: {len(oof):,} de {len(train):,} líneas de entrenamiento tienen predicción fuera de pliegue")
    # Referencia empírica sin fuga: tasas por modo × jornada del ENTRENAMIENTO evaluadas en la prueba (no es un modelo nuevo).
    tasa_celda = train.groupby("mode_x_daypart")[TARGET].mean()
    p_ref = test["mode_x_daypart"].map(tasa_celda).fillna(ytr.mean()).to_numpy()
    guardar(pd.DataFrame([{"referencia": "Tabla de tasas modo × jornada (tasas del entrenamiento, evaluada en la prueba)",
                           **classification_metrics(yte, (p_ref >= UMBRAL_FINAL).astype(int), p_ref)}]), "referencia_empirica.csv")
    celda = (train.groupby("mode_x_daypart")[TARGET].agg(["mean", "size"]).rename(columns={"mean": "tasa_train", "size": "lineas_train"})
             .join(preds.groupby("mode_x_daypart")[f"p::{list(MODELOS)[-1]}"].mean().rename("p_predicha")))
    guardar(celda.reset_index(), "probabilidad_por_celda.csv")
    contexto["modelado_train"], contexto["modelado_test"] = int(len(train)), int(len(test))
    contexto["modelado_pedidos_train"], contexto["modelado_pedidos_test"] = int(train[ORDER_COL].nunique()), int(test[ORDER_COL].nunique())
    contexto["oof_lineas"] = int(len(oof))
    contexto["tasa_tardia_modelado"] = float(ytr.mean())
    (OUT / "contexto.json").write_text(json.dumps(contexto, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Datos del dashboard guardados en", OUT)


if __name__ == "__main__":
    main()
