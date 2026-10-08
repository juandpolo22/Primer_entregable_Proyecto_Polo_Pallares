"""Figuras Plotly del dashboard DataCo (sin fuga).

Las MISMAS funciones las usan:
- ``dashboard/app.py`` (Dash, interactivo),
- ``dashboard/exportar_html.py`` (vista previa estática),
- los capítulos 13–15 del Jupyter Book (informe con interpretaciones).

Así cada figura del dashboard y su interpretación en el libro muestran exactamente lo mismo.
Todas las figuras leen archivos pequeños de ``dashboard/datos/`` (ver ``preparar_datos.py``).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn import metrics as skm

DATOS = Path(__file__).resolve().parent / "datos"

# ---------------------------------------------------------------------------
# Paleta (validada para daltonismo: azul / naranja / aqua; rojo-azul para divergente)
# ---------------------------------------------------------------------------
AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
ROJO = "#e34948"
TEXTO, TEXTO_2, SUPERFICIE, REJILLA = "#0b0b0b", "#52514e", "#fcfcfb", "#e7e6e2"
COLOR_CLASE = {"no tardía": AZUL, "tardía": ROJO}
MODELO_DUMMY = "Dummy — clase mayoritaria / prior"      # predice la clase mayoritaria; su probabilidad es la prevalencia
COLOR_MODELO = {MODELO_DUMMY: GRIS, "Logística aditiva (modo + hora)": NARANJA, "Logística modo × jornada": AZUL}
COLOR_CONJUNTO = {"Entrenamiento": AZUL, "Gap (descartado)": GRIS, "Prueba": NARANJA}
# Origen de los datos de cada resultado (mismos colores que la partición): TRAIN, OOF (fuera de pliegue en train) y TEST.
COLOR_FUENTE = {"TRAIN": AZUL, "OOF": AQUA, "TEST": NARANJA}
DIVERGENTE = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
MODELO_ELEGIDO = "Logística modo × jornada"

ETIQUETAS = {
    "shipping_mode": "Modo de envío", "payment_type": "Medio de pago", "market": "Mercado de destino",
    "order_region": "Región de destino", "customer_segment": "Segmento del cliente", "department_name": "Departamento",
    "category_name": "Categoría de producto", "order_weekend": "Fin de semana (1 = sí)",
    "order_afternoon": "Pedido desde las 12:00 (1 = sí)", "sales": "Venta de la línea (USD)",
    "order_item_quantity": "Unidades", "order_item_discount_rate": "Tasa de descuento",
    "order_item_profit_ratio": "Margen de ganancia", "items_in_order": "Líneas del pedido",
    "orders_prev_7d": "Pedidos de los 7 días previos", "order_hour": "Hora del pedido",
    "days_shipping_scheduled": "Días de envío programados", "periodo": "Periodo de la prueba",
}


def _estilo(fig: go.Figure, titulo: str, alto: int = 380) -> go.Figure:
    fig.update_layout(
        title={"text": titulo, "x": 0.0, "xanchor": "left", "font": {"size": 15, "color": TEXTO}},
        template="plotly_white", height=alto, margin={"l": 60, "r": 20, "t": 60, "b": 50},
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "size": 12, "color": TEXTO_2},
        paper_bgcolor=SUPERFICIE, plot_bgcolor=SUPERFICIE,
        hoverlabel={"bgcolor": "white", "font_size": 12},
    )
    if fig.layout.legend.orientation is None:          # no pisa una leyenda ya configurada
        fig.update_layout(legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0, "title": None})
    fig.update_xaxes(gridcolor=REJILLA, zeroline=False, linecolor=REJILLA)
    fig.update_yaxes(gridcolor=REJILLA, zeroline=False, linecolor=REJILLA)
    return fig


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f} %".replace(".", ",")


def num(x: float, d: int = 3) -> str:
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


# ---------------------------------------------------------------------------
# Lectura de datos (en caché)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def contexto() -> dict:
    return json.loads((DATOS / "contexto.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def tabla(nombre: str) -> pd.DataFrame:
    return pd.read_csv(DATOS / nombre)


@lru_cache(maxsize=None)
def predicciones() -> pd.DataFrame:
    return pd.read_csv(DATOS / "predicciones_prueba.csv.gz", parse_dates=["order_date"])


@lru_cache(maxsize=None)
def predicciones_oof() -> pd.DataFrame:
    """Predicciones fuera de pliegue (OOF) del entrenamiento con los pliegues temporales de la CV."""
    return pd.read_csv(DATOS / "predicciones_oof_train.csv.gz", parse_dates=["order_date"])


@lru_cache(maxsize=None)
def calidad() -> dict:
    return json.loads((DATOS / "calidad_datos.json").read_text(encoding="utf-8"))


def es_dummy(modelo: str) -> bool:
    return modelo.startswith("Dummy")


def modelos() -> list[str]:
    return [c.split("::", 1)[1] for c in predicciones().columns if c.startswith("p::")]


# ---------------------------------------------------------------------------
# Pestaña 1: contexto
# ---------------------------------------------------------------------------
def fig_particion() -> go.Figure:
    v = tabla("volumen_semanal.csv")
    fig = go.Figure()
    for conj, color in COLOR_CONJUNTO.items():
        d = v[v["conjunto"] == conj]
        fig.add_bar(x=d["semana"], y=d["lineas"], name=conj, marker_color=color, marker_line_width=0,
                    hovertemplate="Semana %{x|%d-%b-%Y}<br>%{y:,} líneas<extra>" + conj + "</extra>")
    c = contexto()
    fig.add_vline(x=c["inicio_test"], line_dash="dot", line_color=TEXTO_2)
    fig.add_annotation(x=c["inicio_test"], y=1.0, yref="paper", text="inicio de la prueba", showarrow=False,
                       xanchor="left", font={"size": 11, "color": TEXTO_2})
    fig.update_layout(barmode="stack", bargap=0.05, xaxis_title=None, yaxis_title="líneas de pedido por semana")
    return _estilo(fig, "Partición cronológica: entrenamiento, gap de 7 días y prueba", 330)


# ---------------------------------------------------------------------------
# Pestaña 2: EDA (solo entrenamiento)
# ---------------------------------------------------------------------------
def fig_objetivo() -> go.Figure:
    """Distribución del objetivo por estado de entrega (entrenamiento completo).
    Los cancelados/sospechosos de fraude tienen etiqueta 0 pero NO son entregas a tiempo: se muestran aparte (gris)."""
    d = tabla("objetivo_por_estado.csv").copy()
    d["clase"] = d["late_delivery_risk"].map({0: "no tardía", 1: "tardía"})
    orden = d.groupby("delivery_status")["lineas"].sum().sort_values().index
    cancelado = d["delivery_status"].str.contains("cancel", case=False)
    grupos = [("Late_delivery_risk = 1 (tardía)", d[d["clase"] == "tardía"], ROJO),
              ("Late_delivery_risk = 0 (no tardía: adelantada o a tiempo)", d[(d["clase"] == "no tardía") & ~cancelado], AZUL),
              ("Late_delivery_risk = 0 pero cancelado/fraude (no entregado; excluido del modelado)", d[(d["clase"] == "no tardía") & cancelado], GRIS)]
    fig = go.Figure()
    for nombre, g, color in grupos:
        s_ = g.set_index("delivery_status").reindex(orden)
        fig.add_bar(y=orden, x=s_["lineas"], name=nombre, orientation="h", marker_color=color,
                    hovertemplate="%{y}<br>%{x:,} líneas<extra>" + nombre + "</extra>")
    total = d["lineas"].sum()
    tasa = d.loc[d["clase"] == "tardía", "lineas"].sum() / total
    fig.update_layout(barmode="stack", xaxis_title="líneas de entrenamiento", yaxis_title=None,
                      legend={"orientation": "h", "x": 0, "y": -0.28, "xanchor": "left", "yanchor": "top"})
    return _estilo(fig, f"Objetivo: {pct(tasa)} de líneas tardías (entrenamiento completo, con cancelados) — origen en el estado de entrega", 360)


def fig_tasa_categoria(variable: str = "shipping_mode") -> go.Figure:
    t = tabla("tasas_por_categoria.csv")
    t = t[t["variable"] == variable].sort_values("tasa")
    global_ = contexto()["tasa_tardia_train"]
    fig = go.Figure(go.Bar(
        y=t["categoria"].astype(str), x=100 * t["tasa"], orientation="h", marker_color=AZUL,
        error_x={"type": "data", "symmetric": False, "array": 100 * (t["ic_sup"] - t["tasa"]),
                 "arrayminus": 100 * (t["tasa"] - t["ic_inf"]), "color": TEXTO_2, "thickness": 1.2},
        customdata=np.c_[t["n"], 100 * t["ic_inf"], 100 * t["ic_sup"]],
        hovertemplate="%{y}<br>%{x:.1f} % tardías (IC 95 %: %{customdata[1]:.1f}–%{customdata[2]:.1f})"
                      "<br>%{customdata[0]:,} líneas<extra></extra>"))
    fig.add_vline(x=100 * global_, line_dash="dash", line_color=GRIS)
    fig.add_annotation(x=100 * global_, y=0, yref="paper", yshift=-28, text=f"global {pct(global_)}", showarrow=False,
                       font={"size": 11, "color": TEXTO_2})
    fig.update_layout(xaxis_title="% de entregas tardías (IC Wilson 95 %)", yaxis_title=None, showlegend=False)
    alto = max(300, 32 * len(t) + 120)
    return _estilo(fig, f"Tasa de entregas tardías por {ETIQUETAS.get(variable, variable).lower()}", alto)


def fig_numerica(variable: str = "sales") -> go.Figure:
    m = pd.read_csv(DATOS / "muestra_numericas.csv.gz")
    fig = go.Figure()
    for val, clase in [(0, "no tardía"), (1, "tardía")]:
        fig.add_histogram(x=m.loc[m["late_delivery_risk"] == val, variable], name=clase, histnorm="percent",
                          marker_color=COLOR_CLASE[clase], opacity=0.6, nbinsx=40,
                          hovertemplate="%{x}<br>%{y:.1f} % de las líneas<extra>" + clase + "</extra>")
    fig.update_layout(barmode="overlay", xaxis_title=ETIQUETAS.get(variable, variable), yaxis_title="% de líneas de la clase")
    return _estilo(fig, f"Distribución según el objetivo: {ETIQUETAS.get(variable, variable)} (muestra de 15.000)", 340)


def fig_hora_modo() -> go.Figure:
    hm = pd.read_csv(DATOS / "hora_por_modo.csv", index_col=0)
    z = 100 * hm.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=[int(float(c)) for c in hm.columns], y=hm.index, colorscale=DIVERGENTE, zmid=contexto()["tasa_tardia_train"] * 100,
        zmin=0, zmax=100, text=np.round(z).astype(int), texttemplate="%{text}", textfont={"size": 10},
        colorbar={"title": "% tardías", "thickness": 12}, xgap=2, ygap=2,
        hovertemplate="%{y}, %{x}:00 h<br>%{z:.1f} % tardías<extra></extra>"))
    fig.update_layout(xaxis_title="hora del pedido", yaxis_title=None, xaxis={"dtick": 1})
    return _estilo(fig, "Hora del pedido × modo de envío: el corte del mediodía en Same Day", 320)


def fig_serie(mercado: str = "Todos") -> go.Figure:
    s = tabla("serie_semanal_mercado.csv")
    s = s[(s["market"] == mercado) & (s["n"] >= 30)]
    global_ = contexto()["tasa_tardia_train"]
    fig = go.Figure(go.Scatter(x=s["semana"], y=100 * s["tasa"], mode="lines", line={"color": AZUL, "width": 2},
                               customdata=s["n"], name=mercado,
                               hovertemplate="Semana %{x|%d-%b-%Y}<br>%{y:.1f} % tardías<br>%{customdata:,} líneas<extra></extra>"))
    fig.add_hline(y=100 * global_, line_dash="dash", line_color=GRIS)
    fig.update_layout(xaxis_title=None, yaxis_title="% tardías por semana", yaxis_range=[30, 80], hovermode="x unified",
                      showlegend=False)
    return _estilo(fig, f"Serie semanal de la tasa de entregas tardías — {mercado} (entrenamiento)", 330)


def fig_mapa() -> go.Figure:
    """Celdas de 1° sobre ejes longitud/latitud (sin mapa base: no depende de servicios externos)."""
    c = tabla("mapa_celdas.csv")
    global_ = contexto()["tasa_tardia_train"]
    fig = go.Figure()
    for nombre, d in [("EE. UU. continental", c[c["lat"] > 20]), ("Puerto Rico", c[c["lat"] <= 20])]:
        fig.add_scatter(
            x=d["lon"], y=d["lat"], mode="markers", name=nombre, showlegend=False,
            marker={"size": np.clip(np.sqrt(d["n"]) / 2.2, 5, 30), "color": 100 * d["tasa"], "colorscale": DIVERGENTE,
                    "cmid": 100 * global_, "cmin": 40, "cmax": 70, "line": {"width": 1, "color": "white"},
                    "colorbar": {"title": "% tardías", "thickness": 12}, "showscale": nombre != "Puerto Rico"},
            customdata=np.c_[d["n"], d["celda"]],
            hovertemplate="Celda %{customdata[1]}<br>%{marker.color:.1f} % tardías<br>%{customdata[0]:,} líneas<extra></extra>")
    for ciudad, lat, lon in [("Nueva York", 40.7, -74.0), ("Los Ángeles", 34.05, -118.25), ("Chicago", 41.9, -87.6),
                             ("Houston", 29.8, -95.4), ("Caguas (PR)", 18.23, -66.04)]:
        fig.add_annotation(x=lon, y=lat, text=ciudad, showarrow=False, yshift=16, font={"size": 10, "color": TEXTO_2})
    fig.update_xaxes(title="longitud", range=[-126, -63])
    fig.update_yaxes(title="latitud", range=[16, 50])
    return _estilo(fig, "Tasa de entregas tardías por celda de 1° (ubicación del cliente; tamaño ∝ √líneas)", 460)


def _posteriores() -> set:
    """Variables que solo se conocen DESPUÉS de crear el pedido: las del cap. 9 y las derivadas de ellas."""
    base = set(calidad()["fuga"]["posteriores_al_pedido"])
    derivadas = {"retraso_real_menos_programado", "horas_hasta_despacho"}      # calculadas con días reales / fecha de despacho
    return base | derivadas


def fig_fuga() -> go.Figure:
    """AUC univariado: rojo = posterior al pedido (fuga); azul = disponible al crear el pedido."""
    todas = tabla("auc_univariado_todas.csv")
    post = _posteriores()
    oficiales = [v for v in calidad()["fuga"]["posteriores_al_pedido"] if v in set(todas["variable"])]
    a = pd.concat([todas.head(12), todas[todas["variable"].isin(oficiales)]]).drop_duplicates("variable")
    a = a.sort_values("AUC univariado").reset_index(drop=True)
    fig = go.Figure()
    for etiqueta, es_post, color in [("Disponible al crear el pedido", False, AZUL), ("Posterior al pedido (fuga)", True, ROJO)]:
        d = a[a["variable"].isin(post) == es_post]
        fig.add_bar(y=d["variable"], x=d["AUC univariado"], orientation="h", name=etiqueta, marker_color=color,
                    hovertemplate="%{y}<br>AUC univariado %{x:.3f}<extra>" + etiqueta + "</extra>")
    fig.add_vline(x=0.5, line_dash="dash", line_color=GRIS)
    fig.update_layout(xaxis_title="AUC univariado (0,5 = sin señal)", yaxis_title=None, xaxis_range=[0.45, 1.02],
                      yaxis={"categoryorder": "array", "categoryarray": list(a["variable"])}, barmode="overlay")
    return _estilo(fig, "Auditoría de fuga: señal de cada variable por sí sola", 430)


def fig_con_sin_fuga() -> go.Figure:
    d = tabla("con_fuga_vs_sin_fuga.csv")
    ref = tabla("techo_sin_fuga.csv").iloc[0]
    nombres = list(d["escenario"]) + ["Referencia empírica sin fuga: tabla modo × jornada"]
    valores = list(d.iloc[:, 1]) + [ref["ROC-AUC"]]
    colores = [ROJO if n.startswith("Con fuga") else AZUL for n in nombres]
    fig = go.Figure(go.Bar(x=valores, y=nombres, orientation="h", marker_color=colores,
                           text=[num(v) for v in valores], textposition="outside",
                           hovertemplate="%{y}<br>ROC-AUC %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="ROC-AUC (CV temporal; referencia: entrenamiento)", yaxis_title=None,
                      xaxis_range=[0.5, 1.05])
    return _estilo(fig, "Comparación metodológica: con fuga vs sin fuga", 300)


# ---------------------------------------------------------------------------
# Pestaña 3: modelos base (prueba cronológica)
# ---------------------------------------------------------------------------
METRICAS_TABLA = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy", "Average Precision", "Brier"]


def tabla_modelos() -> pd.DataFrame:
    """Métricas en la prueba con umbral fijado de antemano. Cada celda: estimación y, debajo, el IC 95 % (bootstrap por pedido)."""
    ic = tabla("modelos_ic.csv")
    cv = tabla("modelos_cv.csv")
    filas = []
    for m in modelos():
        i = ic[ic["modelo"] == m].set_index("Métrica")
        c = cv[cv["modelo"] == m].set_index("métrica")
        fila = {"Modelo": m}
        for met in METRICAS_TABLA:
            fila[met] = f"{num(i.loc[met, 'Estimación'])}\n[{num(i.loc[met].iloc[1])}–{num(i.loc[met].iloc[2])}]"
        fila["ROC-AUC CV temporal"] = f"{num(c.loc['ROC-AUC', 'media'])}\n± {num(c.loc['ROC-AUC', 'desv'])}"
        filas.append(fila)
    return pd.DataFrame(filas)


def _curvas_titulo(m: str, y, s) -> str:
    return f"{m} (ROC-AUC {num(skm.roc_auc_score(y, s))} · AP {num(skm.average_precision_score(y, s))})"


def fig_roc(seleccion: list[str] | None = None) -> go.Figure:
    p = predicciones()
    seleccion = seleccion or modelos()
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": GRIS, "width": 1},
                    name="azar (AUC 0,5)", hoverinfo="skip")
    for m in seleccion:
        s_ = p[f"p::{m}"]
        fpr, tpr, thr = skm.roc_curve(p["y"], s_)
        fig.add_scatter(x=fpr, y=tpr, mode="lines+markers", name=_curvas_titulo(m, p["y"], s_),
                        line={"color": COLOR_MODELO.get(m, AZUL), "width": 2}, marker={"size": 6},
                        customdata=np.clip(thr, 0, 1),
                        hovertemplate="FPR %{x:.3f} · TPR %{y:.3f}<br>umbral %{customdata:.3f}<extra>" + m + "</extra>")
    fig.update_layout(xaxis_title="tasa de falsos positivos", yaxis_title="tasa de verdaderos positivos (recall)",
                      legend={"orientation": "h", "x": 0, "y": -0.22, "xanchor": "left", "yanchor": "top"})
    return _estilo(fig, "Curva ROC en la prueba", 460)


def fig_pr(seleccion: list[str] | None = None) -> go.Figure:
    """Curva Precision–Recall en la prueba. La línea de referencia es la prevalencia (precisión de un clasificador sin información)."""
    p = predicciones()
    seleccion = seleccion or modelos()
    prev = float(p["y"].mean())
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[prev, prev], mode="lines", line={"dash": "dash", "color": GRIS, "width": 1},
                    name=f"azar (precisión = prevalencia {num(prev)})", hoverinfo="skip")
    for m in seleccion:
        s_ = p[f"p::{m}"]
        prec, rec, _ = skm.precision_recall_curve(p["y"], s_)
        fig.add_scatter(x=rec, y=prec, mode="lines+markers", line_shape="hv", name=_curvas_titulo(m, p["y"], s_),
                        line={"color": COLOR_MODELO.get(m, AZUL), "width": 2}, marker={"size": 6},
                        hovertemplate="recall %{x:.3f} · precisión %{y:.3f}<extra>" + m + "</extra>")
    fig.update_layout(xaxis_title="recall (tardías detectadas)", yaxis_title="precision (de lo marcado tardío, % que lo es)",
                      yaxis_range=[0, 1.02], xaxis_range=[0, 1.02],
                      legend={"orientation": "h", "x": 0, "y": -0.22, "xanchor": "left", "yanchor": "top"})
    return _estilo(fig, "Curva Precision–Recall en la prueba", 460)


def umbral_final() -> float:
    return float(contexto().get("umbral_final", 0.5))


def metricas_umbral(modelo: str = MODELO_ELEGIDO, umbral: float = 0.5, fuente: str = "oof") -> dict:
    """Métricas con un umbral. ``fuente='oof'`` (predicciones fuera de pliegue del entrenamiento) es la ÚNICA que se usa para
    explorar el umbral; ``fuente='test'`` solo se usa con el umbral fijado de antemano (``umbral_final()``)."""
    d = predicciones_oof() if fuente == "oof" else predicciones()
    s_ = d[f"p::{modelo}"].to_numpy()
    y = d["y"].to_numpy()
    pred = (s_ >= umbral).astype(int)
    return {"Accuracy": skm.accuracy_score(y, pred), "Precision": skm.precision_score(y, pred, zero_division=0),
            "Recall": skm.recall_score(y, pred, zero_division=0), "F1": skm.f1_score(y, pred, zero_division=0),
            "Balanced Accuracy": skm.balanced_accuracy_score(y, pred), "ROC-AUC": skm.roc_auc_score(y, s_),
            "marcados_tardía": float(pred.mean()), "n": int(len(y)), "cm": skm.confusion_matrix(y, pred, labels=[0, 1])}


def fig_confusion(modelo: str = MODELO_ELEGIDO, umbral: float = 0.5, fuente: str = "oof") -> go.Figure:
    m = metricas_umbral(modelo, umbral, fuente)
    cm = m["cm"]
    pct_ = cm / cm.sum(axis=1, keepdims=True)
    texto = [[f"{cm[i, j]:,}".replace(",", ".") + f"<br>{pct(pct_[i, j])}" for j in range(2)] for i in range(2)]
    color = COLOR_FUENTE["OOF" if fuente == "oof" else "TEST"]
    fig = go.Figure(go.Heatmap(z=pct_, x=["predicho: no tardía", "predicho: tardía"], y=["real: no tardía", "real: tardía"],
                               colorscale=[[0, "#f0efec"], [1, color]], zmin=0, zmax=1, showscale=False, text=texto,
                               texttemplate="%{text}", textfont={"size": 14}, xgap=3, ygap=3,
                               hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    origen = "OOF del entrenamiento" if fuente == "oof" else "prueba"
    return _estilo(fig, f"Matriz de confusión ({origen}) — umbral {num(umbral, 2)}", 340)


def _bins_calibracion(p: pd.Series, y: pd.Series, max_unicos: int = 20, n_bins: int = 10) -> pd.DataFrame:
    """Agrupa SIN partir empates: con pocas probabilidades distintas, un grupo por probabilidad predicha única;
    con muchas, deciles sobre los valores (``qcut`` con ``duplicates='drop'``), de modo que dos observaciones con
    exactamente la misma probabilidad siempre caen en el mismo grupo."""
    s_ = p.round(6)
    clave = s_ if s_.nunique() <= max_unicos else pd.qcut(s_, n_bins, duplicates="drop")
    g = (pd.DataFrame({"p": p.to_numpy(), "y": y.to_numpy(), "k": clave.to_numpy()})
         .groupby("k", observed=True).agg(pm=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index(drop=True))
    return g.sort_values("pm")


def fig_calibracion(seleccion: list[str] | None = None) -> go.Figure:
    p = predicciones()
    seleccion = [m for m in (seleccion or modelos()) if not es_dummy(m)]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": GRIS, "width": 1},
                    name="calibración perfecta", hoverinfo="skip")
    for m in seleccion:
        g = _bins_calibracion(p[f"p::{m}"], p["y"])
        agrupacion = "probabilidad única" if p[f"p::{m}"].round(6).nunique() <= 20 else "grupo de probabilidades"
        fig.add_scatter(x=g["pm"], y=g["obs"], mode="markers+lines", name=f"{m} ({len(g)} grupos)",
                        marker={"size": np.clip(np.sqrt(g["n"]) / 4.5, 8, 26), "color": COLOR_MODELO.get(m, AZUL),
                                "line": {"width": 2, "color": SUPERFICIE}},
                        line={"color": COLOR_MODELO.get(m, AZUL), "width": 1}, customdata=g["n"],
                        hovertemplate=agrupacion + "<br>probabilidad media predicha %{x:.3f}<br>frecuencia observada %{y:.3f}"
                                      "<br>n = %{customdata:,} líneas<extra>" + m + "</extra>")
    fig.update_layout(xaxis_title="probabilidad media predicha", yaxis_title="frecuencia observada de retraso",
                      xaxis_range=[-0.02, 1.02], yaxis_range=[-0.02, 1.02],
                      legend={"orientation": "h", "x": 0, "y": -0.2, "xanchor": "left", "yanchor": "top"})
    return _estilo(fig, "Calibración en la prueba (tamaño del punto ∝ √n; los empates no se parten)", 430)


def fig_celdas() -> go.Figure:
    c = tabla("probabilidad_por_celda.csv").sort_values("tasa_train")
    fig = go.Figure()
    fig.add_bar(y=c["mode_x_daypart"], x=100 * c["tasa_train"], orientation="h", name="tasa observada (TRAIN)",
                marker_color=GRIS, hovertemplate="%{y}<br>observada %{x:.1f} %<extra></extra>")
    fig.add_scatter(y=c["mode_x_daypart"], x=100 * c["p_predicha"], mode="markers", name="probabilidad promedio predicha (TEST)",
                    marker={"color": AZUL, "size": 12, "symbol": "diamond", "line": {"width": 2, "color": SUPERFICIE}},
                    hovertemplate="%{y}<br>predicha %{x:.1f} %<extra></extra>")
    fig.update_layout(xaxis_title="% de entregas tardías", yaxis_title=None, xaxis_range=[0, 105])
    return _estilo(fig, "Probabilidad del modelo vs tasa observada, por modo × jornada", 380)


def fig_subgrupos(variable: str = "market", seleccion: list[str] | None = None) -> go.Figure:
    p = predicciones()
    seleccion = [m for m in (seleccion or modelos()) if not es_dummy(m)]
    fig = go.Figure()
    for m in seleccion:
        filas = []
        for g, d in p.groupby(variable):
            if len(d) >= 300 and d["y"].nunique() == 2:
                filas.append((str(g), skm.roc_auc_score(d["y"], d[f"p::{m}"]), len(d)))
        f = pd.DataFrame(filas, columns=["grupo", "auc", "n"])
        fig.add_bar(x=f["grupo"], y=f["auc"], name=m, marker_color=COLOR_MODELO.get(m, AZUL), customdata=f["n"],
                    hovertemplate="%{x}<br>ROC-AUC %{y:.3f}<br>%{customdata:,} líneas<extra>" + m + "</extra>")
    fig.add_hline(y=0.5, line_dash="dash", line_color=GRIS)
    fig.update_layout(barmode="group", yaxis_title="ROC-AUC en la prueba", xaxis_title=None, yaxis_range=[0.4, 1.02],
                      bargap=0.3, bargroupgap=0.08)
    return _estilo(fig, f"ROC-AUC por {ETIQUETAS.get(variable, variable).lower()} (prueba)", 360)


def fig_curva_aprendizaje() -> go.Figure:
    c = tabla("curva_aprendizaje.csv")
    fig = go.Figure()
    for col, nombre, color in [("AUC_train", "entrenamiento", NARANJA), ("AUC_valid", "validación (bloque futuro)", AZUL)]:
        fig.add_scatter(x=c["n_train"], y=c[col], mode="lines+markers", name=nombre, line={"color": color, "width": 2},
                        marker={"size": 8}, error_y={"type": "data", "array": c[f"{col}_sd"], "thickness": 1, "color": color},
                        hovertemplate="%{x:,.0f} líneas<br>ROC-AUC %{y:.3f}<extra>" + nombre + "</extra>")
    fig.update_layout(xaxis_type="log", xaxis_title="líneas de entrenamiento (escala log)", yaxis_title="ROC-AUC")
    return _estilo(fig, "Curva de aprendizaje (pliegues temporales)", 340)


def fig_seleccion() -> go.Figure:
    s = tabla("seleccion_conjuntos.csv")
    colores = [AZUL if r.startswith("C ") else GRIS for r in s["conjunto"]]
    fig = go.Figure(go.Bar(x=s["ROC-AUC media"], y=s["conjunto"], orientation="h", marker_color=colores,
                           error_x={"type": "data", "array": s["ROC-AUC desv."], "color": TEXTO_2},
                           customdata=s["n_variables"],
                           hovertemplate="%{y}<br>ROC-AUC %{x:.4f}<br>%{customdata} variables<extra></extra>"))
    fig.update_layout(xaxis_title="ROC-AUC medio (CV temporal, entrenamiento)", yaxis_title=None, xaxis_range=[0.72, 0.785],
                      yaxis={"autorange": "reversed"})
    return _estilo(fig, "Selección de variables: conjuntos comparados", 340)
