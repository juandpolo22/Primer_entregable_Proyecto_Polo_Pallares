"""Apariencia de las figuras Plotly para el dashboard (solo estética).

No cambia datos, métricas ni cálculos: toma las figuras de ``figuras.py`` y ajusta títulos, márgenes, leyendas y colores.
Lo usan la app Dash (``app.py``) y la vista previa estática (``exportar_html.py``) para que se vean idénticas.
Reglas de color: azul oscuro/azul para entrenamiento y modelo, verde para validación (OOF), naranja SOLO para TEST,
rojo SOLO para alertas (variables con fuga).
"""
from __future__ import annotations

import re

import plotly.graph_objects as go

import figuras as F

NAVY, AZUL, AZUL_2, GRIS = "#0d366b", "#2a78d6", "#8fb8ea", "#9aa4b2"
SEQ = [[0.0, "#eaf2fc"], [0.5, "#8fb8ea"], [1.0, "#0d366b"]]           # escala secuencial (sin rojo)
COLOR_MODELO = {"aditiva": AZUL_2, "Dummy": GRIS, "modo × jornada": AZUL}
CORTO = {F.MODELO_DUMMY: "Dummy", "Logística aditiva (modo + hora)": "Aditiva (modo + hora)", "Logística modo × jornada": "Modo × jornada"}


def _color_modelo(nombre: str):
    for clave, col in COLOR_MODELO.items():
        if clave in nombre:
            return col
    return None


def limpiar(fig: go.Figure, alto: int | None = None, leyenda_abajo: bool = False) -> go.Figure:
    """Quita el título interno (el título del hallazgo va en HTML), aligera ejes y leyenda y unifica los colores."""
    fig.update_layout(
        title=None, margin={"l": 58, "r": 16, "t": 30, "b": 48}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "size": 12, "color": "#5b6776"},
        hoverlabel={"bgcolor": "white", "font_size": 12},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": -0.2 if leyenda_abajo else 1.08,
                "yanchor": "top" if leyenda_abajo else "bottom", "title": None, "font": {"size": 11}, "bgcolor": "rgba(0,0,0,0)"})
    fig.update_xaxes(gridcolor="#eef1f5", zeroline=False, linecolor="#d9dee5")
    fig.update_yaxes(gridcolor="#eef1f5", zeroline=False, linecolor="#d9dee5")
    if alto:
        fig.update_layout(height=alto)
    for tr in fig.data:
        nombre = str(tr.name or "")
        if tr.type == "heatmap" and getattr(tr, "zmid", None) is not None:       # divergente con rojo -> secuencial azul
            tr.update(colorscale=SEQ, zmid=None)
        if tr.type == "scatter" and tr.marker is not None and getattr(tr.marker, "cmid", None) is not None:
            tr.marker.update(colorscale=SEQ, cmid=None)
        if tr.type == "histogram":
            tr.update(marker_color=NAVY if nombre == "tardía" else AZUL_2, opacity=0.6)
        if tr.type == "bar" and nombre.startswith("Late_delivery_risk = 1"):
            tr.update(marker_color=NAVY, name="Tardía (1)")
        elif tr.type == "bar" and nombre.startswith("Late_delivery_risk = 0 (no"):
            tr.update(marker_color=AZUL_2, name="No tardía (0)")
        elif tr.type == "bar" and nombre.startswith("Late_delivery_risk = 0 pero"):
            tr.update(name="Cancelado / fraude (0, no entregado; excluido)")
        col = _color_modelo(nombre)
        if col and tr.type in ("scatter", "bar"):
            if tr.type == "scatter":
                tr.update(line_color=col)
                if "markers" in (tr.mode or ""):
                    tr.update(marker_color=col)
                if col == GRIS:
                    tr.update(line={"color": GRIS, "width": 1.6, "dash": "dot"}, opacity=0.9)
                elif col == AZUL:
                    tr.update(line_width=3)
            else:
                tr.update(marker_color=col)
        if tr.type == "scatter" and nombre == "entrenamiento":                    # el naranja queda reservado a TEST
            tr.update(line_color=NAVY, marker_color=NAVY)
            if tr.error_y is not None and tr.error_y.visible:
                tr.error_y.update(color=NAVY)
        if nombre in CORTO:
            tr.update(name=CORTO[nombre])
    return fig


def acortar_curvas(fig: go.Figure) -> go.Figure:
    """Leyenda simple de la curva ROC: «Modo × jornada · AUC 0,775» en lugar del nombre largo."""
    patron = re.compile(r"^(.*) \(ROC-AUC ([\d,]+) · AP ([\d,]+)\)$")
    for tr in fig.data:
        m = patron.match(str(tr.name or ""))
        if m:
            tr.update(name=f"{CORTO.get(m.group(1), m.group(1))} · AUC {m.group(2)}")
        elif str(tr.name).startswith("azar"):
            tr.update(name="azar")
    fig.update_layout(yaxis_title="tasa de verdaderos positivos",
                      legend={"orientation": "v", "x": 0.98, "xanchor": "right", "y": 0.04, "yanchor": "bottom",
                              "bgcolor": "rgba(255,255,255,0.85)", "bordercolor": "#e6e9ee", "borderwidth": 1, "font": {"size": 11}})
    return fig
