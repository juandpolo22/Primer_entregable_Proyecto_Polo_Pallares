"""Dashboard DataCo (sin fuga) — Entregable 2.

Ejecutar desde la raíz del proyecto:

    python dashboard/app.py                      # http://127.0.0.1:8050  (puerto configurable con la variable PORT)

En producción (Render) se usa gunicorn sobre el objeto ``server``:

    gunicorn dashboard.app:server

Tres pestañas principales: (1) Contexto del problema, (2) EDA y (3) Modelos base (regresión logística), con navegación
secundaria dentro de cada una. Las figuras vienen de ``figuras.py`` (aspecto en ``estilo_figuras.py``), los textos de
``textos.py``, los títulos con hallazgos de ``hallazgos.py`` (todas las cifras se calculan desde los artefactos) y los datos de
``dashboard/datos/`` (generados por ``preparar_datos.py``). La métrica principal y la única que se muestra es el ROC-AUC;
las demás métricas siguen en las tablas de resultados y en los capítulos del Jupyter Book. Interpretaciones completas: capítulos 13–15.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estilo_figuras as E  # noqa: E402
import figuras as F  # noqa: E402
import hallazgos as H  # noqa: E402
import textos as T  # noqa: E402

CONFIG_GRAF = {"displaylogo": False, "responsive": True}
limpiar = E.limpiar


# ---------------------------------------------------------------------------
# Componentes reutilizables
# ---------------------------------------------------------------------------
def insignia(tipo: str) -> html.Span:
    """Etiqueta de origen de los datos: TRAIN (entrenamiento), OOF (fuera de pliegue en el entrenamiento) o TEST (prueba)."""
    etiquetas = {"train": ("TRAIN", "Calculado con el conjunto de entrenamiento"),
                 "oof": ("OOF", "Predicciones fuera de pliegue (validación cruzada temporal) del entrenamiento"),
                 "test": ("TEST", "Evaluado en la prueba cronológica")}
    texto, ayuda = etiquetas[tipo]
    return html.Span(texto, className=f"badge {tipo}", title=ayuda)


def miles(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def seccion(titulo: str, sub: str = "", tipo: str = "", *hijos) -> html.Section:
    cab = [html.H3([titulo, " ", insignia(tipo)] if tipo else titulo, className="sec-title")]
    if sub:
        cab.append(html.P(sub, className="sec-sub"))
    return html.Section(cab + list(hijos), className="sec")


def grafico(id_: str, figura: go.Figure, alto: int | None = None, **kw) -> dcc.Graph:
    return dcc.Graph(id=id_, figure=limpiar(figura, alto, **kw), config=CONFIG_GRAF)


def kpi(valor: str, nombre: str, sub: str = "", clase: str = "") -> html.Div:
    hijos = [html.Div(valor, className="kpi-valor"), html.Div(nombre, className="kpi-nombre")]
    if sub:
        hijos.append(html.Div(sub, className="kpi-sub"))
    return html.Div(hijos, className=f"kpi {clase}".strip())


def conclusion(texto: str, tono: str = "") -> html.Div:
    """Una línea visible; si el texto original es largo, el resto queda en «Ver más» (no se elimina nada)."""
    partes = re.split(r"(?<=\.)\s+", texto.strip(), maxsplit=1)
    if len(texto) > 190 and len(partes) == 2:
        return html.Div([html.Span(partes[0]), html.Details([html.Summary("Ver más"), partes[1]])], className=f"callout {tono}".strip())
    return html.Div(texto, className=f"callout {tono}".strip())


def detalle(titulo: str, *hijos) -> html.Details:
    return html.Details([html.Summary(titulo), html.Div(list(hijos), className="mas-cuerpo")], className="mas")


def chips(items, tipo: str) -> html.Div:
    return html.Div([html.Span(i, className=f"chip {tipo}") for i in items], className="chips")


def control(etiqueta: str, componente) -> html.Div:
    return html.Div([html.Label(etiqueta), componente], className="control")


def subpestanas(id_: str, valor: str, pestanas: list) -> dcc.Tabs:
    return dcc.Tabs(id=id_, value=valor, className="subnav-dash-in", parent_className="subnav-dash", children=[
        dcc.Tab(label=etiqueta, value=v, className="tab", selected_className="tab--selected", children=hijo)
        for etiqueta, v, hijo in pestanas])


# ---------------------------------------------------------------------------
# Pestaña 1: contexto del problema
# ---------------------------------------------------------------------------
def linea_tiempo(mini: bool = False) -> html.Div:
    c = F.contexto()
    tr, te = c["modelado_train"], c["modelado_test"]
    p_tr = 100 * tr / (tr + te)

    def tramo(clase, nombre, valor, flex):
        hijos = [html.B(nombre)] + ([html.Span(valor)] if valor else [])
        return html.Div(hijos, className=f"lt-tramo {clase}", style={"flex": flex})

    if mini:
        return html.Div([tramo("train", "TRAIN", "", f"{p_tr:.1f}"), tramo("gap", "GAP", "", "0 0 46px"),
                         tramo("test", "TEST", "", f"{100 - p_tr:.1f}")], className="linea-tiempo mini")
    return html.Div([tramo("train", "TRAIN", f"{miles(tr)} líneas", f"{p_tr:.1f}"), html.Div("→", className="lt-flecha"),
                     tramo("gap", "GAP", f"{c['gap_dias']} días", "0 0 118px"), html.Div("→", className="lt-flecha"),
                     tramo("test", "TEST", f"{miles(te)} líneas", f"{100 - p_tr:.1f}")], className="linea-tiempo")


def pestana_contexto() -> html.Div:
    c = F.contexto()
    q = F.calidad()["variables"]
    ref = F.tabla("referencia_empirica.csv").iloc[0]
    tasa = c["tasa_tardia_train"]

    resumen = html.Div([
        html.Div([
            html.Div([html.Div(F.pct(tasa), className="hero-valor"), html.Div("Líneas con entrega tardía", className="hero-nombre"),
                      html.Div(html.I(style={"width": f"{100 * tasa:.1f}%"}), className="hero-barra"),
                      html.Div("Más de la mitad de las líneas llegan después de la fecha prometida" if tasa > 0.5
                               else "Proporción de líneas que llegan después de la fecha prometida", className="hero-sub")], className="hero"),
            html.Div([kpi(miles(c["pedidos"]), "pedidos"), kpi(miles(c["lineas"]), "líneas de pedido"),
                      kpi(f"{c['desde'][:4]}–{c['hasta'][:4]}", "periodo")], className="kpi-col"),
        ], className="hero-grid"),
        html.P(["DataCo vende a ", str(c["paises_destino"]), " países y una parte importante de sus entregas llega después de la fecha prometida. "
                "Predecirlo ", html.B("al crear el pedido"), " permite actuar antes del despacho."], className="lead"),
        html.P(T.PREGUNTA, className="pregunta"),
        html.Div([
            html.Div([html.Div("Objetivo", className="mini-etq"), html.Code("Late_delivery_risk"),
                      html.Div([html.Span(className="pt azul"), "0 = no tardía ", html.Span(className="pt navy"), "1 = tardía"], className="mini-fila")],
                     className="mini-panel"),
            html.Div([html.Div("Unidad de análisis", className="mini-etq"), html.B("Línea de pedido"),
                      html.Div(f"≈ {F.num(T.LINEAS_POR_PEDIDO, 1)} líneas por pedido, misma etiqueta", className="mini-fila")], className="mini-panel"),
            html.Div([html.Div("Partición cronológica", className="mini-etq"), linea_tiempo(mini=True),
                      html.Div("entrena con el pasado · evalúa en el futuro", className="mini-fila")], className="mini-panel"),
        ], className="trio"),
    ])

    texto_part = ("Los pedidos se ordenan por fecha y se dividen en dos bloques: **entrenamiento** (TRAIN, el pasado, con el que el modelo aprende "
                  "y con el que se hace todo el análisis exploratorio) y **prueba** (TEST, el futuro, que solo se usa al final para medir el "
                  "desempeño). Entre ambos hay un **gap** que evita que información de un bloque se filtre al otro. "
                  "Por eso la pestaña de modelos muestra resultados en la prueba y la de exploración solo mira el entrenamiento.")
    datos = html.Div([
        seccion("Entrenar con el pasado, evaluar en el futuro", "Los pedidos se ordenan por fecha y se separan en dos bloques con un margen entre ambos.", "",
                linea_tiempo(),
                html.Div([html.Span(f"hasta {c['inicio_gap']}"), html.Span(f"{c['gap_dias']} días descartados"),
                          html.Span(f"desde {c['inicio_test']}")], className="lt-fechas")),
        html.Div([kpi(miles(c["modelado_pedidos_train"]), "pedidos en TRAIN", "el modelo aprende aquí y se explora el EDA", "train"),
                  kpi(miles(c["modelado_pedidos_test"]), "pedidos en TEST", "solo se usa para medir al final", "test"),
                  kpi(str(q["originales"]), "variables originales"),
                  kpi(str(q["candidatas_auditoria"]), "predictoras candidatas", "tras la auditoría de fuga")], className="kpi-row"),
        seccion(H.titulo_particion(), "", "", grafico("g-particion", F.fig_particion(), 330)),
        detalle("Explicación metodológica completa", dcc.Markdown(texto_part), conclusion(T.LECTURAS["particion"])),
    ])

    objetivo = html.Div([
        html.Div([
            html.Div([html.Div("Variable objetivo", className="mini-etq"), html.Code("Late_delivery_risk", className="grande"),
                      html.Div([html.Div([html.B("0"), html.Span("no tardía"), html.Small("adelantada o a tiempo")], className="clase no"),
                                html.Div([html.B("1"), html.Span("tardía"), html.Small("días reales > programados")], className="clase si")], className="dos"),
                      html.Div(["Cancelados y sospechosos de fraude tienen etiqueta 0 pero ", html.B("no son entregas a tiempo"),
                                ": se excluyen del modelado."], className="alerta-chip")], className="objetivo-card"),
            html.Div([html.Div("Unidad de análisis", className="mini-etq"), html.Div("Línea de pedido", className="grande-txt"),
                      html.P(["Un producto dentro de un pedido. Un pedido tiene ≈ ", F.num(T.LINEAS_POR_PEDIDO, 1), " líneas y ",
                              html.B("todas comparten la misma etiqueta"), ". Las incertidumbres se calculan agrupando por pedido."], className="peq")],
                     className="objetivo-card"),
        ], className="objetivo-grid"),
        detalle("Definición completa del objetivo y de la unidad", dcc.Markdown(T.OBJETIVO), dcc.Markdown(T.UNIDAD)),
    ])

    ruta = ("- **Ruta A (clasificación)** con las secciones **2.6–2.8** porque hay fecha y coordenadas.\n"
            f"- Partición **cronológica**: entrenamiento hasta {c['inicio_gap']}, gap de {c['gap_dias']} días y prueba desde {c['inicio_test']}.\n"
            "- Población de modelado sin cancelados ni fraude.\n"
            f"- Referencia empírica sin fuga (tasas modo × jornada): ROC-AUC {F.num(ref['ROC-AUC'])} en la prueba; línea base trivial (Dummy): 0,5.")
    metodo = html.Div([
        html.Div([kpi("ROC-AUC", "métrica principal", "ordena por riesgo; no depende del umbral"),
                  kpi("Validación temporal", "sin mezclar el futuro", f"gap de {c['gap_dias']} días"),
                  kpi("Sin fuga", "solo datos del momento del pedido", "ver EDA · D", "ok"),
                  kpi("Regresión logística", "modelo base", "modo de envío × jornada")], className="kpi-row"),
        html.P("Fuente: " + T.FUENTE.replace("*", ""), className="peq fuente"),
        detalle("Cómo se evalúa y ruta del enunciado", dcc.Markdown(T.METRICA.split(" Accuracy,")[0]), dcc.Markdown(T.SIN_FUGA), dcc.Markdown(ruta)),
    ])

    return subpestanas("sub-ctx", "resumen", [("Resumen", "resumen", resumen), ("Datos y partición", "datos", datos),
                                              ("Objetivo", "objetivo", objetivo), ("Método y fuente", "metodo", metodo)])


# ---------------------------------------------------------------------------
# Pestaña 2: EDA (navegación secundaria A–D)
# ---------------------------------------------------------------------------
CAT_OPC = [{"label": F.ETIQUETAS[v], "value": v} for v in
           ["shipping_mode", "payment_type", "market", "order_region", "customer_segment", "department_name",
            "category_name", "order_weekend", "order_afternoon"]]
NUM_OPC = [{"label": F.ETIQUETAS[v], "value": v} for v in
           ["sales", "order_item_quantity", "order_item_discount_rate", "order_item_profit_ratio", "items_in_order",
            "orders_prev_7d", "order_hour", "days_shipping_scheduled"]]
MERCADOS = ["Todos"] + sorted(m for m in F.tabla("serie_semanal_mercado.csv")["market"].unique() if m != "Todos")
ALTO_CAT, ALTO_SERIE = 330, 320


def fig_faltantes() -> go.Figure:
    f = F.calidad()["faltantes"]["detalle"]
    fig = go.Figure(go.Bar(y=list(f.keys()), x=list(f.values()), orientation="h", marker_color=E.NAVY,
                           text=[F.pct(x / 100) for x in f.values()], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:.1f} % de faltantes<extra></extra>"))
    fig.update_layout(xaxis={"range": [0, 118], "title": "% de valores faltantes"}, yaxis={"autorange": "reversed"}, showlegend=False)
    return fig


def stat(valor: str, nombre: str, sub: str = "", clase: str = "") -> html.Div:
    hijos = [html.Div(valor, className="stat-v"), html.Div(nombre, className="stat-n")]
    if sub:
        hijos.append(html.Div(sub, className="stat-s"))
    return html.Div(hijos, className=f"stat {clase}".strip())


def pestana_eda() -> html.Div:
    q = F.calidad()
    f, i, inc, fu, v = q["faltantes"], q["integridad"], q["inconsistencias"], q["fuga"], q["variables"]
    comp = i["pedidos_compartidos_train_test"]

    calidad = html.Div([
        html.Div([
            html.Div([html.Div([f"{f['n_variables']} ", html.Small(f"/ {v['originales']}")], className="hero-valor"),
                      html.Div("variables con faltantes", className="hero-nombre"),
                      grafico("g-faltantes", fig_faltantes(), 170)], className="hero-claro"),
            html.Div([
                stat(f"✓ {comp}", "pedidos compartidos TRAIN/TEST", "ningún pedido queda dividido" if comp == 0 else "revisar la partición",
                     "ok" if comp == 0 else "alerta"),
                stat(str(i["columnas_duplicadas"]), "columnas duplicadas o equivalentes"),
                stat(f"{inc['con_hallazgos']} / {inc['comprobaciones']}", "comprobaciones con hallazgos",
                     f"{miles(inc['lineas_tardias_por_dias_con_etiqueta_0'])} líneas «tardías» por días con etiqueta 0"),
                stat(str(len(fu["alto_riesgo"])), "variables de alto riesgo de fuga", "ver bloque D", "alerta"),
            ], className="stat-grid"),
        ], className="a-grid"),
        seccion(H.titulo_objetivo(), "Distribución del objetivo por estado de entrega (entrenamiento completo)", "train",
                grafico("g-objetivo", F.fig_objetivo(), 320, leyenda_abajo=True), conclusion(T.LECTURAS["objetivo"])),
        detalle("Nota sobre la calidad de datos", T.CALIDAD_NOTA + " Todo el EDA usa solo el conjunto de entrenamiento; la prueba no se mira."),
    ])

    relaciones = html.Div([
        seccion(H.titulo_categoria(), "Tasa de entregas tardías por variable categórica, con IC 95 % de Wilson", "train",
                control("Variable categórica", dcc.Dropdown(CAT_OPC, "shipping_mode", id="dd-cat", clearable=False)),
                dcc.Graph(id="g-cat", config=CONFIG_GRAF), conclusion(T.LECTURAS["tasa_categoria"])),
        seccion(H.TITULO_HORA, "Tasa de tardías (%) según hora del pedido y modo de envío", "train",
                grafico("g-hora", F.fig_hora_modo(), 330), conclusion(T.LECTURAS["hora_modo"])),
        seccion(H.TITULO_NUMERICA, "Variable numérica según el objetivo (muestra de 15.000 líneas)", "train",
                control("Variable numérica", dcc.Dropdown(NUM_OPC, "sales", id="dd-num", clearable=False)),
                dcc.Graph(id="g-num", config=CONFIG_GRAF), conclusion(T.LECTURAS["numerica"])),
    ])

    tiempo = html.Div([
        seccion(H.titulo_serie(), "Serie semanal de la tasa de entregas tardías (entrenamiento)", "train",
                control("Mercado de destino", dcc.Dropdown(MERCADOS, "Todos", id="dd-mercado", clearable=False)),
                dcc.Graph(id="g-serie", config=CONFIG_GRAF), conclusion(T.LECTURAS["serie"])),
        seccion(H.titulo_moran(), "Tasa de tardías por celda de 1° de latitud/longitud (ubicación del cliente)", "train",
                grafico("g-mapa", F.fig_mapa(), 470), conclusion(T.LECTURAS["mapa"])),
    ])

    fuga = html.Div([
        html.Div([html.Div("✓ SIN FUGA", className="sello-marca"),
                  html.Div([html.B("Los modelos usan solo información disponible al crear el pedido. "),
                            f"Las {len(fu['posteriores_al_pedido'])} variables que se conocen después del despacho quedan fuera:"], className="sello-txt"),
                  chips(fu["posteriores_al_pedido"], "fuga")], className="sello"),
        html.Div([
            html.Div([html.Div(f"Disponibles al crear el pedido ({len(fu['disponibles_al_pedido'])})", className="mini-etq"),
                      chips(fu["disponibles_al_pedido"], "ok")]),
            html.Div([html.Div("Posteriores al pedido · excluidas", className="mini-etq"), chips(fu["posteriores_al_pedido"], "fuga"),
                      html.P("Solo aparecen en la comparación con/sin fuga, como demostración metodológica.", className="peq")]),
        ], className="dos-col"),
        seccion(H.titulo_fuga_auc(), "AUC univariado de cada variable auditada", "train", grafico("g-fuga", F.fig_fuga(), 520)),
        seccion(H.titulo_con_sin_fuga(), "Validación cruzada temporal, con y sin variables posteriores al pedido", "train",
                grafico("g-consin", F.fig_con_sin_fuga(), 260), conclusion(T.LECTURAS["fuga"])),
    ])

    return subpestanas("sub-eda", "calidad", [("A · Objetivo y calidad", "calidad", calidad),
                                              ("B · Relaciones con el objetivo", "relaciones", relaciones),
                                              ("C · Tiempo y espacio", "tiempo", tiempo),
                                              ("D · Auditoría de fuga", "fuga", fuga)])


# ---------------------------------------------------------------------------
# Pestaña 3: modelos base (regresión logística). Única métrica mostrada: ROC-AUC
# ---------------------------------------------------------------------------
MODELOS = F.modelos()


def tabla_roc() -> html.Table:
    """Tabla compacta: ROC-AUC en la prueba (IC 95 %) y en validación cruzada temporal, de los tres modelos."""
    t = F.tabla_modelos()[["Modelo", "ROC-AUC", "ROC-AUC CV temporal"]]
    cab = html.Tr([html.Th(c, title=T.TOOLTIPS.get(c, "")) for c in t.columns])
    filas = []
    for _, r in t.iterrows():
        celdas = [html.Td(r["Modelo"])] + [html.Td(str(r[c]).replace("\n", "  ")) for c in ("ROC-AUC", "ROC-AUC CV temporal")]
        filas.append(html.Tr(celdas, className="elegido" if r["Modelo"] == F.MODELO_ELEGIDO else ""))
    return html.Table([html.Thead(cab), html.Tbody(filas)], className="tabla")


def tiles_umbral(marcado: str, auc: str, nota: str) -> html.Div:
    return html.Div([html.Div([html.Div(marcado, className="tile-v"), html.Div("marcado como tardío", className="tile-n")], className="tile"),
                     html.Div([html.Div(auc, className="tile-v"), html.Div(nota, className="tile-n")], className="tile")],
                    className="tiles dos-t")


def pestana_modelos() -> html.Div:
    M, D, umbral = F.MODELO_ELEGIDO, F.MODELO_DUMMY, F.umbral_final()
    prueba = F.tabla("modelos_prueba.csv").set_index("modelo")
    ic = F.tabla("modelos_ic.csv")
    e = ic[ic["modelo"] == M].set_index("Métrica")
    cv = F.tabla("modelos_cv.csv")
    cv_m = cv[(cv["modelo"] == M) & (cv["métrica"] == "ROC-AUC")].iloc[0]
    v_m, v_d = float(prueba.loc[M, "ROC-AUC"]), float(prueba.loc[D, "ROC-AUC"])
    pred = F.predicciones()
    m_test, m_oof = F.metricas_umbral(M, umbral, "test"), F.metricas_umbral(M, umbral, "oof")

    banner = html.Div([html.Span("MODELO BASE", className="bm-etq"), html.Span("REGRESIÓN LOGÍSTICA", className="bm-nombre"),
                       html.Span(f"{M} · evaluado una sola vez en la prueba ({pred['order_date'].min():%d-%m-%Y} a "
                                 f"{pred['order_date'].max():%d-%m-%Y}) con umbral {F.num(umbral, 2)} fijado de antemano", className="bm-det"),
                       insignia("test")], className="banner-modelo")

    comparacion = html.Div([
        html.Div([html.Span([html.I(className="b-modelo"), "Modelo base"]), html.Span([html.I(className="b-dummy"), "Dummy (referencia)"])], className="cmp-leyenda"),
        html.Div([html.Div("ROC-AUC", className="cmp-etq"),
                  html.Div([html.I(className="b-modelo", style={"width": f"{100 * v_m:.1f}%"}), html.I(className="b-dummy", style={"width": f"{100 * v_d:.1f}%"})], className="cmp-barras"),
                  html.Div([html.B(F.num(v_m)), html.Span(F.num(v_d))], className="cmp-val")], className="cmp-fila"),
    ], className="cmp")

    desempeno = html.Div([
        banner,
        html.Div([
            html.Div([html.Div(F.num(v_m), className="hero-valor"), html.Div("ROC-AUC · métrica principal", className="hero-nombre"),
                      html.Div("Mide qué tan bien el modelo ordena las líneas por riesgo de retraso", className="hero-sub")], className="hero"),
            html.Div([kpi(f"{F.num(e.loc['ROC-AUC'].iloc[1])}–{F.num(e.loc['ROC-AUC'].iloc[2])}", "IC 95 % (bootstrap por pedido)"),
                      kpi(f"{F.num(cv_m['media'])} ± {F.num(cv_m['desv'])}", "ROC-AUC en validación cruzada temporal"),
                      kpi(F.num(v_d), "Dummy (referencia)", "clase mayoritaria / prior", "ref")], className="kpi-col"),
        ], className="hero-grid modelo"),
        seccion("El modelo supera claramente la línea base", "ROC-AUC en la prueba: modelo base frente a Dummy", "test", comparacion),
        detalle("Tabla de ROC-AUC con IC 95 % (3 modelos)", tabla_roc(),
                html.P(["Fila resaltada: modelo elegido. ", T.IC_NOTA], className="peq"),
                conclusion(T.LECTURAS["tabla"].split(" Con umbral")[0]),
                html.P("Dummy: predice siempre la clase mayoritaria y usa la prevalencia como probabilidad; su ROC-AUC vale 0,5 por construcción.", className="peq")),
    ])

    roc = html.Div([
        html.Div(seccion(H.titulo_roc(), "Curva ROC en la prueba", "test",
                         control("Modelos a mostrar", dcc.Checklist(MODELOS, MODELOS, id="ck-modelos", inline=True,
                                                                    inputStyle={"marginRight": "4px", "marginLeft": "12px"})),
                         dcc.Graph(id="g-roc", config=CONFIG_GRAF)), className="una-graf"),
        conclusion(T.LECTURAS["roc"]),
    ])

    matriz = html.Div([
        html.Div([
            seccion(H.titulo_matriz_test(m_test, umbral), f"Matriz de confusión en la prueba · umbral {F.num(umbral, 2)} fijado de antemano · {miles(m_test['n'])} líneas",
                    "test", grafico("g-cm-test", F.fig_confusion(M, umbral, "test"), 420)),
            seccion(H.titulo_matriz_oof(m_oof), f"Matriz de confusión del entrenamiento · umbral {F.num(umbral, 2)} · {miles(m_oof['n'])} líneas",
                    "oof", grafico("g-cm-oof", F.fig_confusion(M, umbral, "oof"), 420)),
        ], className="dos-graf"),
        html.Div(H.lectura_matriz(m_test), className="callout"),
    ])

    umbral_tab = html.Div([
        html.Div([
            html.Div([insignia("oof"), html.Span(T.LECTURAS["umbral_aviso"])], className="oof-cab"),
            html.Div([
                html.Div([
                    control("Modelo", dcc.Dropdown(MODELOS, M, id="dd-modelo-umbral", clearable=False)),
                    html.Div([html.Label("Umbral de decisión"),
                              dcc.Slider(0.05, 0.95, 0.05, value=umbral, id="sl-umbral", marks={v / 100: f"{v / 100:.2f}".replace(".", ",") for v in (5, 50, 95)})],
                             className="slider-wrap"),
                    html.Div(id="txt-umbral"),
                    html.Div(H.LECTURA_UMBRAL, className="callout"),
                ]),
                html.Div([html.Div("Matriz de confusión (OOF)", className="mini-etq"), dcc.Graph(id="g-cm", config=CONFIG_GRAF)]),
            ], className="umbral-grid"),
        ], className="panel-oof"),
        html.Div([
            html.Div([insignia("test"), html.Span(T.LECTURAS["umbral_test"] + ". Sin control interactivo: el umbral no se ajusta con la prueba.")], className="oof-cab test"),
            tiles_umbral(F.pct(m_test["marcados_tardía"]), F.num(v_m), "ROC-AUC en la prueba"),
        ], className="panel-test"),
    ])

    calibracion = html.Div([
        seccion(H.titulo_calibracion(), "Calibración en la prueba", "test", grafico("g-cal", F.fig_calibracion(), 420), conclusion(T.LECTURAS["calibracion"])),
        seccion(H.TITULO_CELDAS, "Probabilidad predicha frente a tasa observada en entrenamiento", "test",
                grafico("g-celdas", F.fig_celdas(), 360), conclusion(T.LECTURAS["celdas"])),
        seccion(H.TITULO_SELECCION, "Selección de conjuntos de variables (validación cruzada temporal)", "train",
                grafico("g-sel", F.fig_seleccion(), 360), conclusion(T.LECTURAS["seleccion"])),
        seccion(H.TITULO_APRENDIZAJE, "Curva de aprendizaje", "train", grafico("g-ap", F.fig_curva_aprendizaje(), 360), conclusion(T.LECTURAS["aprendizaje"])),
        seccion(H.TITULO_SUBGRUPOS, "Desempeño por subgrupo en la prueba", "test",
                control("Subgrupo", dcc.Dropdown([{"label": F.ETIQUETAS[v], "value": v} for v in ["market", "customer_segment", "shipping_mode", "periodo"]],
                                                 "market", id="dd-sub", clearable=False)),
                dcc.Graph(id="g-sub", config=CONFIG_GRAF), conclusion(T.LECTURAS["subgrupos"])),
    ])

    return subpestanas("sub-modelos", "desempeno", [("Desempeño", "desempeno", desempeno), ("ROC", "roc", roc),
                                                    ("Matriz de confusión", "matriz", matriz), ("Umbral", "umbral", umbral_tab),
                                                    ("Calibración y diagnóstico", "calib", calibracion)])


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
app = Dash(__name__, title="DataCo · Riesgo de entrega tardía", suppress_callback_exceptions=True)
server = app.server          # objeto WSGI que usa gunicorn (gunicorn dashboard.app:server)


def _pestana(etiqueta: str, valor: str, hijo) -> dcc.Tab:
    return dcc.Tab(label=etiqueta, value=valor, className="tab", selected_className="tab--selected", children=html.Div(hijo, className="contenido"))


app.layout = html.Div([
    html.Header(html.Div([
        html.Div([html.H1("Riesgo de entrega tardía en DataCo"),
                  html.P("Predicción sin fuga · Entregable 2 · la explicación detallada está en el Jupyter Book"),
                  html.P("Autores: Juan David Polo y Gabriella Pallares", className="autores")]),
        html.Div([html.Span("Origen de los datos"), insignia("train"), insignia("oof"), insignia("test")], className="origen"),
    ], className="app-cab-in"), className="app-cab"),
    dcc.Tabs(id="pestanas", value="contexto", className="principal-dash-in", parent_className="principal-dash", children=[
        _pestana("1 · Contexto del problema", "contexto", pestana_contexto()),
        _pestana("2 · EDA", "eda", pestana_eda()),
        _pestana("3 · Modelos base (logístico)", "modelos", pestana_modelos()),
    ]),
])


@app.callback(Output("g-cat", "figure"), Input("dd-cat", "value"))
def _cat(v):
    return limpiar(F.fig_tasa_categoria(v), ALTO_CAT)


@app.callback(Output("g-num", "figure"), Input("dd-num", "value"))
def _num(v):
    return limpiar(F.fig_numerica(v), 320)


@app.callback(Output("g-serie", "figure"), Input("dd-mercado", "value"))
def _serie(v):
    return limpiar(F.fig_serie(v), ALTO_SERIE)


@app.callback(Output("g-roc", "figure"), Input("ck-modelos", "value"))
def _roc(sel):
    return E.acortar_curvas(limpiar(F.fig_roc(sel or [F.MODELO_ELEGIDO]), 560))


@app.callback(Output("g-cm", "figure"), Output("txt-umbral", "children"),
              Input("dd-modelo-umbral", "value"), Input("sl-umbral", "value"))
def _umbral(modelo, umbral):
    """El deslizador trabaja SOLO con las predicciones OOF del entrenamiento (nunca con la prueba)."""
    m = F.metricas_umbral(modelo, umbral, fuente="oof")
    return (limpiar(F.fig_confusion(modelo, umbral, fuente="oof"), 400),
            tiles_umbral(F.pct(m["marcados_tardía"]), F.num(m["ROC-AUC"]), "ROC-AUC (no cambia con el umbral)"))


@app.callback(Output("g-sub", "figure"), Input("dd-sub", "value"))
def _sub(v):
    return limpiar(F.fig_subgrupos(v), 360)


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", "8050"))
    # 0.0.0.0 permite abrirlo desde otro equipo/contenedor; en local basta http://127.0.0.1:<puerto>.
    app.run(debug=False, host="0.0.0.0", port=puerto)
