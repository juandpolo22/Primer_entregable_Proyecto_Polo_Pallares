"""Exporta una VISTA PREVIA estática del dashboard a un solo archivo HTML.

    python dashboard/exportar_html.py

Genera ``dashboard/vista_previa_dashboard.html`` (funciona sin servidor; el JavaScript de Plotly va incluido, así que
tampoco necesita internet). Reproduce las tres pestañas y su navegación secundaria con el mismo diseño de la app Dash
(``dashboard/assets/estilo.css``); los gráficos conservan zoom y tooltips, el control de umbral funciona con valores
precalculados (predicciones OOF) y los menús desplegables muestran su valor por defecto: solo son interactivos en la app Dash
(``python dashboard/app.py``). Usa las MISMAS funciones de figuras, textos y títulos que la app. Única métrica mostrada: ROC-AUC.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from markdown_it import MarkdownIt  # viene con jupyter-book (myst-parser)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estilo_figuras as E  # noqa: E402
import figuras as F  # noqa: E402
import hallazgos as H  # noqa: E402
import textos as T  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "vista_previa_dashboard.html"
CSS = (Path(__file__).resolve().parent / "assets" / "estilo.css").read_text(encoding="utf-8")
_primera = {"hecho": False}


def md(texto: str) -> str:
    return MarkdownIt().render(texto)


def esc(x) -> str:
    return html.escape(str(x))


def miles(x) -> str:
    return f"{int(x):,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Apariencia de las figuras: módulo compartido con la app Dash
# ---------------------------------------------------------------------------
limpiar = E.limpiar
CORTO = E.CORTO
NAVY, AZUL, AZUL_2 = E.NAVY, E.AZUL, E.AZUL_2


def grafico(fig: go.Figure) -> str:
    incluir = "inline" if not _primera["hecho"] else False
    _primera["hecho"] = True
    return pio.to_html(fig, full_html=False, include_plotlyjs=incluir, config={"displaylogo": False, "responsive": True})


# ---------------------------------------------------------------------------
# Componentes HTML
# ---------------------------------------------------------------------------
def insignia(tipo: str) -> str:
    return f'<span class="badge {tipo}">{ {"train": "TRAIN", "oof": "OOF", "test": "TEST"}[tipo] }</span>'


def seccion(titulo: str, sub: str = "", tipo: str = "", cuerpo: str = "") -> str:
    b = " " + insignia(tipo) if tipo else ""
    s = f'<p class="sec-sub">{sub}</p>' if sub else ""
    return f'<section class="sec"><h3 class="sec-title">{esc(titulo)}{b}</h3>{s}{cuerpo}</section>'


def kpi(valor: str, nombre: str, sub: str = "", clase: str = "") -> str:
    s = f'<div class="kpi-sub">{esc(sub)}</div>' if sub else ""
    return f'<div class="kpi {clase}"><div class="kpi-valor">{esc(valor)}</div><div class="kpi-nombre">{esc(nombre)}</div>{s}</div>'


def conclusion(texto: str, tono: str = "") -> str:
    """Una línea visible; si el texto original es largo, el resto queda en «Ver más» (no se elimina nada)."""
    partes = re.split(r"(?<=\.)\s+", texto.strip(), maxsplit=1)
    if len(texto) > 190 and len(partes) == 2:
        return (f'<div class="callout {tono}"><span>{esc(partes[0])}</span>'
                f'<details><summary>Ver más</summary>{esc(partes[1])}</details></div>')
    return f'<div class="callout {tono}">{esc(texto)}</div>'


def detalle(titulo: str, cuerpo: str) -> str:
    return f'<details class="mas"><summary>{esc(titulo)}</summary><div class="mas-cuerpo">{cuerpo}</div></details>'


def chips(items, tipo: str) -> str:
    return '<div class="chips">' + "".join(f'<span class="chip {tipo}">{esc(i)}</span>' for i in items) + "</div>"


def falso_select(etiqueta: str, valor: str) -> str:
    return (f'<div class="filtro"><label>{esc(etiqueta)}</label><div class="select" title="En la app Dash este menú es funcional">'
            f'{esc(valor)}<span>▾</span></div></div>')


def subnav(grupo: str, pestanas: list) -> str:
    botones = "".join(f'<button class="{"activa" if i == 0 else ""}" data-g="{grupo}" data-p="{grupo}-{v}">{esc(e)}</button>'
                      for i, (e, v, _) in enumerate(pestanas))
    paneles = "".join(f'<div id="{grupo}-{v}" class="subpanel {"activo" if i == 0 else ""}" data-g="{grupo}">{h}</div>'
                      for i, (_, v, h) in enumerate(pestanas))
    return f'<nav class="subnav">{botones}</nav>{paneles}'


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------
def construir() -> str:
    c = F.contexto()
    q = F.calidad()
    tasa = c["tasa_tardia_train"]
    prueba = F.tabla("modelos_prueba.csv").set_index("modelo")
    ic = F.tabla("modelos_ic.csv")
    e = ic[ic["modelo"] == F.MODELO_ELEGIDO].set_index("Métrica")
    ref = F.tabla("referencia_empirica.csv").iloc[0]
    M, D = F.MODELO_ELEGIDO, F.MODELO_DUMMY
    umbral = F.umbral_final()

    def ic_txt(m: str) -> str:
        return f"IC 95 %: {F.num(e.loc[m].iloc[1])}–{F.num(e.loc[m].iloc[2])}"

    # =====================================================================================================
    # 1 · CONTEXTO
    # =====================================================================================================
    tr, te = c["modelado_train"], c["modelado_test"]
    p_tr = 100 * tr / (tr + te)
    cronologia = (f'<div class="linea-tiempo"><div class="lt-tramo train" style="flex:{p_tr:.1f}"><b>TRAIN</b><span>{miles(tr)} líneas</span></div>'
                  f'<div class="lt-flecha">→</div><div class="lt-tramo gap" style="flex:0 0 118px"><b>GAP</b><span>{c["gap_dias"]} días</span></div>'
                  f'<div class="lt-flecha">→</div><div class="lt-tramo test" style="flex:{100 - p_tr:.1f}"><b>TEST</b><span>{miles(te)} líneas</span></div></div>')
    cronologia_mini = (f'<div class="linea-tiempo mini"><div class="lt-tramo train" style="flex:{p_tr:.1f}"><b>TRAIN</b></div>'
                       f'<div class="lt-tramo gap" style="flex:0 0 46px"><b>GAP</b></div>'
                       f'<div class="lt-tramo test" style="flex:{100 - p_tr:.1f}"><b>TEST</b></div></div>')

    resumen = f"""
    <div class="hero-grid">
      <div class="hero">
        <div class="hero-valor">{F.pct(tasa)}</div>
        <div class="hero-nombre">Líneas con entrega tardía</div>
        <div class="hero-barra"><i style="width:{100 * tasa:.1f}%"></i></div>
        <div class="hero-sub">Más de la mitad de las líneas llegan después de la fecha prometida</div>
      </div>
      <div class="kpi-col">
        {kpi(miles(c['pedidos']), 'pedidos')}{kpi(miles(c['lineas']), 'líneas de pedido')}
        {kpi(c['desde'][:4] + '–' + c['hasta'][:4], 'periodo')}
      </div>
    </div>
    <p class="lead">DataCo vende a {c['paises_destino']} países y una parte importante de sus entregas llega después de la fecha prometida.
      Predecirlo <b>al crear el pedido</b> permite actuar antes del despacho.</p>
    <p class="pregunta">{esc(T.PREGUNTA)}</p>
    <div class="trio">
      <div class="mini-panel"><div class="mini-etq">Objetivo</div><code>Late_delivery_risk</code>
        <div class="mini-fila"><span class="pt azul"></span>0 = no tardía <span class="pt navy"></span>1 = tardía</div></div>
      <div class="mini-panel"><div class="mini-etq">Unidad de análisis</div><b>Línea de pedido</b>
        <div class="mini-fila">≈ {F.num(T.LINEAS_POR_PEDIDO, 1)} líneas por pedido, misma etiqueta</div></div>
      <div class="mini-panel"><div class="mini-etq">Partición cronológica</div>{cronologia_mini}
        <div class="mini-fila">entrena con el pasado · evalúa en el futuro</div></div>
    </div>"""

    fechas = (f'<div class="lt-fechas"><span>hasta {esc(c["inicio_gap"])}</span><span>{c["gap_dias"]} días descartados</span>'
              f'<span>desde {esc(c["inicio_test"])}</span></div>')
    texto_part = ("Los pedidos se ordenan por fecha y se dividen en dos bloques: **entrenamiento** (TRAIN, el pasado, con el que el modelo aprende "
                  "y con el que se hace todo el análisis exploratorio) y **prueba** (TEST, el futuro, que solo se usa al final para medir el "
                  "desempeño). Entre ambos hay un **gap** que evita que información de un bloque se filtre al otro. "
                  "Por eso la pestaña de modelos muestra resultados en la prueba y la de exploración solo mira el entrenamiento.")
    tit_part = H.titulo_particion()
    datos = f"""
    {seccion('Entrenar con el pasado, evaluar en el futuro', 'Los pedidos se ordenan por fecha y se separan en dos bloques con un margen entre ambos.', cuerpo=cronologia + fechas)}
    <div class="kpi-row">
      {kpi(miles(c['modelado_pedidos_train']), 'pedidos en TRAIN', 'el modelo aprende aquí y se explora el EDA', 'train')}
      {kpi(miles(c['modelado_pedidos_test']), 'pedidos en TEST', 'solo se usa para medir al final', 'test')}
      {kpi(str(q['variables']['originales']), 'variables originales')}
      {kpi(str(q['variables']['candidatas_auditoria']), 'predictoras candidatas', 'tras la auditoría de fuga')}
    </div>
    {seccion(tit_part, '', cuerpo=grafico(limpiar(F.fig_particion(), 330)))}
    {detalle('Explicación metodológica completa', md(texto_part) + conclusion(T.LECTURAS['particion']))}"""

    objetivo = f"""
    <div class="objetivo-grid">
      <div class="objetivo-card">
        <div class="mini-etq">Variable objetivo</div><code class="grande">Late_delivery_risk</code>
        <div class="dos"><div class="clase no"><b>0</b><span>no tardía</span><small>adelantada o a tiempo</small></div>
        <div class="clase si"><b>1</b><span>tardía</span><small>días reales &gt; programados</small></div></div>
        <div class="alerta-chip">Cancelados y sospechosos de fraude tienen etiqueta 0 pero <b>no son entregas a tiempo</b>: se excluyen del modelado.</div>
      </div>
      <div class="objetivo-card">
        <div class="mini-etq">Unidad de análisis</div><div class="grande-txt">Línea de pedido</div>
        <p class="peq">Un producto dentro de un pedido. Un pedido tiene ≈ {F.num(T.LINEAS_POR_PEDIDO, 1)} líneas y <b>todas comparten la misma etiqueta</b>.
        Las incertidumbres se calculan agrupando por pedido.</p>
      </div>
    </div>
    {detalle('Definición completa del objetivo y de la unidad', md(T.OBJETIVO) + md(T.UNIDAD))}"""

    ruta_md = (
        "- **Ruta A (clasificación)** con las secciones **2.6–2.8** porque hay fecha y coordenadas.\n"
        f"- Partición **cronológica**: entrenamiento hasta {c['inicio_gap']}, gap de {c['gap_dias']} días y prueba desde {c['inicio_test']}.\n"
        "- Población de modelado sin cancelados ni fraude.\n"
        f"- Referencia empírica sin fuga (tasas modo × jornada): ROC-AUC {F.num(ref['ROC-AUC'])} en la prueba; línea base trivial (Dummy): 0,5.")
    metodo = f"""
    <div class="kpi-row">
      {kpi('ROC-AUC', 'métrica principal', 'ordena por riesgo; no depende del umbral')}
      {kpi('Validación temporal', 'sin mezclar el futuro', f'gap de {c["gap_dias"]} días')}
      {kpi('Sin fuga', 'solo datos del momento del pedido', 'ver EDA · D', 'ok')}
      {kpi('Regresión logística', 'modelo base', 'modo de envío × jornada')}
    </div>
    <p class="peq fuente">Fuente: {esc(T.FUENTE).replace('*', '')}</p>
    {detalle('Cómo se evalúa y ruta del enunciado', md(T.METRICA.split(' Accuracy,')[0]) + md(T.SIN_FUGA) + md(ruta_md))}"""

    t1 = subnav("ctx", [("Resumen", "resumen", resumen), ("Datos y partición", "datos", datos),
                        ("Objetivo", "objetivo", objetivo), ("Método y fuente", "metodo", metodo)])

    # =====================================================================================================
    # 2 · EDA
    # =====================================================================================================
    f_, i_, inc, fu, v_ = q["faltantes"], q["integridad"], q["inconsistencias"], q["fuga"], q["variables"]
    comp = i_["pedidos_compartidos_train_test"]
    fig_falt = go.Figure(go.Bar(
        y=list(f_["detalle"].keys()), x=list(f_["detalle"].values()), orientation="h", marker_color=NAVY,
        text=[F.pct(x / 100) for x in f_["detalle"].values()], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.1f} % de faltantes<extra></extra>"))
    fig_falt.update_layout(xaxis={"range": [0, 118], "title": "% de valores faltantes"}, yaxis={"autorange": "reversed"}, showlegend=False)
    limpiar(fig_falt, 170)
    eda_a = f"""
    <div class="a-grid">
      <div>
        <div class="hero-claro"><div class="hero-valor">{f_['n_variables']} <small>/ {v_['originales']}</small></div>
          <div class="hero-nombre">variables con faltantes</div></div>
        {grafico(fig_falt)}
      </div>
      <div class="stat-grid">
        <div class="stat ok"><div class="stat-v">✓ {comp}</div><div class="stat-n">pedidos compartidos TRAIN/TEST</div><div class="stat-s">ningún pedido queda dividido</div></div>
        <div class="stat"><div class="stat-v">{i_['columnas_duplicadas']}</div><div class="stat-n">columnas duplicadas o equivalentes</div></div>
        <div class="stat"><div class="stat-v">{inc['con_hallazgos']} / {inc['comprobaciones']}</div><div class="stat-n">comprobaciones con hallazgos</div>
          <div class="stat-s">{miles(inc['lineas_tardias_por_dias_con_etiqueta_0'])} líneas «tardías» por días con etiqueta 0</div></div>
        <div class="stat alerta"><div class="stat-v">{len(fu['alto_riesgo'])}</div><div class="stat-n">variables de alto riesgo de fuga</div><div class="stat-s">ver bloque D</div></div>
      </div>
    </div>
    {seccion(H.titulo_objetivo(), 'Distribución del objetivo por estado de entrega (entrenamiento completo)', 'train',
             grafico(limpiar(F.fig_objetivo(), 320, leyenda_abajo=True)) + conclusion(T.LECTURAS['objetivo']))}
    {detalle('Nota sobre la calidad de datos', esc(T.CALIDAD_NOTA) + '<br>Todo el EDA usa solo el conjunto de entrenamiento; la prueba no se mira.')}"""

    titulo_cat = H.titulo_categoria()
    eda_b = f"""
    {seccion(titulo_cat, 'Tasa de entregas tardías por modo de envío, con IC 95 % de Wilson', 'train',
             falso_select('Variable categórica', 'Modo de envío') + grafico(limpiar(F.fig_tasa_categoria('shipping_mode'), 330)) + conclusion(T.LECTURAS['tasa_categoria']))}
    {seccion(H.TITULO_HORA, 'Tasa de tardías (%) según hora del pedido y modo de envío', 'train',
             grafico(limpiar(F.fig_hora_modo(), 330)) + conclusion(T.LECTURAS['hora_modo']))}
    {seccion(H.TITULO_NUMERICA, 'Variable numérica según el objetivo (muestra de 15.000 líneas)', 'train',
             falso_select('Variable numérica', 'Venta de la línea (USD)') + grafico(limpiar(F.fig_numerica('sales'), 320)) + conclusion(T.LECTURAS['numerica']))}"""

    eda_c = f"""
    {seccion(H.titulo_serie(), 'Serie semanal de la tasa de entregas tardías (entrenamiento)', 'train',
             falso_select('Mercado de destino', 'Todos') + grafico(limpiar(F.fig_serie('Todos'), 320)) + conclusion(T.LECTURAS['serie']))}
    {seccion(H.titulo_moran(), 'Tasa de tardías por celda de 1° de latitud/longitud (ubicación del cliente)', 'train',
             grafico(limpiar(F.fig_mapa(), 470)) + conclusion(T.LECTURAS['mapa']))}"""

    eda_d = f"""
    <div class="sello">
      <div class="sello-marca">✓ SIN FUGA</div>
      <div class="sello-txt"><b>Los modelos usan solo información disponible al crear el pedido.</b>
        Las {len(fu['posteriores_al_pedido'])} variables que se conocen después del despacho quedan fuera:</div>
      {chips(fu['posteriores_al_pedido'], 'fuga')}
    </div>
    <div class="dos-col">
      <div><div class="mini-etq">Disponibles al crear el pedido ({len(fu['disponibles_al_pedido'])})</div>{chips(fu['disponibles_al_pedido'], 'ok')}</div>
      <div><div class="mini-etq">Posteriores al pedido · excluidas</div>{chips(fu['posteriores_al_pedido'], 'fuga')}
        <p class="peq">Solo aparecen en la comparación con/sin fuga, como demostración metodológica.</p></div>
    </div>
    {seccion(H.titulo_fuga_auc(), 'AUC univariado de cada variable auditada', 'train',
             grafico(limpiar(F.fig_fuga(), 520)))}
    {seccion(H.titulo_con_sin_fuga(), 'Validación cruzada temporal, con y sin variables posteriores al pedido', 'train',
             grafico(limpiar(F.fig_con_sin_fuga(), 260)) + conclusion(T.LECTURAS['fuga']))}"""

    t2 = subnav("eda", [("A · Objetivo y calidad", "calidad", eda_a), ("B · Relaciones con el objetivo", "relaciones", eda_b),
                        ("C · Tiempo y espacio", "tiempo", eda_c), ("D · Auditoría de fuga", "fuga", eda_d)])

    # =====================================================================================================
    # 3 · MODELOS BASE
    # =====================================================================================================
    pred = F.predicciones()
    banner = (f'<div class="banner-modelo"><span class="bm-etq">MODELO BASE</span><span class="bm-nombre">REGRESIÓN LOGÍSTICA</span>'
              f'<span class="bm-det">{esc(M)} · evaluado una sola vez en la prueba ({pred["order_date"].min():%d-%m-%Y} a {pred["order_date"].max():%d-%m-%Y}) con umbral {F.num(umbral, 2)} fijado de antemano</span>'
              f'{insignia("test")}</div>')

    # --- Desempeño (única métrica: ROC-AUC)
    cv = F.tabla("modelos_cv.csv")
    cv_m = cv[(cv["modelo"] == M) & (cv["métrica"] == "ROC-AUC")].iloc[0]
    v_m, v_d = float(prueba.loc[M, "ROC-AUC"]), float(prueba.loc[D, "ROC-AUC"])
    comparacion = (f'<div class="cmp"><div class="cmp-leyenda"><span><i class="b-modelo"></i>Modelo base</span>'
                   f'<span><i class="b-dummy"></i>Dummy (referencia)</span></div>'
                   f'<div class="cmp-fila"><div class="cmp-etq">ROC-AUC</div>'
                   f'<div class="cmp-barras"><i class="b-modelo" style="width:{100 * v_m:.1f}%"></i><i class="b-dummy" style="width:{100 * v_d:.1f}%"></i></div>'
                   f'<div class="cmp-val"><b>{F.num(v_m)}</b><span>{F.num(v_d)}</span></div></div></div>')
    tabla = F.tabla_modelos()[["Modelo", "ROC-AUC", "ROC-AUC CV temporal"]]
    filas = "".join("<tr" + (' class="elegido"' if r["Modelo"] == M else "") + ">" +
                    "".join(f"<td>{esc(v).replace(chr(10), '<br>')}</td>" for v in r.values) + "</tr>" for _, r in tabla.iterrows())
    cab = "".join(f'<th title="{esc(T.TOOLTIPS.get(cn, ""))}">{esc(cn)}</th>' for cn in tabla.columns)
    tabla_html = (f'<div class="tabla-scroll"><table class="tabla"><thead><tr>{cab}</tr></thead><tbody>{filas}</tbody></table></div>'
                  f'<p class="peq">Fila resaltada: modelo elegido. {esc(T.IC_NOTA)}</p>'
                  + conclusion(T.LECTURAS['tabla'].split(' Con umbral')[0])
                  + '<p class="peq">Dummy: predice siempre la clase mayoritaria y usa la prevalencia como probabilidad; su ROC-AUC vale 0,5 por construcción.</p>')

    desempeno = f"""
    {banner}
    <div class="hero-grid modelo">
      <div class="hero">
        <div class="hero-valor">{F.num(v_m)}</div>
        <div class="hero-nombre">ROC-AUC · métrica principal</div>
        <div class="hero-sub">Mide qué tan bien el modelo ordena las líneas por riesgo de retraso</div>
      </div>
      <div class="kpi-col">
        {kpi(F.num(e.loc['ROC-AUC'].iloc[1]) + '–' + F.num(e.loc['ROC-AUC'].iloc[2]), 'IC 95 % (bootstrap por pedido)')}
        {kpi(F.num(cv_m['media']) + ' ± ' + F.num(cv_m['desv']), 'ROC-AUC en validación cruzada temporal')}
        {kpi(F.num(v_d), 'Dummy (referencia)', 'clase mayoritaria / prior', 'ref')}
      </div>
    </div>
    {seccion('El modelo supera claramente la línea base', 'ROC-AUC en la prueba: modelo base frente a Dummy', 'test', comparacion)}
    {detalle('Tabla de ROC-AUC con IC 95 % (3 modelos)', tabla_html)}"""

    # --- ROC
    roc = E.acortar_curvas(limpiar(F.fig_roc(), 560))
    roc_html = f"""
    <div class="una-graf">
      {seccion(H.titulo_roc(), 'Curva ROC en la prueba', 'test', grafico(roc))}
    </div>
    {conclusion(T.LECTURAS['roc'])}"""

    # --- Matriz de confusión
    m_test = F.metricas_umbral(M, umbral, "test")
    m_oof = F.metricas_umbral(M, umbral, "oof")
    cm_test = m_test["cm"]
    matriz = f"""
    <div class="dos-graf">
      {seccion(H.titulo_matriz_test(m_test, umbral),
               f'Matriz de confusión en la prueba · umbral {F.num(umbral, 2)} fijado de antemano · {miles(m_test["n"])} líneas', 'test',
               grafico(limpiar(F.fig_confusion(M, umbral, "test"), 420)))}
      {seccion(H.titulo_matriz_oof(m_oof),
               f'Matriz de confusión del entrenamiento · umbral {F.num(umbral, 2)} · {miles(m_oof["n"])} líneas', 'oof',
               grafico(limpiar(F.fig_confusion(M, umbral, "oof"), 420)))}
    </div>
    <div class="callout">{esc(H.lectura_matriz(m_test))}</div>"""

    # --- Umbral (interactivo en la vista previa: precalculado con OOF)
    grilla = [round(float(x), 2) for x in np.arange(0.05, 0.951, 0.05)]
    tabla_u = {}
    for u in grilla:
        mu = F.metricas_umbral(M, u, "oof")
        cm = mu["cm"]
        pc = cm / cm.sum(axis=1, keepdims=True)
        tabla_u[f"{u:.2f}"] = {
            "marcado": F.pct(mu["marcados_tardía"]),
            "z": pc.tolist(), "text": [[miles(cm[i, j]) + "<br>" + F.pct(pc[i, j]) for j in range(2)] for i in range(2)]}
    cm0 = limpiar(F.fig_confusion(M, umbral, "oof"), 400)
    n_oof = m_oof["n"]
    umbral_html = f"""
    <div class="panel-oof">
      <div class="oof-cab">{insignia('oof')}<span>{esc(T.LECTURAS['umbral_aviso'])}</span></div>
      <div class="umbral-grid">
        <div>
          <div class="filtro"><label>Modelo</label><div class="select">{esc(M)}<span>▾</span></div></div>
          <div class="slider-wrap"><label>Umbral de decisión: <b id="u-val">{F.num(umbral, 2)}</b></label>
            <input type="range" id="u-slider" min="0.05" max="0.95" step="0.05" value="{umbral:.2f}">
            <div class="slider-marcas"><span>0,05</span><span>0,50</span><span>0,95</span></div></div>
          <div class="tiles dos-t">
            <div class="tile"><div class="tile-v" id="t-marcado">—</div><div class="tile-n">marcado como tardío</div></div>
            <div class="tile"><div class="tile-v">{F.num(m_oof['ROC-AUC'])}</div><div class="tile-n">ROC-AUC (no cambia con el umbral)</div></div>
          </div>
          <div class="callout">{esc(H.LECTURA_UMBRAL)}</div>
        </div>
        <div><div class="mini-etq">Matriz de confusión (OOF) · umbral <span id="u-val2">{F.num(umbral, 2)}</span> · {miles(n_oof)} líneas</div>{grafico(cm0)}</div>
      </div>
    </div>
    <div class="panel-test">
      <div class="oof-cab test">{insignia('test')}<span>{esc(T.LECTURAS['umbral_test'])}. Sin control interactivo: el umbral no se ajusta con la prueba.</span></div>
      <div class="tiles dos-t">
        <div class="tile"><div class="tile-v">{F.pct(m_test["marcados_tardía"])}</div><div class="tile-n">marcado como tardío</div></div>
        <div class="tile"><div class="tile-v">{F.num(v_m)}</div><div class="tile-n">ROC-AUC en la prueba</div></div>
      </div>
    </div>"""

    # --- Calibración y diagnóstico
    calib = f"""
    {seccion(H.titulo_calibracion(), 'Calibración en la prueba', 'test',
             grafico(limpiar(F.fig_calibracion(), 420)) + conclusion(T.LECTURAS['calibracion']))}
    {seccion(H.TITULO_CELDAS, 'Probabilidad predicha frente a tasa observada en entrenamiento', 'test',
             grafico(limpiar(F.fig_celdas(), 360)) + conclusion(T.LECTURAS['celdas']))}
    {seccion(H.TITULO_SELECCION, 'Selección de conjuntos de variables (validación cruzada temporal)', 'train',
             grafico(limpiar(F.fig_seleccion(), 360)) + conclusion(T.LECTURAS['seleccion']))}
    {seccion(H.TITULO_APRENDIZAJE, 'Curva de aprendizaje', 'train',
             grafico(limpiar(F.fig_curva_aprendizaje(), 360)) + conclusion(T.LECTURAS['aprendizaje']))}
    {seccion(H.TITULO_SUBGRUPOS, 'Desempeño por subgrupo en la prueba', 'test',
             falso_select('Subgrupo', 'Mercado de destino') + grafico(limpiar(F.fig_subgrupos('market'), 360)) + conclusion(T.LECTURAS['subgrupos']))}"""

    t3 = subnav("mod", [("Desempeño", "desempeno", desempeno), ("ROC", "roc", roc_html), ("Matriz de confusión", "matriz", matriz),
                        ("Umbral", "umbral", umbral_html), ("Calibración y diagnóstico", "calib", calib)])

    # =====================================================================================================
    # Documento
    # =====================================================================================================
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DataCo · Dashboard (vista previa)</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<header class="app-cab"><div class="app-cab-in">
  <div><h1>Riesgo de entrega tardía en DataCo</h1>
  <p>Predicción sin fuga · Entregable 2 · la explicación detallada está en el Jupyter Book</p>
  <p class="autores">Autores: Juan David Polo y Gabriella Pallares</p></div>
  <div class="origen"><span>Origen de los datos</span>{insignia('train')}{insignia('oof')}{insignia('test')}</div>
</div></header>
<nav class="principal"><div class="principal-in" role="tablist">
  <button class="activa" data-p="p1" role="tab">1 · Contexto del problema</button>
  <button data-p="p2" role="tab">2 · EDA</button>
  <button data-p="p3" role="tab">3 · Modelos base (logístico)</button></div></nav>
<main class="contenido">
  <div class="aviso">Vista previa estática: los menús desplegables muestran su valor por defecto y funcionan en la app Dash. El umbral sí es interactivo aquí.</div>
  <div id="p1" class="panel activo">{t1}</div><div id="p2" class="panel">{t2}</div><div id="p3" class="panel">{t3}</div>
</main>
<script>
const UMBRALES = {json.dumps(tabla_u)};
function redimensionar(c) {{ c.querySelectorAll('.js-plotly-plot').forEach(g => Plotly.Plots.resize(g)); }}
document.querySelectorAll('.principal button').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('.principal button').forEach(x => x.classList.remove('activa'));
  document.querySelectorAll('.panel').forEach(x => x.classList.remove('activo'));
  b.classList.add('activa'); const p = document.getElementById(b.dataset.p); p.classList.add('activo'); redimensionar(p); window.scrollTo(0, 0);
}}));
document.querySelectorAll('.subnav button').forEach(b => b.addEventListener('click', () => {{
  const g = b.dataset.g;
  b.parentElement.querySelectorAll('button').forEach(x => x.classList.remove('activa'));
  document.querySelectorAll('.subpanel[data-g="' + g + '"]').forEach(x => x.classList.remove('activo'));
  b.classList.add('activa'); const p = document.getElementById(b.dataset.p); p.classList.add('activo'); redimensionar(p);
}}));
const sl = document.getElementById('u-slider');
function actualizarUmbral() {{
  const k = parseFloat(sl.value).toFixed(2), d = UMBRALES[k]; if (!d) return;
  const f = k.replace('.', ',');
  document.getElementById('u-val').textContent = f; document.getElementById('u-val2').textContent = f;
  document.getElementById('t-marcado').textContent = d.marcado;
  const gd = document.querySelector('#mod-umbral .js-plotly-plot'); if (gd) Plotly.restyle(gd, {{z: [d.z], text: [d.text]}}, [0]);
}}
sl.addEventListener('input', actualizarUmbral); actualizarUmbral();
</script></body></html>"""


if __name__ == "__main__":
    SALIDA.write_text(construir(), encoding="utf-8")
    print("HTML escrito en", SALIDA)
