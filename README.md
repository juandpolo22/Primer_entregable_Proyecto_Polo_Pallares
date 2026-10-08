# Riesgo de entrega tardía en DataCo (sin fuga) — Entregables 1 y 2

Primer entregable de Machine Learning construido como **Jupyter Book** sobre
[DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
(Constante, Silva y Pereira, 2019; [Mendeley Data, DOI 10.17632/8gx2fvg2k6.5](https://data.mendeley.com/datasets/8gx2fvg2k6/5), CC BY 4.0).

**Pregunta:** al crear un pedido, y solo con información disponible en ese momento,
¿llegará tarde su entrega (`Late_delivery_risk = 1`)?

**Métrica principal: ROC-AUC.** Versión **sin fuga**: se excluyen `Days for shipping (real)`,
`Delivery Status`, `Order Status` y `shipping date`, que se conocen después del despacho.

| Resultado (prueba cronológica) | ROC-AUC | Accuracy |
|---|---|---|
| Dummy — clase mayoritaria / prior | 0,500 | 0,573 |
| **Regresión logística (modo de envío × jornada)** | **0,775** (IC 95 %: 0,767–0,783) | 0,725 |
| Referencia empírica sin fuga (tabla de tasas modo × jornada) | 0,775 | 0,725 |
| *Con fuga* (días reales de envío, solo demostración) | *0,975* | — |

## Estructura

```
_config.yml, _toc.yml      configuración y tabla de contenidos del libro
notebooks/                 12 capítulos, ya ejecutados (con salidas)
src/project_utils.py       lectura de datos, partición cronológica con gap, pliegues temporales,
                           medidas de asociación, herramientas espaciales, Pipeline y bootstrap
data/                      DataCoSupplyChainDataset.zip (CSV original, se extrae solo)
tables/, figures/          resultados regenerados por los notebooks
colab/                     lanzador para Google Colab
references.bib             bibliografía
requirements.txt           versiones exactas verificadas (Python 3.11)
```

## Ver el libro

### Opción 1: construir localmente (recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter-book build .
```

Abre `_build/html/index.html` en el navegador. Como los notebooks ya tienen sus
salidas, la construcción tarda segundos (`execute_notebooks: "off"`).

Para **re-ejecutar todo desde los datos** (≈ 5 minutos):

1. En `_config.yml` cambia `execute_notebooks: "off"` por `execute_notebooks: force`.
2. Ejecuta `jupyter-book clean . --all` y luego `jupyter-book build .`

o bien abre los notebooks `01` a `12` en orden en Jupyter/VS Code.

### Opción 2: Google Colab

Abre `colab/DataCo_Jupyter_Book_Colab.ipynb` en Colab y ejecuta las celdas: clona
este repositorio, instala dependencias, construye el libro y lo muestra en el propio
notebook (también permite descargar el HTML en un ZIP).

### Opción 3: GitHub Pages

```bash
pip install ghp-import
jupyter-book build .
ghp-import -n -p -f _build/html
```

## Decisiones metodológicas principales

- **Partición cronológica** 80/20 por fecha del pedido con **gap de 7 días** (la etiqueta se conoce
  hasta 6 días después). Validación cruzada temporal con ventana creciente y el mismo gap.
  Ningún pedido queda dividido entre entrenamiento y prueba.
- **Unidad de observación:** la **línea de pedido** (un pedido tiene varias líneas y todas comparten la etiqueta).
- **Población:** se excluyen pedidos cancelados o sospechosos de fraude (etiqueta 0 por construcción, no son entregas a tiempo).
- **Bootstrap por pedido** (las líneas de un pedido comparten etiqueta) para los intervalos de confianza.
- Semilla global **42**; constantes al inicio de `src/project_utils.py`.

## Hallazgo clave

Sin fuga, la etiqueta depende del **modo de envío** (First Class siempre tarde; Second Class 80 %;
Standard 40 %, al azar) y, en **Same Day**, de la **hora de corte del mediodía**. Una regresión logística
con la interacción modo × jornada alcanza prácticamente la misma discriminación que la tabla de tasas
modo × jornada (referencia empírica sin fuga, ROC-AUC ≈ 0,775; no se afirma que sea un máximo demostrado). El famoso
«AUC ≈ 1» de DataCo solo aparece al usar variables posteriores al pedido.

## Entregable 2: Dashboard en Dash/Plotly

Tres pestañas principales: **(1) Contexto del problema**, **(2) EDA** (subpestañas: objetivo y calidad, relaciones con el
objetivo, tiempo y espacio, auditoría de fuga) y **(3) Modelos base** (regresión logística; subpestañas: desempeño, ROC, matriz de confusión,
umbral, calibración y diagnóstico). Autores: **Juan David Polo y Gabriella Pallares**. El dashboard muestra solo el **ROC-AUC** (métrica principal); las demás métricas siguen en las tablas y en los capítulos 11 y 15. El informe con la interpretación de cada resultado del dashboard (y detalles que no aparecen
en él) son los capítulos **13, 14 y 15** del Jupyter Book.

Cada resultado lleva una insignia de origen: **TRAIN** (entrenamiento), **OOF** (predicciones fuera de pliegue del entrenamiento) y
**TEST** (prueba cronológica). El control de umbral usa solo OOF; la prueba se evalúa con el umbral 0,50 fijado de antemano.

### 1. Instalación

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Ejecución local

```bash
python dashboard/app.py                 # abrir http://127.0.0.1:8050 (puerto configurable: PORT=9000 python dashboard/app.py)
```

- `dashboard/datos/` ya viene generado. Si cambias datos o modelos: `python dashboard/preparar_datos.py` (≈ 1 min) y después
  `python dashboard/verificar_consistencia.py` (comprueba que dashboard, CSV y tablas del libro usan las mismas cifras).
- Vista previa estática (sin servidor): abre `dashboard/vista_previa_dashboard.html`
  (se regenera con `python dashboard/exportar_html.py`).

### 3. Construir el Jupyter Book

```bash
jupyter-book build .                    # abrir _build/html/index.html
```

Los capítulos 13–15 muestran las mismas figuras del dashboard (se generan con `dashboard/figuras.py`). Para re-ejecutarlos:
`jupyter nbconvert --to notebook --execute --inplace notebooks/13_dashboard_contexto.ipynb` (igual con el 14 y el 15).

### 4. Despliegue en Render

El repositorio incluye `render.yaml` y `gunicorn` en `requirements.txt`. El servicio se inicia desde la raíz del proyecto con:

```bash
gunicorn dashboard.app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

1. Sube el repositorio completo a GitHub (incluida la carpeta `dashboard/datos/`; el dashboard no necesita el CSV original).
2. En Render: **New + → Blueprint**, elige el repositorio y confirma el servicio `dataco-dashboard` (plan gratuito).
   Alternativa sin Blueprint: **New + → Web Service**, *Build Command* `pip install -r requirements.txt`,
   *Start Command* el comando de arriba y la variable `PYTHON_VERSION=3.11.9`.
3. Al terminar el despliegue, la URL pública abre el dashboard. En el plan gratuito el servicio se duerme tras unos minutos de
   inactividad y la primera visita tarda en responder.

| Archivo | Función |
|---|---|
| `dashboard/preparar_datos.py` | Lee el CSV y las tablas del Entregable 1, ajusta los modelos, genera las predicciones OOF y guarda agregados pequeños |
| `dashboard/figuras.py` | Figuras Plotly compartidas por el dashboard, la vista previa y el Jupyter Book |
| `dashboard/estilo_figuras.py` | Apariencia de las figuras (colores, leyendas, márgenes); no modifica datos |
| `dashboard/hallazgos.py` | Títulos que comunican el hallazgo de cada gráfico, calculados desde los artefactos |
| `dashboard/assets/estilo.css` | Estilos compartidos por la app Dash y la vista previa HTML |
| `dashboard/textos.py` | Textos del dashboard (sus cifras se calculan desde los artefactos) |
| `dashboard/app.py` | App Dash (3 pestañas, subpestañas y controles); expone `server` para gunicorn |
| `dashboard/exportar_html.py` | Exporta la vista previa HTML |
| `dashboard/verificar_consistencia.py` | Comprueba que dashboard, CSV y tablas del libro coinciden |
| `render.yaml` | Configuración del servicio en Render |
