"""Utilidades del proyecto: DataCo Smart Supply Chain for Big Data Analysis.

Entregable 1, ruta A (clasificación) con componente temporal (2.6), espacial (2.7)
y espacio-temporal (2.8). Contiene la lectura de los datos, la partición
cronológica con separación (*gap*), los pliegues temporales, las medidas de
asociación, las herramientas espaciales (haversine, I de Moran, LISA, Gi*), el
Pipeline y el bootstrap por pedido que usan los capítulos 01 a 12.

Compatibilidad: Python 3.10+ y las versiones fijadas en ``requirements.txt``.

Estructura de carpetas esperada (la raíz es la que contiene ``_config.yml``)::

    <raíz>/_config.yml
    <raíz>/data/DataCoSupplyChainDataset.csv   (o el .zip incluido en el repositorio)
    <raíz>/tables/  y  <raíz>/figures/          (se crean solas)
"""
from __future__ import annotations

import shutil
import urllib.request
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics as skm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_FILE = "DataCoSupplyChainDataset.csv"
ZIP_FILE = "DataCoSupplyChainDataset.zip"
ENCODING = "latin-1"                       # el CSV original no es UTF-8
KAGGLE_URL = "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
MENDELEY_URL = "https://data.mendeley.com/datasets/8gx2fvg2k6/5"
# Copias públicas idénticas (mismo MD5) usadas solo si falta el archivo local.
MIRROR_URLS = [
    "https://raw.githubusercontent.com/vanshsaini01/DataCO/main/DataCoSupplyChainDataset.csv",
    "https://raw.githubusercontent.com/rkbh199-rgb/Supply_Chain_Analytics/main/DataCoSupplyChainDataset.csv",
]
EXPECTED_ROWS = 180_519

TARGET = "late_delivery_risk"              # 1 = entrega tardía (días reales > días programados)
DATE_COL = "order_date"                    # fecha y hora del pedido: el momento de predicción
ORDER_COL = "order_id"                     # un pedido tiene varios ítems (filas) con la misma etiqueta
CUSTOMER_COL = "customer_id"
ITEM_COL = "order_item_id"                 # identificador de la fila
LAT, LON = "latitude", "longitude"
N_SPLITS = 5                               # pliegues de la validación cruzada temporal
TEST_FRACTION = 0.20                       # último 20 % del tiempo (en filas) para la prueba
GAP_DAYS = 7                               # separación train/test: la etiqueta tarda hasta 6 días en conocerse
CELL_DEG = 1.0                             # tamaño de la celda espacial (grados) para agregaciones y bloques
EARTH_RADIUS_KM = 6371.0088
# Orden único de las métricas: classification_metrics(), _weighted_metrics() y los intervalos bootstrap lo comparten.
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy", "ROC-AUC", "Average Precision", "Brier"]

# Nombres originales -> nombres de trabajo (snake_case). El diccionario conserva el original.
RENAME = {
    "Type": "payment_type",
    "Days for shipping (real)": "days_shipping_real",
    "Days for shipment (scheduled)": "days_shipping_scheduled",
    "Benefit per order": "benefit_per_order",
    "Sales per customer": "sales_per_customer",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "late_delivery_risk",
    "Category Id": "category_id",
    "Category Name": "category_name",
    "Customer City": "customer_city",
    "Customer Country": "customer_country",
    "Customer Email": "customer_email",
    "Customer Fname": "customer_fname",
    "Customer Id": "customer_id",
    "Customer Lname": "customer_lname",
    "Customer Password": "customer_password",
    "Customer Segment": "customer_segment",
    "Customer State": "customer_state",
    "Customer Street": "customer_street",
    "Customer Zipcode": "customer_zipcode",
    "Department Id": "department_id",
    "Department Name": "department_name",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Market": "market",
    "Order City": "order_city",
    "Order Country": "order_country",
    "Order Customer Id": "order_customer_id",
    "order date (DateOrders)": "order_date",
    "Order Id": "order_id",
    "Order Item Cardprod Id": "order_item_cardprod_id",
    "Order Item Discount": "order_item_discount",
    "Order Item Discount Rate": "order_item_discount_rate",
    "Order Item Id": "order_item_id",
    "Order Item Product Price": "order_item_product_price",
    "Order Item Profit Ratio": "order_item_profit_ratio",
    "Order Item Quantity": "order_item_quantity",
    "Sales": "sales",
    "Order Item Total": "order_item_total",
    "Order Profit Per Order": "order_profit_per_order",
    "Order Region": "order_region",
    "Order State": "order_state",
    "Order Status": "order_status",
    "Order Zipcode": "order_zipcode",
    "Product Card Id": "product_card_id",
    "Product Category Id": "product_category_id",
    "Product Description": "product_description",
    "Product Image": "product_image",
    "Product Name": "product_name",
    "Product Price": "product_price",
    "Product Status": "product_status",
    "shipping date (DateOrders)": "shipping_date",
    "Shipping Mode": "shipping_mode",
}
ORIGINAL_NAME = {v: k for k, v in RENAME.items()}

# Variables que solo se conocen DESPUÉS de despachar o entregar el pedido (fuga directa).
LEAKAGE_COLS = ["days_shipping_real", "delivery_status", "order_status", "shipping_date"]
# Identificadores (no son características del pedido).
ID_COLS = ["order_item_id", "order_id", "customer_id", "order_customer_id", "order_item_cardprod_id",
           "product_card_id", "category_id", "product_category_id", "department_id"]
# Datos personales o enmascarados (sin valor predictivo y con riesgo de reidentificación).
PII_COLS = ["customer_fname", "customer_lname", "customer_email", "customer_password", "customer_street",
            "customer_zipcode"]
# Columnas que son copia exacta de otra (se verifica en el capítulo 2).
DUPLICATE_PAIRS = [("order_customer_id", "customer_id"), ("order_profit_per_order", "benefit_per_order"),
                   ("order_item_total", "sales_per_customer"), ("order_item_product_price", "product_price"),
                   ("product_category_id", "category_id"), ("order_item_cardprod_id", "product_card_id")]

_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Rutas, gráficos y formato
# ---------------------------------------------------------------------------
def find_project_root(start: Path | str | None = None) -> Path:
    """Sube desde ``start`` (por defecto el directorio actual) hasta hallar ``_config.yml``."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "_config.yml").exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró _config.yml subiendo desde {here}. "
        "Ejecuta el notebook dentro de la carpeta del Jupyter Book."
    )


def configure_plots() -> None:
    """Estilo común de las figuras."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.titlesize": 12,
                         "axes.labelsize": 10, "figure.max_open_warning": 0})


def save_figure(fig, name: str, root: Path | str | None = None, dpi: int = 150) -> Path:
    root = Path(root) if root is not None else find_project_root()
    folder = root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def es(x, decimals: int = 1, pct: bool = False) -> str:
    """Formato numérico en español: 1.234,5 (y ' %' si pct=True, con x en escala 0-1)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    value = 100 * x if pct else x
    text = f"{value:,.{decimals}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return text + (" %" if pct else "")


# ---------------------------------------------------------------------------
# Lectura de datos
# ---------------------------------------------------------------------------
def ensure_data(root: Path | str | None = None) -> Path:
    """Devuelve la ruta del CSV. Orden: CSV en data/ -> ZIP incluido en data/ -> copia pública -> error explicativo."""
    root = Path(root) if root is not None else find_project_root()
    data = root / "data"
    data.mkdir(exist_ok=True)
    csv_path = data / DATA_FILE
    if csv_path.exists():
        return csv_path
    zip_path = data / ZIP_FILE
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            member = next(n for n in archive.namelist() if Path(n).name.lower() == DATA_FILE.lower())
            with archive.open(member) as source, open(csv_path, "wb") as target:
                shutil.copyfileobj(source, target)
        return csv_path
    errors = []
    for url in MIRROR_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=300) as response, open(csv_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
            return csv_path
        except Exception as error:      # sin red, enlace cambiado, etc.
            errors.append(type(error).__name__)
            csv_path.unlink(missing_ok=True)
    raise FileNotFoundError(
        f"No encuentro {csv_path} ni {zip_path} y no pude descargarlo ({', '.join(errors)}). "
        f"Descárgalo de {KAGGLE_URL} (o {MENDELEY_URL}) y copia {DATA_FILE} dentro de {data}."
    )


def _read_raw(root: Path) -> pd.DataFrame:
    """Lee el CSV por bloques (menos memoria pico, útil en equipos con poca RAM) y prepara la tabla."""
    chunks = []
    for chunk in pd.read_csv(ensure_data(root), encoding=ENCODING, chunksize=30_000):
        chunk = chunk.rename(columns=RENAME)
        for col in chunk.select_dtypes("object").columns:
            if col not in (DATE_COL, "shipping_date"):
                chunk[col] = chunk[col].str.strip()
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], format="%m/%d/%Y %H:%M")
        chunk["shipping_date"] = pd.to_datetime(chunk["shipping_date"], format="%m/%d/%Y %H:%M")
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    missing = [c for c in (TARGET, DATE_COL, ORDER_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"No encuentro las columnas {missing}. Columnas: {list(df.columns)[:20]}")
    # Columnas de un solo valor o vacías: se guardan como categoría para no ocupar memoria.
    for col in ("customer_email", "customer_password"):
        df[col] = df[col].astype("category")
    df = df.sort_values([DATE_COL, ORDER_COL, ITEM_COL]).reset_index(drop=True)
    return add_time_features(df)


def load_raw(root: Path | str | None = None) -> pd.DataFrame:
    """Tabla completa con nombres snake_case, fechas parseadas y variables derivadas sin etiqueta.

    Las derivadas solo usan información disponible al momento del pedido:
    componentes de calendario, número de ítems del pedido y carga de pedidos de
    los 7 días previos (ventana estrictamente pasada).
    """
    root = Path(root) if root is not None else find_project_root()
    key = ("raw", str(root))
    if key not in _CACHE:
        _CACHE[key] = _read_raw(root)
    return _CACHE[key].copy()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendario, componentes cíclicos, jornada, ítems por pedido y carga pasada de pedidos (sin usar la etiqueta)."""
    out = df.copy()
    d = out[DATE_COL]
    out["order_year"] = d.dt.year
    out["order_month"] = d.dt.month
    out["order_dow"] = d.dt.dayofweek
    out["order_hour"] = d.dt.hour
    out["order_weekend"] = (out["order_dow"] >= 5).astype(int)
    out["order_afternoon"] = (out["order_hour"] >= 12).astype(int)
    # Interacción modo de envío × jornada del pedido (ambos se conocen al comprar; motivada por el cap. 7).
    out["mode_x_daypart"] = out["shipping_mode"] + np.where(out["order_afternoon"] == 1, " | PM", " | AM")
    for name, period in (("order_hour", 24), ("order_dow", 7), ("order_month", 12)):
        angle = 2 * np.pi * (out[name] - (1 if name == "order_month" else 0)) / period
        out[f"{name}_sin"], out[f"{name}_cos"] = np.sin(angle), np.cos(angle)
    out["items_in_order"] = out.groupby(ORDER_COL)[ITEM_COL].transform("size")
    # Carga logística de los 7 días previos: pedidos (no ítems) con marca de tiempo estrictamente anterior.
    orders = out.groupby(ORDER_COL)[DATE_COL].first().sort_values()
    counts = pd.Series(1.0, index=orders.values)
    rolled = counts.rolling("7D", closed="left").sum().fillna(0.0)
    rolled = rolled.groupby(level=0).first()          # pedidos con la misma marca: solo cuenta lo anterior
    out["orders_prev_7d"] = out[DATE_COL].map(rolled).astype(float)
    return out


def split_dates(df: pd.DataFrame, test_fraction: float = TEST_FRACTION, gap_days: int = GAP_DAYS) -> tuple:
    """Corte cronológico: el último ``test_fraction`` de filas es prueba; se descartan ``gap_days`` intermedios."""
    cutoff = df[DATE_COL].quantile(1 - test_fraction).floor("D")
    return cutoff - pd.Timedelta(days=gap_days), cutoff


def _split(root: Path | str | None) -> tuple:
    """Partición fija: train < inicio del gap <= gap < corte <= test. Se guarda en tables/particion_pedidos.csv."""
    root = Path(root) if root is not None else find_project_root()
    key = ("split", str(root))
    if key not in _CACHE:
        raw = _CACHE.get(("raw", str(root)))
        if raw is None:                       # no se guarda la tabla completa en caché: ahorra memoria
            raw = _read_raw(root)
        gap_start, cutoff = split_dates(raw)
        conjunto = pd.Series(np.where(raw[DATE_COL] < gap_start, "train", np.where(raw[DATE_COL] < cutoff, "gap", "test")))
        path = root / "tables" / "particion_pedidos.csv"
        path.parent.mkdir(exist_ok=True)
        raw[[ITEM_COL, ORDER_COL, DATE_COL]].assign(conjunto=conjunto.to_numpy()).to_csv(path, index=False)
        train = raw[(conjunto == "train").to_numpy()].reset_index(drop=True)
        gap = raw[(conjunto == "gap").to_numpy()].reset_index(drop=True)
        test = raw[(conjunto == "test").to_numpy()].reset_index(drop=True)
        del raw
        _CACHE[key] = (train, gap, test, gap_start, cutoff)
    return _CACHE[key]


def _apply_population(df: pd.DataFrame, include_canceled: bool) -> pd.DataFrame:
    if include_canceled:
        return df.copy()
    return df[df["delivery_status"] != "Shipping canceled"].reset_index(drop=True)


def load_train(include_canceled: bool = True, root: Path | str | None = None) -> pd.DataFrame:
    """Entrenamiento. ``include_canceled=False`` aplica la población de modelado (solo pedidos despachados)."""
    return _apply_population(_split(root)[0], include_canceled)


def load_test(include_canceled: bool = True, root: Path | str | None = None) -> pd.DataFrame:
    return _apply_population(_split(root)[2], include_canceled)


def load_gap(root: Path | str | None = None) -> pd.DataFrame:
    return _split(root)[1].copy()


def split_boundaries(root: Path | str | None = None) -> tuple:
    """(inicio del gap, inicio de la prueba)."""
    _, _, _, gap_start, cutoff = _split(root)
    return gap_start, cutoff


# ---------------------------------------------------------------------------
# Validación cruzada temporal
# ---------------------------------------------------------------------------
def temporal_folds(dates, n_splits: int = N_SPLITS, gap_days: int = GAP_DAYS, min_train_fraction: float = 0.3) -> list:
    """Ventana creciente con *gap*: cada pliegue valida un bloque posterior y entrena solo con lo anterior al gap.

    Equivale a ``TimeSeriesSplit(gap=...)`` pero el gap se mide en días (no en filas), de modo que
    ningún pedido cuya etiqueta aún no se conocería entra al entrenamiento del pliegue.
    """
    d = pd.Series(pd.to_datetime(np.asarray(dates))).reset_index(drop=True)
    qs = np.linspace(min_train_fraction, 1.0, n_splits + 1)
    bounds = d.quantile(qs).to_list()
    folds = []
    for k in range(n_splits):
        start, end = bounds[k], bounds[k + 1]
        val = np.where((d >= start) & ((d < end) if k < n_splits - 1 else (d <= end)))[0]
        tr = np.where(d < start - pd.Timedelta(days=gap_days))[0]
        folds.append((tr, val))
    return folds


def spatial_cells(lat, lon, size: float = CELL_DEG) -> pd.Series:
    """Celda de una rejilla regular en grados (identificador «lat_lon» de la esquina suroeste)."""
    la = np.floor(np.asarray(lat, dtype=float) / size) * size
    lo = np.floor(np.asarray(lon, dtype=float) / size) * size
    return pd.Series([f"{a:.0f}_{b:.0f}" for a, b in zip(la, lo)])


# ---------------------------------------------------------------------------
# Diccionario de variables
# ---------------------------------------------------------------------------
_DESCRIPTIONS = {
    "payment_type": ("Tipo de transacción o medio de pago (DEBIT, TRANSFER, CASH, PAYMENT)", "categórica nominal", "no aplica"),
    "days_shipping_real": ("Días reales entre la compra y el envío/entrega", "numérica discreta", "días"),
    "days_shipping_scheduled": ("Días de envío programados (promesa al cliente)", "numérica discreta", "días"),
    "benefit_per_order": ("Ganancia por ítem de pedido", "numérica continua", "USD"),
    "sales_per_customer": ("Venta total por cliente en la línea (igual a order_item_total)", "numérica continua", "USD"),
    "delivery_status": ("Estado de la entrega: adelantada, a tiempo, tardía, cancelada", "categórica nominal", "no aplica"),
    "late_delivery_risk": ("Objetivo: 1 si la entrega fue tardía; 0 en otro caso", "objetivo binario", "no aplica"),
    "category_id": ("Código de la categoría del producto", "identificador", "código"),
    "category_name": ("Categoría del producto", "categórica nominal", "no aplica"),
    "customer_city": ("Ciudad del cliente", "categórica de alta cardinalidad", "no aplica"),
    "customer_country": ("País del cliente", "categórica nominal", "no aplica"),
    "customer_email": ("Correo del cliente (enmascarado)", "texto enmascarado", "no aplica"),
    "customer_fname": ("Nombre del cliente", "texto personal", "no aplica"),
    "customer_id": ("Identificador del cliente", "identificador / entidad", "código"),
    "customer_lname": ("Apellido del cliente", "texto personal", "no aplica"),
    "customer_password": ("Contraseña (enmascarada)", "texto enmascarado", "no aplica"),
    "customer_segment": ("Segmento: Consumer, Corporate, Home Office", "categórica nominal", "no aplica"),
    "customer_state": ("Estado del cliente", "categórica nominal", "no aplica"),
    "customer_street": ("Dirección del cliente", "texto personal", "no aplica"),
    "customer_zipcode": ("Código postal del cliente", "código postal", "código"),
    "department_id": ("Código del departamento de la tienda", "identificador", "código"),
    "department_name": ("Departamento de la tienda", "categórica nominal", "no aplica"),
    "latitude": ("Latitud de la tienda (según la ficha del dataset)", "coordenada", "grados (EPSG:4326)"),
    "longitude": ("Longitud de la tienda (según la ficha del dataset)", "coordenada", "grados (EPSG:4326)"),
    "market": ("Mercado de destino: Africa, Europe, LATAM, Pacific Asia, USCA", "categórica nominal", "no aplica"),
    "order_city": ("Ciudad de destino del pedido", "categórica de alta cardinalidad", "no aplica"),
    "order_country": ("País de destino del pedido", "categórica de alta cardinalidad", "no aplica"),
    "order_customer_id": ("Identificador del cliente en el pedido (copia de customer_id)", "identificador", "código"),
    "order_date": ("Fecha y hora del pedido", "fecha-hora", "minuto"),
    "order_id": ("Identificador del pedido (varios ítems por pedido)", "identificador / entidad", "código"),
    "order_item_cardprod_id": ("Código del producto en la línea (copia de product_card_id)", "identificador", "código"),
    "order_item_discount": ("Descuento de la línea", "numérica continua", "USD"),
    "order_item_discount_rate": ("Tasa de descuento de la línea", "numérica continua", "proporción"),
    "order_item_id": ("Identificador de la línea del pedido", "identificador", "código"),
    "order_item_product_price": ("Precio del producto sin descuento (copia de product_price)", "numérica continua", "USD"),
    "order_item_profit_ratio": ("Margen de ganancia de la línea", "numérica continua", "proporción"),
    "order_item_quantity": ("Cantidad de unidades de la línea", "numérica discreta", "unidades"),
    "sales": ("Venta bruta de la línea (precio × cantidad)", "numérica continua", "USD"),
    "order_item_total": ("Total de la línea tras descuento", "numérica continua", "USD"),
    "order_profit_per_order": ("Ganancia del pedido (copia de benefit_per_order)", "numérica continua", "USD"),
    "order_region": ("Región de destino (23 regiones)", "categórica nominal", "no aplica"),
    "order_state": ("Estado o provincia de destino", "categórica de alta cardinalidad", "no aplica"),
    "order_status": ("Estado del pedido (COMPLETE, PENDING, CANCELED, SUSPECTED_FRAUD...)", "categórica nominal", "no aplica"),
    "order_zipcode": ("Código postal de destino", "código postal", "código"),
    "product_card_id": ("Código del producto", "identificador", "código"),
    "product_category_id": ("Código de la categoría (copia de category_id)", "identificador", "código"),
    "product_description": ("Descripción del producto (vacía)", "texto", "no aplica"),
    "product_image": ("URL de la imagen del producto", "texto", "no aplica"),
    "product_name": ("Nombre del producto", "categórica de alta cardinalidad", "no aplica"),
    "product_price": ("Precio del producto", "numérica continua", "USD"),
    "product_status": ("Disponibilidad del producto (1 = no disponible)", "categórica binaria", "no aplica"),
    "shipping_date": ("Fecha y hora del despacho", "fecha-hora", "minuto"),
    "shipping_mode": ("Modo de envío: Standard, Second, First Class, Same Day", "categórica ordinal", "no aplica"),
    "order_year": ("Año del pedido (derivada)", "derivada temporal", "año"),
    "order_month": ("Mes del pedido (derivada)", "derivada temporal", "mes"),
    "order_dow": ("Día de la semana del pedido, 0 = lunes (derivada)", "derivada temporal", "día"),
    "order_hour": ("Hora del pedido (derivada)", "derivada temporal", "hora"),
    "order_weekend": ("1 si el pedido se hizo en sábado o domingo (derivada)", "derivada temporal", "no aplica"),
    "order_afternoon": ("1 si el pedido se hizo a partir de las 12:00 (derivada)", "derivada temporal", "no aplica"),
    "mode_x_daypart": ("Interacción modo de envío × jornada (AM/PM) del pedido (derivada)", "derivada categórica", "no aplica"),
    "items_in_order": ("Número de líneas del pedido (derivada, conocida al cerrar la compra)", "derivada numérica", "ítems"),
    "orders_prev_7d": ("Pedidos registrados en los 7 días previos (derivada, carga logística pasada)", "derivada temporal", "pedidos"),
}


def build_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Diccionario verificable: nombre original, significado, tipo, unidad, valores, faltantes y papel."""
    rows = []
    for col in df.columns:
        if col.endswith(("_sin", "_cos")):
            continue
        desc, kind, unit = _DESCRIPTIONS.get(col, ("Variable sin documentación", "—", "—"))
        unique = df[col].nunique(dropna=True)
        if unique <= 10:
            values = ", ".join(map(str, sorted(df[col].dropna().astype(str).unique())))
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            values = f"{df[col].min():%Y-%m-%d} a {df[col].max():%Y-%m-%d}"
        elif pd.api.types.is_numeric_dtype(df[col]):
            values = f"rango: {df[col].min():.4g} a {df[col].max():.4g}"
        else:
            values = f"{unique} categorías"
        if col == TARGET:
            role = "variable objetivo"
        elif col in LEAKAGE_COLS:
            role = "posterior al pedido; excluida (fuga)"
        elif col in PII_COLS:
            role = "dato personal; excluida"
        elif col in ID_COLS:
            role = "identificador; excluido (orden/cliente solo para agrupar)"
        elif col == DATE_COL:
            role = "marca temporal; define la partición"
        else:
            role = "predictora candidata"
        rows.append({"variable": col, "nombre_original": ORIGINAL_NAME.get(col, "(derivada)"), "tipo": kind,
                     "significado": desc, "unidad": unit, "valores": values, "únicos": unique,
                     "faltantes": int(df[col].isna().sum()), "faltantes_%": round(100 * df[col].isna().mean(), 2),
                     "papel": role})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Resumen y asociación
# ---------------------------------------------------------------------------
def numeric_summary(X: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos por variable numérica, incluida la proporción de outliers por regla IQR."""
    q = X.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).T
    q1, med, q3 = q[0.25], q[0.5], q[0.75]
    iqr = q3 - q1
    outside = X.lt(q1 - 1.5 * iqr, axis=1) | X.gt(q3 + 1.5 * iqr, axis=1)
    return pd.DataFrame({
        "count": X.count(), "faltantes": X.isna().sum(), "media": X.mean(), "mediana": med,
        "desv_est": X.std(), "mínimo": X.min(), "p01": q[0.01], "p25": q1, "p75": q3, "p99": q[0.99],
        "máximo": X.max(), "IQR": iqr, "asimetría": X.skew(), "curtosis": X.kurt(),
        "ceros_%": 100 * X.eq(0).mean(), "únicos": X.nunique(),
        "outliers_IQR_%": 100 * outside.sum() / X.notna().sum().replace(0, np.nan),
    })


def cramers_v(x: pd.Series, y: pd.Series) -> tuple:
    """V de Cramér con corrección de sesgo, χ² y p-valor. Devuelve (v, chi2, p)."""
    x_filled = x.astype("object").where(x.notna(), "Missing")
    table = pd.crosstab(x_filled, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.values.sum()
    rows, cols = table.shape
    phi2 = max(0.0, chi2 / n - (cols - 1) * (rows - 1) / (n - 1))
    r_c, k_c = rows - (rows - 1) ** 2 / (n - 1), cols - (cols - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(min(k_c - 1, r_c - 1), 1e-12))), float(chi2), float(p)


def fdr_bh(pvalues) -> np.ndarray:
    """P-valores ajustados por Benjamini-Hochberg (los NaN se ignoran)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return adjusted
    values = p[valid]
    order = np.argsort(values, kind="mergesort")
    scaled = values[order] * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[valid] = result
    return adjusted


def variance_inflation_factors(array) -> np.ndarray:
    """VIF de cada columna (columnas centradas): 1 / (1 - R²) al regresarla contra las demás."""
    X = np.asarray(array, dtype=float)
    X = X - X.mean(axis=0)
    vif = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        y, others = X[:, j], np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        residual = y - others @ beta
        total = float(y @ y)
        r2 = 1.0 - float(residual @ residual) / total if total > 0 else 1.0
        vif[j] = np.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
    return vif


def univariate_auc(x: pd.Series, y: pd.Series) -> float:
    """max(AUC, 1-AUC) de una variable numérica sola (0,5 = sin señal; 1 = separación perfecta)."""
    x, y = pd.Series(np.asarray(x, dtype=float)), pd.Series(np.asarray(y))
    mask = x.notna()
    if mask.sum() < 50 or y[mask].nunique() < 2 or x[mask].nunique() < 2:
        return np.nan
    a = skm.roc_auc_score(y[mask], x[mask])
    return float(max(a, 1 - a))


def categorical_auc_oof(x: pd.Series, y: pd.Series, folds: list, smooth: float = 20.0) -> float:
    """AUC de una categórica sola, codificada con la tasa del objetivo calculada fuera de pliegue.

    Con pliegues temporales solo se puntúan las filas de validación (las primeras nunca se validan).
    """
    s = pd.Series(x.astype("object").where(x.notna(), "Missing").to_numpy())
    yy = pd.Series(np.asarray(y))
    encoded = np.full(len(yy), np.nan)
    for tr, va in folds:
        prior = float(yy.iloc[tr].mean())
        g = pd.DataFrame({"c": s.iloc[tr].to_numpy(), "y": yy.iloc[tr].to_numpy()}).groupby("c")["y"].agg(["sum", "count"])
        rate = (g["sum"] + smooth * prior) / (g["count"] + smooth)
        encoded[va] = s.iloc[va].map(rate).fillna(prior).to_numpy()
    mask = ~np.isnan(encoded)
    a = skm.roc_auc_score(yy[mask], encoded[mask])
    return float(max(a, 1 - a))


def wilson_interval(k, n, z: float = 1.96) -> tuple:
    """Intervalo de Wilson para una proporción k/n (vectorizado)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def rate_table(x: pd.Series, y: pd.Series, min_n: int = 100, top: int | None = None) -> pd.DataFrame:
    """Tasa del objetivo por categoría con intervalo de Wilson; ordena por tamaño de categoría."""
    s = x.astype("object").where(x.notna(), "Missing")
    g = (pd.DataFrame({"categoria": s.to_numpy(), "y": np.asarray(y)}).groupby("categoria")["y"]
         .agg(["sum", "count"]).reset_index())
    g = g[g["count"] >= min_n].sort_values("count", ascending=False)
    if top:
        g = g.head(top)
    lo, hi = wilson_interval(g["sum"], g["count"])
    return pd.DataFrame({"categoria": g["categoria"].astype(str).to_numpy(), "n": g["count"].to_numpy(),
                         "tasa": (g["sum"] / g["count"]).to_numpy(), "ic_inf": lo, "ic_sup": hi})


def psi(expected, actual, bins: int = 10, categorical: bool = False) -> float:
    """Population Stability Index entre dos muestras (deriva de covariables). <0,1 estable; >0,25 deriva fuerte."""
    eps = 1e-6
    if categorical:
        e = pd.Series(expected).astype(str).value_counts(normalize=True)
        a = pd.Series(actual).astype(str).value_counts(normalize=True)
        idx = e.index.union(a.index)
        e, a = e.reindex(idx, fill_value=0).to_numpy() + eps, a.reindex(idx, fill_value=0).to_numpy() + eps
    else:
        edges = np.unique(np.quantile(np.asarray(expected, dtype=float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return psi(expected, actual, categorical=True)
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected) + eps
        a = np.histogram(actual, edges)[0] / len(actual) + eps
    return float(np.sum((a - e) * np.log(a / e)))


# ---------------------------------------------------------------------------
# Herramientas espaciales
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia de gran círculo en km (no se usa distancia euclidiana sobre grados)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_weights(lat, lon, k: int = 8) -> np.ndarray:
    """Matriz de pesos espaciales k vecinos más cercanos (haversine), estandarizada por filas."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    n = len(lat)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, np.inf)
    k = min(k, n - 1)
    W = np.zeros((n, n))
    nn = np.argsort(D, axis=1)[:, :k]
    W[np.repeat(np.arange(n), k), nn.ravel()] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def morans_i(values, W, permutations: int = 999, seed: int = RANDOM_STATE) -> dict:
    """I de Moran global con prueba de permutación (pesos estandarizados por filas)."""
    z = np.asarray(values, float) - np.mean(values)
    n, s0 = len(z), W.sum()
    denom = float(z @ z)
    observed = (n / s0) * float(z @ W @ z) / denom
    rng = np.random.default_rng(seed)
    sims = np.empty(permutations)
    for b in range(permutations):
        zp = rng.permutation(z)
        sims[b] = (n / s0) * float(zp @ W @ zp) / denom
    p = (1 + np.sum(np.abs(sims - sims.mean()) >= abs(observed - sims.mean()))) / (permutations + 1)
    return {"I": observed, "E[I]": -1 / (n - 1), "p_permutacion": float(p), "z": float((observed - sims.mean()) / sims.std())}


def local_morans(values, W, permutations: int = 999, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """LISA (I de Moran local) con permutación condicional y cuadrante (HH, LL, HL, LH)."""
    x = np.asarray(values, float)
    z = (x - x.mean()) / x.std()
    lag = W @ z
    Ii = z * lag
    rng = np.random.default_rng(seed)
    n = len(z)
    p = np.empty(n)
    for i in range(n):
        others = np.delete(z, i)
        neigh = W[i] > 0
        k = int(neigh.sum())
        w = W[i, neigh]
        sims = np.array([z[i] * (w @ rng.choice(others, k, replace=False)) for _ in range(permutations)])
        p[i] = (1 + np.sum(np.abs(sims) >= abs(Ii[i]))) / (permutations + 1)
    quad = np.select([(z > 0) & (lag > 0), (z < 0) & (lag < 0), (z > 0) & (lag < 0)], ["HH", "LL", "HL"], "LH")
    return pd.DataFrame({"z": z, "lag": lag, "I_local": Ii, "p": p, "cuadrante": quad})


def getis_ord_gi_star(values, W) -> np.ndarray:
    """Estadístico Gi* (z) de Getis-Ord con la propia celda incluida en el vecindario."""
    x = np.asarray(values, float)
    n = len(x)
    Wb = (W > 0).astype(float)
    np.fill_diagonal(Wb, 1.0)
    xbar, s = x.mean(), np.sqrt((x ** 2).mean() - x.mean() ** 2)
    wsum, w2sum = Wb.sum(axis=1), (Wb ** 2).sum(axis=1)
    num = Wb @ x - xbar * wsum
    den = s * np.sqrt((n * w2sum - wsum ** 2) / (n - 1))
    return num / den


def clark_evans(lat, lon, area_km2: float) -> dict:
    """Índice del vecino más cercano (Clark-Evans): R<1 agrupado, R≈1 aleatorio, R>1 disperso."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, np.inf)
    observed = float(D.min(axis=1).mean())
    density = len(lat) / area_km2
    expected = 0.5 / np.sqrt(density)
    se = 0.26136 / np.sqrt(len(lat) * density)
    return {"R": observed / expected, "distancia_media_obs_km": observed, "distancia_esperada_km": expected,
            "z": (observed - expected) / se}


def ripley_k(lat, lon, radii_km, area_km2: float) -> pd.DataFrame:
    """Función K de Ripley (sin corrección de borde) y su transformación L(r) - r."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, np.inf)
    n = len(lat)
    K = np.array([area_km2 * (D < r).sum() / (n * (n - 1)) for r in radii_km])
    L = np.sqrt(K / np.pi)
    return pd.DataFrame({"r_km": radii_km, "K": K, "L_menos_r": L - np.asarray(radii_km)})


# ---------------------------------------------------------------------------
# Modelo, métricas y bootstrap por pedido
# ---------------------------------------------------------------------------
def make_preprocessor(num_lin: list, num_log: list, cat: list, min_frequency: float = 0.005) -> ColumnTransformer:
    """Imputación, transformación, escalado y codificación; todo se ajusta solo con el entrenamiento."""
    parts = []
    if num_lin:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num_lin))
    if num_log:
        parts.append(("numlog", Pipeline([("imp", SimpleImputer(strategy="median")),
                                          ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                          ("sc", StandardScaler())]), num_log))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing",
                                                               keep_empty_features=True)),
                                       ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                            min_frequency=min_frequency,
                                                            drop="first"))]), cat))
    return ColumnTransformer(parts, remainder="drop")


def make_model_pipeline(num_lin: list, num_log: list, cat: list, C: float = 1.0,
                        class_weight=None, max_iter: int = 3000) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(num_lin, num_log, cat)),
                     ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter,
                                                random_state=RANDOM_STATE))])


def classification_metrics(y_true, y_pred, y_score) -> dict:
    """Métricas de clasificación binaria (clase positiva = entrega tardía). El orden de las claves es METRIC_NAMES."""
    y_true, y_pred, y_score = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score, float)
    return {"Accuracy": skm.accuracy_score(y_true, y_pred),
            "Precision": skm.precision_score(y_true, y_pred, zero_division=0),
            "Recall": skm.recall_score(y_true, y_pred, zero_division=0),
            "F1": skm.f1_score(y_true, y_pred, zero_division=0),
            "Balanced Accuracy": skm.balanced_accuracy_score(y_true, y_pred),
            "ROC-AUC": skm.roc_auc_score(y_true, y_score),
            "Average Precision": skm.average_precision_score(y_true, y_score),
            "Brier": skm.brier_score_loss(y_true, y_score)}


def _weighted_metrics(y, pred, score, w) -> list:
    """Las mismas métricas con pesos de remuestreo (bootstrap por pedido); el orden es METRIC_NAMES."""
    tp = np.sum(w * ((y == 1) & (pred == 1)))
    fp = np.sum(w * ((y == 0) & (pred == 1)))
    fn = np.sum(w * ((y == 1) & (pred == 0)))
    tn = np.sum(w * ((y == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
    return [(tp + tn) / w.sum(), precision, recall, f1, (recall + specificity) / 2,
            skm.roc_auc_score(y, score, sample_weight=w),
            skm.average_precision_score(y, score, sample_weight=w),
            float(np.sum(w * (score - y) ** 2) / w.sum())]


def _cluster_weights(groups, rng) -> np.ndarray:
    """Pesos de un remuestreo de pedidos completos (cada ítem hereda el peso de su pedido)."""
    codes, n_groups = pd.factorize(groups)[0], groups.nunique()
    counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups))
    return counts[codes].astype(float)


def cluster_bootstrap(y, pred, score, groups, B: int = 1000, seed: int = RANDOM_STATE, alpha: float = 0.05) -> pd.DataFrame:
    """Intervalos bootstrap percentil remuestreando PEDIDOS (no filas): los ítems de un pedido comparten etiqueta."""
    y, pred, score = np.asarray(y), np.asarray(pred), np.asarray(score, dtype=float)
    groups = pd.Series(np.asarray(groups))
    rng = np.random.default_rng(seed)
    draws = np.array([_weighted_metrics(y, pred, score, _cluster_weights(groups, rng)) for _ in range(B)])
    point = _weighted_metrics(y, pred, score, np.ones(len(y)))
    level = f"{100 * (1 - alpha):g} %"
    return pd.DataFrame({"Estimación": point,
                         f"IC {level} inferior": np.percentile(draws, 100 * alpha / 2, axis=0),
                         f"IC {level} superior": np.percentile(draws, 100 * (1 - alpha / 2), axis=0)},
                        index=pd.Index(METRIC_NAMES, name="Métrica"))


def paired_cluster_bootstrap(y, pred_a, score_a, pred_b, score_b, groups, B: int = 1000,
                             seed: int = RANDOM_STATE, alpha: float = 0.05) -> pd.DataFrame:
    """Diferencia A - B por métrica con los mismos pedidos remuestreados en ambos modelos."""
    y = np.asarray(y)
    pa, sa, pb, sb = np.asarray(pred_a), np.asarray(score_a, float), np.asarray(pred_b), np.asarray(score_b, float)
    groups = pd.Series(np.asarray(groups))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        w = _cluster_weights(groups, rng)
        diffs.append(np.array(_weighted_metrics(y, pa, sa, w)) - np.array(_weighted_metrics(y, pb, sb, w)))
    diffs = np.array(diffs)
    ones = np.ones(len(y))
    point = np.array(_weighted_metrics(y, pa, sa, ones)) - np.array(_weighted_metrics(y, pb, sb, ones))
    return pd.DataFrame({"Diferencia": point, "IC inferior": np.percentile(diffs, 100 * alpha / 2, axis=0),
                         "IC superior": np.percentile(diffs, 100 * (1 - alpha / 2), axis=0)},
                        index=pd.Index(METRIC_NAMES, name="Métrica"))


def odds_ratio_table(pipe: Pipeline) -> pd.DataFrame:
    """Coeficientes y odds ratios con nombres de salida del Pipeline."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_.ravel()
    table = pd.DataFrame({"variable": names, "coeficiente": coef, "odds_ratio": np.exp(coef)})
    table["tipo"] = np.where(table["variable"].str.startswith("cat__"), "indicador de categoría", "numérica estandarizada")
    return (table.assign(abs_coef=lambda d: d["coeficiente"].abs())
            .sort_values("abs_coef", ascending=False).drop(columns="abs_coef").reset_index(drop=True))


def onehot_reference_table(pipe: Pipeline, categorical_columns: list[str]) -> pd.DataFrame:
    """Categoría de referencia eliminada por ``OneHotEncoder(drop='first')``."""
    encoder = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"]
    rows = []
    for variable, categories, drop_idx in zip(categorical_columns, encoder.categories_, encoder.drop_idx_):
        reference = categories[int(drop_idx)] if drop_idx is not None else "(ninguna)"
        rows.append({"variable": variable, "categoría_referencia": reference})
    return pd.DataFrame(rows)
