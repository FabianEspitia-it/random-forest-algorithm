# Random Forest para series financieras (BVC)

Este repositorio contiene dos scripts para descargar, visualizar y modelar series de tiempo de acciones listadas en la BVC (ej. `ECOPETROL.CL`, `ISA.CL`, `GRUPOARGOS.CL`, `GEB.CL`) usando `yfinance`, `pandas`, `matplotlib` y `scikit-learn`.

- `data.py`: descarga históricos, limpia columnas y genera gráficos por ticker (Precio de Cierre y Volumen).
- `random_forest.py`: construye variables técnicas, entrena un `RandomForestRegressor` por ticker usando datos hasta una fecha de corte y evalúa con datos posteriores, mostrando gráficos y el error MAE.


## Requisitos

- Python 3.12+
- Dependencias principales:
  - yfinance
  - pandas
  - numpy
  - scikit-learn
  - matplotlib

Instalación rápida (Windows):

```bash
python -m venv venv
.\venv\Scripts\activate
pip install yfinance pandas numpy scikit-learn matplotlib
```


## Estructura del proyecto

```
random-forest-algorithm/
├─ data.py              # Descarga/limpieza y visualización (precio y volumen)
├─ random_forest.py     # Features técnicas, entrenamiento RF y evaluación
└─ README.md
```


## Uso

Activar el entorno (Windows):

```bash
.\venv\Scripts\activate
```

- Visualización de históricos (precio y volumen) por empresa:

```bash
python data.py
```

- Entrenamiento y evaluación del modelo por ticker:

```bash
python random_forest.py
```

Se abrirán ventanas de `matplotlib` con las gráficas generadas.


## Qué hace cada script

- `data.py`
  - Descarga datos ajustados (`auto_adjust=True`) con `yfinance` para tickers definidos.
  - Selecciona columnas: `Date`, `Ticker`, `Close`, `Volume` y concatena en un único DataFrame.
  - Grafica por ticker:
    - Subgráfico 1: serie del precio de cierre.
    - Subgráfico 2: área del volumen transado (con ejes de fecha configurados).

- `random_forest.py`
  - Descarga datos diarios ajustados y crea variables/indicadores:
    - `Target_Return`: retorno del día siguiente (`Close.pct_change().shift(-1)`).
    - `Return_1d`, `Return_5d`
    - `SMA_10` y `Dist_SMA_10` (= `Close / SMA_10`)
    - `Volatility` (desv. estándar móvil 10)
    - `RSI` (14)
  - Partición temporal por fecha de corte (`DEADLINE`):
    - Entrenamiento: fechas `<= DEADLINE`
    - Prueba: fechas `> DEADLINE`
  - Modelo: `RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=5, random_state=42)`
  - Evalúa prediciendo el retorno del día siguiente y lo convierte a precio estimado:
    - `precio_estimado_t ≈ precio_real_t * (1 + retorno_predicho_t)`
  - Reporta MAE sobre precios estimados y grafica “Realidad vs Modelo” para el tramo de prueba.


## Parámetros y personalización

- Tickers (ambos scripts): `tickers_bvc = ['ECOPETROL.CL', 'ISA.CL', 'GRUPOARGOS.CL', 'GEB.CL']`
- Fecha de corte (`random_forest.py`): `DEADLINE = "2024-12-31"`
- Indicadores:
  - Ventanas: `SMA_10` (10), `Volatility` (10), `RSI` (14)
- Hiperparámetros del bosque aleatorio:
  - `n_estimators=150`, `max_depth=5`, `min_samples_leaf=5`, `random_state=42`

Para usar otros activos, reemplace los símbolos en `tickers_bvc`. Los sufijos `.CL` provienen de Yahoo Finance.


## Notas y limitaciones

- Los precios se descargan con `auto_adjust=True` (incluye ajustes corporativos).
- La conversión de retorno predicho a precio estimado es aproximada y se usa con fines visuales/indicativos.
- No incluye costos de transacción, fricciones de mercado ni lógica de ejecución.
- Si no hay datos posteriores a `DEADLINE` para un ticker, el script avisa y omite su evaluación.


## Solución de problemas

- “No hay datos posteriores a DEADLINE”: cambie `DEADLINE` o el ticker.
- Las ventanas de gráficos no aparecen: ejecute en un entorno con GUI o guarde las figuras con `plt.savefig(...)`.
- Columnas tipo `MultiIndex`: el código ya aplanará los nombres automáticamente si ocurre con `yfinance`.


## Créditos

- Datos: `yfinance`
- Modelado: `scikit-learn`
- Manipulación y gráficos: `pandas`, `matplotlib`


## Licencia

Sin licencia explícita en el repositorio. Si planea distribuir/modificar, considere añadir una licencia (por ejemplo, MIT).