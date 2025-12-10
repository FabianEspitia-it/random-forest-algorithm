"""Train and evaluate a Random Forest regressor on stock returns.

Pipeline:
1) Download adjusted daily data for each ticker from Yahoo Finance.
2) Engineer technical features (returns, moving average distance, volatility, RSI).
3) Split by a time-based cutoff (DEADLINE) to simulate a true out-of-sample test.
4) Train RandomForestRegressor on data <= DEADLINE, test on data > DEADLINE.
5) Convert predicted next-day returns into price estimates for visualization,
   and report MAE at the price level.
"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Yahoo Finance tickers used in this project (Colombian listings with .CL suffix).
# Replace or extend this list to evaluate different assets.
tickers_bvc = ['ECOPETROL.CL', 'ISA.CL', 'GRUPOARGOS.CL', 'GEB.CL']
# Temporal split cutoff used for backtesting. Training uses rows <= DEADLINE,
# testing uses rows > DEADLINE.
DEADLINE = "2024-12-31"  


def calculate_rsi(data, window=14):
    """Compute the Relative Strength Index (RSI).

    Args:
        data (pd.Series): Series of closing prices.
        window (int): Lookback window for RSI calculation.

    Returns:
        pd.Series: RSI values in the range [0, 100].
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(df):
    """Build supervised learning dataset with technical features.

    Creates a next-day target return and several predictors:
      - Return_1d, Return_5d
      - SMA_10 and Dist_SMA_10 (price relative to its 10-day SMA)
      - Volatility (10-day rolling std)
      - RSI (14)

    Rows with missing values (from rolling windows) are dropped.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'Close' column.

    Returns:
        pd.DataFrame: Cleaned feature matrix with target.
    """
    df_ml = df.copy()

    # Supervised target: next-day return derived from close prices.
    df_ml['Target_Return'] = df_ml['Close'].pct_change().shift(-1)

    # Lightweight feature set mixing momentum and mean-reversion signals.
    df_ml['Return_1d'] = df_ml['Close'].pct_change()
    df_ml['Return_5d'] = df_ml['Close'].pct_change(5)
    df_ml['SMA_10'] = df_ml['Close'].rolling(window=10).mean()
    df_ml['Dist_SMA_10'] = df_ml['Close'] / df_ml['SMA_10']
    df_ml['Volatility'] = df_ml['Close'].rolling(window=10).std()
    df_ml['RSI'] = calculate_rsi(df_ml['Close'])

    # Drop rows that contain NaNs introduced by rolling operations.
    df_ml = df_ml.dropna()
    return df_ml

print(f"--- Starting Backtesting (Deadline: {DEADLINE}) ---")

# Store trained models per ticker for later reuse/inspection.
rd_models = {}

for t in tickers_bvc:
    print(f"\nProcessing: {t}...")

    # Download adjusted daily bars for this ticker.
    data_frame = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)

    if len(data_frame) > 0:
        # If Yahoo returns MultiIndex columns, flatten them to level 0.
        if isinstance(data_frame.columns, pd.MultiIndex):
            data_frame.columns = data_frame.columns.get_level_values(0)

        df_processed = prepare_data(data_frame)

        # Predictors used by the Random Forest.
        features = ['Return_1d', 'Return_5d', 'Dist_SMA_10', 'Volatility', 'RSI']

        # Time-based split: strict separation of train/test by calendar date.
        mask_train = df_processed.index <= DEADLINE
        mask_test = df_processed.index > DEADLINE

        X_train = df_processed.loc[mask_train, features]
        y_train = df_processed.loc[mask_train, 'Target_Return']

        X_test = df_processed.loc[mask_test, features]
        y_test = df_processed.loc[mask_test, 'Target_Return']


        if len(X_test) == 0:
            # Nothing to evaluate if the ticker has no dates after the cutoff.
            print(f"   [!] Warning: No hay datos posteriores a {DEADLINE} para {t}.")
            continue

        print(f"   -> Training Data (Until 2024): {len(X_train)} days")
        print(f"   -> Test Data (From 2025): {len(X_test)} days")


        # A compact RF configuration to reduce overfitting and keep variance in check.
        model = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=5, random_state=42)
        model.fit(X_train, y_train)

        pred_returns_2025 = model.predict(X_test)

        # Keep a reference to each per-ticker model.
        rd_models.update({t: model})

        # Align real prices on the test period to convert predicted returns into prices.
        real_prices = data_frame['Close'].loc[X_test.index]

        # Price estimate at time t: P_hat_t ≈ P_t * (1 + r_hat_t).
        predicted_prices = real_prices.values * (1 + pred_returns_2025)

        # Evaluate error directly in price units for interpretability.
        mae = mean_absolute_error(real_prices, predicted_prices)

        # Plot actual vs. estimated series for the 2025 test interval.
        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn-v0_8-whitegrid')

        plt.plot(real_prices.index, real_prices.values,
                 label='Realidad (2025)', color='navy', linewidth=2)

        plt.plot(real_prices.index, predicted_prices,
                 label='Modelo (Entrenado solo con datos <2025)',
                 color='#f97316', linestyle='--', marker='o', markersize=3, alpha=0.8)

        plt.title(f"Prueba de Eficacia 2025: {t}\n(Error Promedio: ${mae:.2f} COP)", fontsize=14)
        plt.ylabel("Precio (COP)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    else:
        print(f"Could not download data for {t}")