import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


tickers_bvc = ['ECOPETROL.CL', 'ISA.CL', 'GRUPOARGOS.CL', 'GEB.CL']
DEADLINE = "2024-12-31"  


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(df):
    df_ml = df.copy()

    df_ml['Target_Return'] = df_ml['Close'].pct_change().shift(-1)

  
    df_ml['Return_1d'] = df_ml['Close'].pct_change()
    df_ml['Return_5d'] = df_ml['Close'].pct_change(5)
    df_ml['SMA_10'] = df_ml['Close'].rolling(window=10).mean()
    df_ml['Dist_SMA_10'] = df_ml['Close'] / df_ml['SMA_10']
    df_ml['Volatility'] = df_ml['Close'].rolling(window=10).std()
    df_ml['RSI'] = calculate_rsi(df_ml['Close'])

    df_ml = df_ml.dropna()
    return df_ml

print(f"--- Starting Backtesting (Deadline: {DEADLINE}) ---")

rd_models = {}

for t in tickers_bvc:
    print(f"\nProcessing: {t}...")

  
    data_frame = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)

    if len(data_frame) > 0:
        if isinstance(data_frame.columns, pd.MultiIndex):
            data_frame.columns = data_frame.columns.get_level_values(0)

        df_processed = prepare_data(data_frame)

        features = ['Return_1d', 'Return_5d', 'Dist_SMA_10', 'Volatility', 'RSI']

        mask_train = df_processed.index <= DEADLINE
        mask_test = df_processed.index > DEADLINE

        X_train = df_processed.loc[mask_train, features]
        y_train = df_processed.loc[mask_train, 'Target_Return']

        X_test = df_processed.loc[mask_test, features]
        y_test = df_processed.loc[mask_test, 'Target_Return']


        if len(X_test) == 0:
            print(f"   [!] Warning: No hay datos posteriores a {DEADLINE} para {t}.")
            continue

        print(f"   -> Training Data (Until 2024): {len(X_train)} days")
        print(f"   -> Test Data (From 2025): {len(X_test)} days")


        model = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=5, random_state=42)
        model.fit(X_train, y_train)

        pred_returns_2025 = model.predict(X_test)

        rd_models.update({t: model})

  
        real_prices = data_frame['Close'].loc[X_test.index]

 
        predicted_prices = real_prices.values * (1 + pred_returns_2025)

  
        mae = mean_absolute_error(real_prices, predicted_prices)

    
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