import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import yfinance as yf

tickers_bvc = ["ECOPETROL.CL", "ISA.CL", "GRUPOARGOS.CL", "GEB.CL"]
dfs_list = []

print("Downloading and formatting data...")

for t in tickers_bvc:

    df = yf.download(t, period="max", auto_adjust=True, progress=False)

    if len(df) > 0:

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        df['Ticker'] = t

        cols_to_keep = ['Date', 'Ticker', 'Close', 'Volume']
        df = df[cols_to_keep]

        dfs_list.append(df)


df_final = pd.concat(dfs_list)


df_final = df_final.dropna()

print(f"Total clean records: {len(df_final)}")
print(df_final.head())

unique_tickers = df_final['Ticker'].unique()

for t in unique_tickers:
    company_data = df_final[df_final['Ticker'] == t].copy()
    company_data = company_data.sort_values('Date')


    company_data['Date'] = pd.to_datetime(company_data['Date'])

    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})


    ax1.plot(company_data['Date'], company_data['Close'], label='Precio Cierre', color='#2563eb', linewidth=1.2)
    ax1.set_title(f"Comportamiento Histórico: {t}", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("Precio (CLP)", fontsize=12)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, which='major', linestyle='-', alpha=0.6)


    ax2.fill_between(company_data['Date'], company_data['Volume'], 0,
                     color='#f97316', alpha=0.8, label='Volumen')

    ax2.set_ylabel("Traded Volume", fontsize=12)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(True, which='major', linestyle='--', alpha=0.5)

    ax2.xaxis.set_major_locator(mdates.YearLocator(2)) 
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    
    plt.xticks(rotation=45, ha='right')

 
    plt.tight_layout()


    print(f"Visualizing: {t}")
    plt.show()