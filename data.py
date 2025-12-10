"""Download, clean and visualize price and volume for selected BVC tickers.

This script pulls adjusted daily data via yfinance, keeps only the relevant
columns and produces two vertically stacked plots per ticker:
1) Close price trend
2) Traded volume (area chart)
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import yfinance as yf

# Yahoo Finance symbols for Colombian listings used in this project.
# Adjust to analyze different tickers.
tickers_bvc = ["ECOPETROL.CL", "ISA.CL", "GRUPOARGOS.CL", "GEB.CL"]

# Accumulates per-ticker DataFrames to be concatenated later.
dfs_list = []

print("Downloading and formatting data...")

for t in tickers_bvc:

    # Fetch full-history daily data; auto_adjust includes corporate actions.
    df = yf.download(t, period="max", auto_adjust=True, progress=False)

    if len(df) > 0:

        # Some yfinance calls return a MultiIndex (e.g., multiple symbols);
        # ensure we have a flat column index.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Move index to a regular column to keep 'Date' explicit for plotting.
        df = df.reset_index()

        # Track the originating ticker for later grouping/plotting.
        df['Ticker'] = t

        # Keep only the minimal set required for the plots.
        cols_to_keep = ['Date', 'Ticker', 'Close', 'Volume']
        df = df[cols_to_keep]

        dfs_list.append(df)


# Combine all tickers into a single tidy DataFrame.
df_final = pd.concat(dfs_list)


# Drop rows with missing values (can appear after rolling ops or data gaps).
df_final = df_final.dropna()

print(f"Total clean records: {len(df_final)}")
print(df_final.head())

# Distinct tickers present in the final dataset.
unique_tickers = df_final['Ticker'].unique()

for t in unique_tickers:
    # Filter and sort data per company to ensure chronological plotting.
    company_data = df_final[df_final['Ticker'] == t].copy()
    company_data = company_data.sort_values('Date')


    # Ensure 'Date' is proper datetime for Matplotlib formatters and locators.
    company_data['Date'] = pd.to_datetime(company_data['Date'])

    # Create two vertically stacked axes: price (top) and volume (bottom).
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})


    ax1.plot(company_data['Date'], company_data['Close'], label='Precio Cierre', color='#2563eb', linewidth=1.2)
    ax1.set_title(f"Comportamiento Histórico: {t}", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("Precio (COP)", fontsize=12)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, which='major', linestyle='-', alpha=0.6)


    ax2.fill_between(company_data['Date'], company_data['Volume'], 0,
                     color='#f97316', alpha=0.8, label='Volumen')

    ax2.set_ylabel("Traded Volume", fontsize=12)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(True, which='major', linestyle='--', alpha=0.5)

    # Major ticks every 2 years, labeled with the year; minor ticks every 6 months.
    ax2.xaxis.set_major_locator(mdates.YearLocator(2)) 
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    # Improve readability of dense date labels.
    plt.xticks(rotation=45, ha='right')

 
    plt.tight_layout()


    print(f"Visualizing: {t}")
    plt.show()