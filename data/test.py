import yfinance as yf

# Use the '^GSPC' ticker for the S&P 500
ticker_symbol = '^GSPC'

# Use period='3y' for the last 3 years of daily data
# auto_adjust=True automatically adjusts prices for splits and dividends
spx_data = yf.download(ticker_symbol, period='3y', auto_adjust=True)

# Select only the 'Close' price column
spx_close_prices = spx_data[['Close']]

# Save the data to a CSV file
spx_close_prices.to_csv('spx_closing_prices_3_years.csv')

print("Saved last 3 years of S&P 500 closing prices to spx_closing_prices_3_years.csv")
print(spx_close_prices.head())