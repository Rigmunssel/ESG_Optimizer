import yfinance as yf
data = yf.download("AAPL", start="2024-01-01", end="2024-12-31")
print(data.tail())