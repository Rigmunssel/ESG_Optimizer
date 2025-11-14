from dotenv import load_dotenv
import os
import refinitiv.data as rd

import eikon as ek
import pandas as pd


# Root data directory
DATA_DIR = "data"


# Load credentials
load_dotenv()
eikon_key = os.getenv("eikon_key")
ek.set_app_key(eikon_key)


def get_rics():
    """Returns a list of RIC-codes for all assets in SP500."""
    path = os.path.join(DATA_DIR, "SP_500_RICS.csv")
    rics = pd.read_csv(path)
    return rics["RIC"].to_list()


def get_esg():
    """Wrtites ESG-scores for each stock"""
    all_esg = []
    rics = get_rics()
    batch_size = 100

    for i in range(0, len(rics), batch_size):
        batch = rics[i:i+batch_size]

        esg_df, err = ek.get_data(
            instruments=batch,
            fields=[
                "TR.TRESGScore",
                "TR.TRESGEnvScore",
                "TR.TRESGSocScore",
                "TR.TRESGGovScore"
            ]
        )

        all_esg.append(esg_df)

    esg_full = pd.concat(all_esg, ignore_index=True)

    out_path = os.path.join(DATA_DIR, "ESG_scores.csv")
    esg_full.to_csv(out_path, index=False)


def get_prices():
    """"Writes daily closing prices for each stock"""
    rics = get_rics()
    prices_list = []
    batch_size = 20

    for i in range(0, len(rics), batch_size):
        batch = rics[i:i+batch_size]

        p = ek.get_timeseries(
            batch,
            fields="CLOSE",
            start_date="2024-11-01",
            end_date="2025-11-01",
            interval="daily",
            normalize=False
        )

        prices_list.append(p)

    prices_full = pd.concat(prices_list, axis=1)

    out_path = os.path.join(DATA_DIR, "Daily_Prices.csv")
    prices_full.to_csv(out_path)


def get_yearly_price_and_cov_matrics():
    """Writes yearly returns and covariance matrix based on daily prices."""
    price_path = os.path.join(DATA_DIR, "Daily_Prices.csv")
    prices_full = pd.read_csv(price_path)

    print(prices_full.info())
    print(prices_full.head())
    print(prices_full.tail())
    print(prices_full.isna().sum().sort_values(ascending=False).head(10))

    # Set date index if present
    if "Date" in prices_full.columns:
        prices_full = prices_full.set_index("Date")

    # Convert everything to numeric
    prices_full = prices_full.apply(pd.to_numeric, errors="coerce")

    # Calculate returns
    returns = prices_full.pct_change().dropna()

    # Yearly returns
    yearly_returns = (1 + returns).prod() - 1
    yearly_path = os.path.join(DATA_DIR, "Yearly_Returns.csv")
    yearly_returns.to_csv(yearly_path, header=["return"])

    # Covariance matrix
    cov_matrix = returns.cov()
    cov_path = os.path.join(DATA_DIR, "Covariance_Matrix.csv")
    cov_matrix.to_csv(cov_path)


if __name__ == "__main__":
    get_yearly_price_and_cov_matrics()