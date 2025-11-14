from dotenv import load_dotenv
import os
import refinitiv.data as rd

import eikon as ek
import pandas as pd




#load credentials
load_dotenv()
eikon_key = os.getenv("eikon_key")
ek.set_app_key(eikon_key)



def get_rics():
    """ Returns a list of RIC-codes for all assets in SP500
    """
    rics = pd.read_csv("SP_500_RICS.csv")
    rics = rics["RIC"].to_list()
    return rics




# Get RICS for SP500
#df, err = ek.get_data(
#    instruments = "0#.SPX",
#    fields = ["TR.IndexConstituentRIC"]
#)
#print(df.columns)
#
#df = df.rename(columns={
#    "Instrument": "stock",
#    "Constituent RIC": "RIC",
#    })
#df.to_csv("SP_500_RICS.csv", index = False)

def get_esg():
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
    esg_full.to_csv("ESG_scores.csv", index=False)

def get_prices():
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
    prices_full.to_csv("Daily_Prices.csv")

def get_yearly_price_and_cov_matrics():
    prices_full = pd.read_csv("Daily_Prices.csv")

    # Set date index if present
    if "Date" in prices_full.columns:
        prices_full = prices_full.set_index("Date")

    # Convert all price columns to numeric
    prices_full = prices_full.apply(pd.to_numeric, errors="coerce")

    # Calculate returns
    returns = prices_full.pct_change().dropna()

    # Yearly returns
    yearly_returns = (1 + returns).prod() - 1
    yearly_returns.to_csv("Yearly_Returns.csv", header=["return"])

    # Covariance matrix
    cov_matrix = returns.cov()
    cov_matrix.to_csv("Covariance_Matrix.csv")

get_yearly_price_and_cov_matrics()

##separate rics into two sets
#rics = df["RIC"].tolist()
#rics_1 = rics[:200]
#print(rics_1)
#
#
#
#
## 2. Prices
#prices_df = ek.get_timeseries(
#    rics[:50],
#    fields="CLOSE",
#    start_date="2020-01-01",
#    end_date="2024-12-31",
#    interval="daily",
#    normalize=True
#)
#
#print(prices_df)
#
#esg_df, err = ek.get_data(
#    instruments = rics[:50],   # or any slice
#    fields = [
#        "TR.TRESGScore",
#        "TR.TRESGEnvScore",
#        "TR.TRESGSocScore",
#        "TR.TRESGGovScore"
#    ]
#)
#
#print(esg_df)
##print(rics)
##test_df = ek.get_timeseries(rics, 
##                            fields = "CLOSE",
##                            start_date = "2024-12-01",
##                            end_date = "2025-01-09",
##                            interval = "daily")
##print(test_df)
