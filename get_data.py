from dotenv import load_dotenv
import os
import refinitiv.data as rd

import eikon as ek
import pandas as pd
import time


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
    batch_size = 3

    for i in range(0, len(rics), batch_size):
        index_end = min(len(rics)-1,i+batch_size)
        batch = rics[i:index_end]
        p = ek.get_timeseries(
            batch,
            fields="CLOSE",
            start_date="2015-11-01",
            end_date="2025-11-01",
            interval="weekly",
            normalize=False
        )

        print(p)
        prices_list.append(p)


    prices_full = pd.concat(prices_list, axis=1)

    out_path = os.path.join(DATA_DIR, "Daily_Prices.csv")
    prices_full.to_csv(out_path)


def get_yearly_price_and_cov_matrics():
    """Writes yearly returns and covariance matrix based on weekly prices."""
    price_path = os.path.join(DATA_DIR, "weekly_prices.csv")
    prices_full = pd.read_csv(price_path)
    
    
    # Drop the last column which contains aggregate close for sp500 "close"
    if "CLOSE" in prices_full.columns:
        prices_full = prices_full.drop(columns=["CLOSE"])
        print("Dropped 'CLOSE' column")
    
    # Set date index if present
    if "Date" in prices_full.columns:
        prices_full = prices_full.set_index("Date")
    
    # Convert everything to numeric
    prices_full = prices_full.apply(pd.to_numeric, errors="coerce")
    
    # Drop stocks with any missing values
    prices_clean = prices_full.dropna(axis=1)
    print(f"Original stocks: {len(prices_full.columns)}, After dropping missing: {len(prices_clean.columns)}")
    
    # Calculate weekly returns
    returns = prices_clean.pct_change().dropna()
    
    # Calculate ANNUALIZED average return (mean * 52 weeks)
    avg_yearly_returns = returns.mean() * 52
    
    # Calculate ANNUALIZED covariance matrix (cov * 52 weeks)
    cov_matrix = returns.cov() * 52
    
    # Save results
    yearly_path = os.path.join(DATA_DIR, "Yearly_Returns.csv")
    avg_yearly_returns.to_csv(yearly_path, header=["return"])
    
    cov_path = os.path.join(DATA_DIR, "Covariance_Matrix.csv")
    cov_matrix.to_csv(cov_path)
    
    return avg_yearly_returns, cov_matrix, prices_clean.columns.tolist()



def get_historical_esg_aggressive():
    """Most aggressive version - maximize batch size"""
    
    rics = get_rics()
    print(f"Fetching ESG data for {len(rics)} RICs...")
    
    # Define the 15-year range
    years = list(range(2010, 2026))  # 2010-2025
    
    all_esg_data = []
    
    # If we have less than 200 RICs, process all at once per year
    if len(rics) <= 200:
        batch_size = len(rics)  # All RICs in one go
    else:
        batch_size = 200  # Max safe batch size
    
    print(f"Using batch size: {batch_size}")
    
    for batch_start in range(0, len(rics), batch_size):
        batch_rics = rics[batch_start:batch_start + batch_size]
        print(f"Processing RICs {batch_start} to {batch_start + len(batch_rics) - 1}...")
        
        for year in years:
            try:
                esg_df, err = ek.get_data(
                    instruments=batch_rics,
                    fields=["TR.TRESGScore", "TR.ESGDate"],
                    parameters={"SDate": f"{year}-01-01", "EDate": f"{year}-12-31"}
                )
                
                if esg_df is not None and not esg_df.empty:
                    esg_df['Year'] = year
                    all_esg_data.append(esg_df)
                    print(f"  Year {year}: ✓ ({len(esg_df)} records)")
                else:
                    print(f"  Year {year}: ✗ No data")
                    
            except Exception as e:
                print(f"  Year {year}: Error - {str(e)[:100]}...")
                continue
        
        time.sleep(1)  # Slightly longer delay between large batches
    
    if all_esg_data:
        esg_full = pd.concat(all_esg_data, ignore_index=True)
        
        # Create data folder if it doesn't exist
        data_folder = "data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            print(f"Created folder: {data_folder}")
        
        # Save to data folder
        file_path = os.path.join(data_folder, "ESG_yearly_2010_2025.csv")
        esg_full.to_csv(file_path, index=False)
        print(f"\n✅ Saved {len(esg_full)} ESG records to {file_path}")
        
        # Quick stats
        ric_count = esg_full['Instrument'].nunique()
        year_count = esg_full['Year'].nunique()
        total_possible = ric_count * year_count
        coverage_pct = (len(esg_full) / total_possible) * 100
        
        print(f"Coverage: {len(esg_full)}/{total_possible} possible records ({coverage_pct:.1f}%)")
        return esg_full
    else:
        print("❌ No data retrieved")
        return None



def get_price_info():
    price_path = os.path.join(DATA_DIR, "weekly_prices.csv")
    df = pd.read_csv(price_path)

    # Find row indices where each year starts
    idx_2015 = df.index[df["Date"] >= "2015-01-01"][0]
    idx_2018 = df.index[df["Date"] >= "2018-01-01"][0]

    # Remove Date column to only evaluate stock columns
    stocks = df.columns.drop("Date")

    # ============ FULL SAMPLE =============
    missing_full = df[stocks].isna().any()
    full_complete = (~missing_full).sum()

    # ============ 2015+ SAMPLE =============
    missing_2015 = df.loc[idx_2015:, stocks].isna().any()
    complete_2015 = (~missing_2015).sum()

    # ============ 2018+ SAMPLE =============
    missing_2018 = df.loc[idx_2018:, stocks].isna().any()
    complete_2018 = (~missing_2018).sum()

    print("Total stocks:", len(stocks))
    print("Complete FULL sample:", full_complete)
    print("Complete 2015+ sample:", complete_2015)
    print("Complete 2018+ sample:", complete_2018)

    print("\nStocks missing FULL sample data:")
    print(list(missing_full[missing_full].index))

    print("\nStocks missing 2015+ sample data:")
    print(list(missing_2015[missing_2015].index))

    print("\nStocks missing 2018+ sample data:")
    print(list(missing_2018[missing_2018].index))

if __name__ == "__main__":
    get_yearly_price_and_cov_matrics()