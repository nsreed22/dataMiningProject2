import pandas as pd
from ucimlrepo import fetch_ucirepo

# --- 1. DATA LOADING (Replicating your exact loading step) ---
try:
    print("Attempting to fetch data from UCI repository (id=373)...")
    drug_consumption_quantified = fetch_ucirepo(id=373)
    
    # Data (features X and targets y)
    X = drug_consumption_quantified.data.features 
    y = drug_consumption_quantified.data.targets 

    # Combine features and targets into one DataFrame
    df = pd.concat([X, y], axis=1)
    print("Data loaded successfully.")

    # --- 2. DEBUG OUTPUT ---
    print("\n" + "="*50)
    print("EXACT COLUMN NAMES IN YOUR DATAFRAME:")
    print("="*50)
    
    # Print the exact list of column names as a list
    print(df.columns.tolist())
    
    print("="*50)
    print("Please copy and paste the list above in our chat.")

except Exception as e:
    print(f"\nERROR: Failed to load data. The error encountered was: {e}")
    print("Please ensure you have the 'ucimlrepo' library installed ('pip install ucimlrepo').")