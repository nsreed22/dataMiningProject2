import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# --- 1. DATA LOADING AND CLEANING ---

# Fetch the dataset
drug_consumption_quantified = fetch_ucirepo(id=373)
df = pd.concat([drug_consumption_quantified.data.features, drug_consumption_quantified.data.targets], axis=1)

# Renaming dictionary for clarity (must match names used in loop below)
feature_renames = {
    'id': 'ID', 'age': 'Age_Score', 'gender': 'Gender_Score',
    'education': 'Education_Score', 'country': 'Country_Score',
    'ethnicity': 'Ethnicity_Score', 'nscore': 'Neuroticism',
    'escore': 'Extraversion', 'oscore': 'Openness',
    'ascore': 'Agreeableness', 'cscore': 'Conscientiousness',
    'impulsive': 'Impulsiveness', 'ss': 'Sensation_Seeking'
}
df = df.rename(columns=feature_renames)

# Define the English meanings for the 7 ordinal classes (CL0 to CL6)
class_map = {
    'CL0': 'Never Used', 'CL1': 'Used over a Decade Ago', 'CL2': 'Used in Last Decade',
    'CL3': 'Used in Last Year', 'CL4': 'Used in Last Month', 'CL5': 'Used in Last Week',
    'CL6': 'Used in Last Day'
}
class_order = list(class_map.values())

# Apply the mapping and set category order for targets
target_columns = drug_consumption_quantified.data.targets.columns
for col in target_columns:
    df[col] = df[col].replace(class_map).astype('category')
    df[col] = df[col].cat.set_categories(class_order, ordered=True)
# List of all column names (Features + Targets)
all_columns = list(df.columns)

# Separate columns by type for plotting
continuous_cols = list(feature_renames.values()) # The 12 Predictors
categorical_cols = list(target_columns)          # The 19 Targets

# --- 2. PLOTTING LOOP ---

print(f"Starting generation of {len(all_columns)} visualizations...")

for col in all_columns:
    plt.figure(figsize=(10, 6))
    
    # ------------------------------------------------
    # Plotting Continuous Variables (Predictors)
    # ------------------------------------------------
    if col in continuous_cols:
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col} Score (Continuous)')
        plt.xlabel(col)
        plt.ylabel('Count')
        filename = f'dist_{col}_hist.png'
    
    # ------------------------------------------------
    # Plotting Categorical Variables (Targets)
    # ------------------------------------------------
    elif col in categorical_cols:
        # Calculate counts and re-index to ensure correct ordinal order
        counts = df[col].value_counts().reindex(class_order, fill_value=0)
        counts_df = counts.reset_index()
        counts_df.columns = ['Consumption Level', 'Count']

        # Use the updated seaborn syntax to avoid FutureWarnings
        sns.barplot(x='Count', y='Consumption Level', data=counts_df, hue='Consumption Level', palette='viridis', legend=False)
        
        plt.title(f'Consumption Distribution of {col} (Ordinal)')
        plt.xlabel('Count (Number of Participants)')
        plt.ylabel('Consumption Level')
        filename = f'dist_{col}_barchart.png'

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Generated {filename}")

print("\nAll 31 visualizations have been generated and saved to your directory.")