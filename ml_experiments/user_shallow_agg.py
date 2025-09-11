import pandas as pd
import numpy as np
import rich
import os
from scipy.stats import entropy

def create_aggregate_features(user_transactions: pd.DataFrame, current_date: pd.Timestamp = None) -> dict:
    """
    Generates a sophisticated dictionary of aggregate features summarizing a user's transaction history.
    """
    if user_transactions.empty or len(user_transactions) < 2:
        return {} # Not enough data to generate meaningful features

    df = user_transactions.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

    if current_date is None:
        current_date = df['timestamp'].max() + pd.Timedelta(days=1)

    # --- 0. Overall User Metrics ---
    total_user_spend = df['amount'].sum()
    if total_user_spend == 0:
        return {}

    features = {
        'user_total_spend': total_user_spend,
        'user_total_transactions': len(df),
        'user_avg_transaction_value': df['amount'].mean(),
        'user_std_dev_transaction_value': df['amount'].std(),
        'user_recency_days': (current_date - df['timestamp'].max()).days,
        'user_avg_days_between_transactions': df['timestamp'].diff().dt.total_seconds().mean() / (3600 * 24)
    }

    # --- 1. Spend Distribution (per Category) ---
    category_summary = df.groupby('merchant_category')['amount'].agg(['sum', 'mean', 'count']).reset_index()
    category_summary.rename(columns={
        'sum': 'total_spend',
        'mean': 'avg_transaction_value',
        'count': 'transaction_count'
    }, inplace=True)
    category_summary['percentage_spend'] = (category_summary['total_spend'] / total_user_spend) * 100

    # --- 2. Frequency and Diversity ---
    diversity_df = df.groupby('merchant_category')['merchant_name'].nunique().reset_index()
    diversity_df.rename(columns={'merchant_name': 'unique_merchant_count'}, inplace=True)
    category_summary = pd.merge(category_summary, diversity_df, on='merchant_category', how='left')

    spend_distribution = category_summary['total_spend'] / total_user_spend
    features['user_category_spending_entropy'] = entropy(spend_distribution, base=2)
    
    merchant_dist = df['merchant_name'].value_counts(normalize=True)
    features['user_merchant_spending_entropy'] = entropy(merchant_dist, base=2)

    # --- 3. Temporal Patterns ---
    temporal_spend = df.groupby(['merchant_category', 'is_weekend'])['amount'].mean().unstack(fill_value=0)
    temporal_spend.rename(columns={True: 'avg_spend_weekend', False: 'avg_spend_weekday'}, inplace=True)
    category_summary = pd.merge(category_summary, temporal_spend, on='merchant_category', how='left')
    
    time_of_day_spend = df.groupby(['merchant_category', 'part_of_day'])['amount'].sum().unstack(fill_value=0)
    time_of_day_spend = time_of_day_spend.add_prefix('total_spend_')
    category_summary = pd.merge(category_summary, time_of_day_spend, on='merchant_category', how='left')
    
    features['user_dominant_part_of_day'] = df['part_of_day'].mode()[0]
    features['user_weekend_spend_percentage'] = (df[df['is_weekend']]['amount'].sum() / total_user_spend) * 100

    # --- 4. Categorical Recency, Frequency, and Monetary (RFM) ---
    rfm = df.groupby('merchant_category').agg(
        recency=('timestamp', lambda date: (current_date - date.max()).days),
        frequency=('timestamp', 'count'),
        monetary=('amount', 'sum')
    ).reset_index()
    category_summary = pd.merge(category_summary, rfm.add_prefix('rfm_'), left_on='merchant_category', right_on='rfm_merchant_category', how='left')

    # --- 5. Affinity Score (New Sophisticated Feature) ---
    # Normalize features for score calculation (higher is better for all)
    ss = category_summary # shorthand
    # Recency: lower is better, so we invert it. Add small epsilon to avoid div by zero.
    max_rec = ss['rfm_recency'].max()
    min_rec = ss['rfm_recency'].min()
    ss['recency_norm'] = (max_rec - ss['rfm_recency']) / (max_rec - min_rec + 1e-6) if max_rec > min_rec else 0.5

    ss['freq_norm'] = ss['rfm_frequency'] / (ss['rfm_frequency'].max() + 1e-6)
    ss['spend_norm'] = ss['percentage_spend'] / (ss['percentage_spend'].max() + 1e-6)
    
    # Calculate weighted affinity score
    weights = {'spend': 0.5, 'freq': 0.3, 'recency': 0.2}
    ss['affinity_score'] = (weights['spend'] * ss['spend_norm'] +
                              weights['freq'] * ss['freq_norm'] +
                              weights['recency'] * ss['recency_norm'])
    
    # --- 6. Other Shallow Features ---
    if not category_summary.empty:
        dominant_spend_cat = category_summary.loc[category_summary['total_spend'].idxmax()]
        features['user_dominant_category_by_spend'] = dominant_spend_cat['merchant_category']
        
        dominant_freq_cat = category_summary.loc[category_summary['transaction_count'].idxmax()]
        features['user_dominant_category_by_freq'] = dominant_freq_cat['merchant_category']

    # --- 7. Final Feature Dictionary Assembly ---
    # Flatten all category-specific features into the main dictionary
    cols_to_drop = ['rfm_merchant_category', 'recency_norm', 'freq_norm', 'spend_norm']
    category_summary.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    for _, row in category_summary.iterrows():
        # Retrieve the original category name before it was modified for RFM merge
        category = df[df['merchant_category'] == row.name] if row.name in df['merchant_category'].values else row['merchant_category']
        
        # This is a fallback to get the category name if the index doesn't align perfectly after merges.
        category_name = row['merchant_category'] if 'merchant_category' in row else "unknown_category"
        
        category_prefix = category_name.replace(' ', '_').lower()
        for col_name, value in row.items():
            if col_name != 'merchant_category':
                features[f"cat_{category_prefix}_{col_name}"] = value
    
    return features


# --- Sample Usage ---
if __name__ == '__main__':
    data_path = "data/tagged_synthetic_transaction_data.csv"

    if not os.path.exists(data_path):
        rich.print(f"[bold red]Error:[/bold red] Data file not found at '{data_path}'.")
    else:
        data = pd.read_csv(data_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['is_weekend'] = data['timestamp'].dt.dayofweek.isin([5, 6])
        bins = [0, 6, 12, 18, 24]
        labels = ['Night', 'Morning', 'Afternoon', 'Evening']
        data['part_of_day'] = pd.cut(data['timestamp'].dt.hour, bins=bins, labels=labels, right=False, ordered=True)
        
        rich.print(f"[green]Successfully loaded and prepared {len(data)} transactions.[/green]")
        
        analysis_date = data['timestamp'].max() + pd.Timedelta(days=1)
        
        rich.print(f"Calculating aggregate features for [bold blue]{data['user_id'].nunique()}[/bold blue] users...")
        
        user_features = data.groupby('user_id').apply(
            lambda user_df: create_aggregate_features(user_df, current_date=analysis_date)
        )
        
        shallow_user_features_data = pd.DataFrame(user_features.tolist(), index=user_features.index).fillna(0)
        
        rich.print("\n[bold green]Successfully generated sophisticated aggregate features![/bold green]")
        rich.print("--- Aggregate Features DataFrame Preview ---")
        # Display a subset of key columns for readability
        display_cols = [
            'user_total_spend', 'user_category_spending_entropy', 'user_merchant_spending_entropy',
            'user_weekend_spend_percentage', 'user_dominant_category_by_spend',
            'cat_groceries_affinity_score', 'cat_restaurants_affinity_score'
        ]
        # Ensure display columns exist in the dataframe before trying to display them
        display_cols = [col for col in display_cols if col in shallow_user_features_data.columns]
        rich.print(shallow_user_features_data[display_cols].head())
        rich.print(shallow_user_features_data.sample(1).to_dict( orient='records'))
        shallow_user_features_data.to_csv("data/shallow_aggregated_user_features.csv")