import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

import rich

class TransactionClassifier:
    """
    A sophisticated classifier to engineer features and assign micro-tags
    to a user's transaction history.
    """

    def __init__(self, user_home_location, user_work_location=None):
        """
        Initializes the classifier with user-specific information.

        Args:
            user_home_location (tuple): A tuple (latitude, longitude) for the user's home.
            user_work_location (tuple, optional): A tuple (latitude, longitude) for the user's work.
        """
        self.user_home_location = user_home_location
        self.user_work_location = user_work_location

        # --- Pre-defined Lists and Dicts for Features and Tags ---

        self.engineered_features = [
            'amount_z_score', 'spend_vs_3d_avg', 'is_round_number', 'hour_of_day',
            'is_weekend', 'distance_from_home', 'time_since_last_transaction',
            'is_online', 'is_foreign_transaction', 'part_of_day', 'is_payday',
            'is_subscription_like', 'merchant_first_time', 'running_daily_spend',
            'distance_from_work', 'is_multi_person_merchant', 'is_investment_merchant',
            'is_savings_merchant', 'is_loan_merchant', 'is_digital_subscription_merchant',
            'is_ecommerce_merchant', 'is_service_signup_merchant'
        ]

        self.micro_tags = {
            'Behavioral': ['Recurring', 'One-Time', 'Discretionary', 'Essential', 'Impulse-Buy', 'Work-Related', 'Bill-Payment'],
            'Value-Based': ['High-Ticket', 'Mid-Range', 'Budget-Friendly', 'Luxury'],
            'Temporal': ['Weekday-Commute', 'Weekend-Leisure', 'Late-Night', 'Holiday-Shopping', 'Lunch-Rush'],
            'Mood-Based': ['Weekend-Splurge', 'Travel-Mode', 'Health-Kick', 'Treat-Yo-Self'],
            'Geographical': ['Near-Home', 'Near-Work', 'City-Explorer'],
            'Social': ['Group-Outing', 'Solo-Venture'],
            'Financial': ['Savings-Goal-Contribution', 'Investment', 'Loan-Repayment'],
            'Online-Behavior': ['Digital-Subscription', 'E-commerce-Shopping', 'Service-Sign-up']
        }


    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates the distance between two geo-coordinates."""
        R = 6371  # Earth radius in kilometers
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        a = sin(dLat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def engineer_features(self, transactions_df):
        """
        Engineers a rich set of features from raw transaction data.

        Args:
            transactions_df (pd.DataFrame): DataFrame with user's raw transactions.
                                            Required columns: ['timestamp', 'amount', 'merchant_name',
                                                               'merchant_category', 'lat', 'lon', 'payment_channel', 'iso_country_code']

        Returns:
            pd.DataFrame: The DataFrame with added engineered features.
        """
        df = transactions_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # --- Feature Engineering ---

        # Amount-based
        user_mean_spend = df['amount'].mean()
        user_std_spend = df['amount'].std()
        df['amount_z_score'] = (df['amount'] - user_mean_spend) / user_std_spend
        df['is_round_number'] = df['amount'].apply(lambda x: x == round(x))

        # Temporal
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['time_since_last_transaction'] = df['timestamp'].diff().dt.total_seconds() / 3600
        df['part_of_day'] = pd.cut(df['hour_of_day'], bins=[-1, 6, 12, 17, 21, 24], labels=['Late-Night', 'Morning', 'Afternoon', 'Evening', 'Late-Night'], ordered=False)
        df['is_payday'] = df['timestamp'].dt.day.isin([1, 15, 30, 31])

        # Location-based
        df['distance_from_home'] = df.apply(lambda row: self._haversine_distance(self.user_home_location[0], self.user_home_location[1], row['lat'], row['lon']), axis=1)
        if self.user_work_location:
            df['distance_from_work'] = df.apply(lambda row: self._haversine_distance(self.user_work_location[0], self.user_work_location[1], row['lat'], row['lon']), axis=1)
        else:
            df['distance_from_work'] = -1
        df['is_online'] = df['payment_channel'] == 'Online'
        df['is_foreign_transaction'] = df['iso_country_code'] != 'USA' # Assuming user's home country is USA

        # Merchant-based
        df['is_multi_person_merchant'] = df['merchant_name'].str.contains('Cinema|Bowling|Group', case=False)
        df['is_investment_merchant'] = df['merchant_name'].str.contains('Invest|Stocks|Crypto', case=False)
        df['is_savings_merchant'] = df['merchant_name'].str.contains('Savings|Acorns|Stash', case=False)
        df['is_loan_merchant'] = df['merchant_name'].str.contains('Loan|Mortgage|Lending', case=False)
        df['is_digital_subscription_merchant'] = df['merchant_name'].str.contains('Netflix|Spotify|Prime', case=False)
        df['is_ecommerce_merchant'] = df['merchant_name'].str.contains('Amazon|Ebay|Asos', case=False)
        df['is_service_signup_merchant'] = df['merchant_name'].str.contains('Trial|Sign-up|Free-Month', case=False)


        # Behavioral
        df['spend_vs_3d_avg'] = df['amount'] / df.rolling('3D', on='timestamp')['amount'].mean()
        df['running_daily_spend'] = df.groupby(df['timestamp'].dt.date)['amount'].cumsum()
        df['merchant_first_time'] = ~df.duplicated(subset=['merchant_name'], keep='first')

        # Subscription-like
        sub_like = df.duplicated(subset=['merchant_name', 'amount'], keep=False)
        df['is_subscription_like'] = sub_like & (df.groupby(['merchant_name', 'amount'])['timestamp'].diff().dt.days.fillna(0).abs().between(28, 32))

        return df


    def assign_micro_tags(self, featured_df):
        """
        Assigns micro-tags to transactions based on engineered features.

        Args:
            featured_df (pd.DataFrame): DataFrame with engineered features.

        Returns:
            pd.DataFrame: The DataFrame with an added 'micro_tags' column.
        """
        df = featured_df.copy()
        tags = []

        for _, row in df.iterrows():
            current_tags = set()

            # --- Rule-based Tagging Logic ---

            # Value-Based Tags
            if row['amount_z_score'] > 2.0: current_tags.add('High-Ticket')
            elif row['amount_z_score'] < -1.0: current_tags.add('Budget-Friendly')
            else: current_tags.add('Mid-Range')
            if row['merchant_category'] in ['Luxury Goods', 'High-End Electronics']: current_tags.add('Luxury')

            # Behavioral Tags
            if row['is_subscription_like']: current_tags.add('Recurring')
            elif row['merchant_first_time']: current_tags.add('One-Time')

            if row['merchant_category'] in ['Groceries', 'Utilities', 'Pharmacy']: current_tags.add('Essential')
            elif row['merchant_category'] in ['Entertainment', 'Hobbies', 'Restaurants']: current_tags.add('Discretionary')

            if row['time_since_last_transaction'] < 0.5 and row['amount_z_score'] < 0: current_tags.add('Impulse-Buy')
            if row['merchant_category'] == 'Office Supplies': current_tags.add('Work-Related')

            # Temporal Tags
            if not row['is_weekend'] and row['hour_of_day'] in [12, 13]: current_tags.add('Lunch-Rush')
            if row['is_weekend'] and row['merchant_category'] == 'Entertainment': current_tags.add('Weekend-Leisure')
            if row['part_of_day'] == 'Late-Night': current_tags.add('Late-Night')

            # Mood/Context-Based Tags
            if row['is_weekend'] and row['amount_z_score'] > 1.5 and 'Discretionary' in current_tags: current_tags.add('Weekend-Splurge')
            if row['distance_from_home'] > 100: current_tags.add('Travel-Mode')
            if row['merchant_category'] in ['Gym', 'Health Foods']: current_tags.add('Health-Kick')

            # Geographical Tags
            if row['distance_from_home'] < 2: current_tags.add('Near-Home')
            if 'distance_from_work' in row and 0 < row['distance_from_work'] < 2: current_tags.add('Near-Work')
            if row['distance_from_home'] > 10 and ('distance_from_work' in row and row['distance_from_work'] > 10): current_tags.add('City-Explorer')

            # Social Tags
            if row['is_multi_person_merchant']: current_tags.add('Group-Outing')
            else: current_tags.add('Solo-Venture')

            # Financial Tags
            if row['is_savings_merchant']: current_tags.add('Savings-Goal-Contribution')
            if row['is_investment_merchant']: current_tags.add('Investment')
            if row['is_loan_merchant']: current_tags.add('Loan-Repayment')

            # Online Behavior Tags
            if row['is_digital_subscription_merchant']: current_tags.add('Digital-Subscription')
            if row['is_ecommerce_merchant']: current_tags.add('E-commerce-Shopping')
            if row['is_service_signup_merchant']: current_tags.add('Service-Sign-up')


            tags.append(list(current_tags))

        df['micro_tags'] = tags
        return df

# --- Sample Usage ---
if __name__ == '__main__':
    # Sample user's last 90 days of transactions
    
    transactions_df = pd.read_csv("data/synthetic_transaction_data.csv")

    # User's home location (e.g., Downtown LA)
    home_location = (37.7749, -122.4194)

    # Initialize and run the classifier
    classifier = TransactionClassifier(user_home_location=home_location)
    featured_transactions = classifier.engineer_features(transactions_df)
    tagged_transactions = classifier.assign_micro_tags(featured_transactions)

    

    
    rich.print(f"\n[bold green]✅ Success! Generated Tags [cyan]{len(tagged_transactions)}[/cyan].[/bold green]")
    rich.print(f"File saved to: '[bold yellow]data/tagged_synthetic_transaction_data.csv[/bold yellow]'")
    tagged_transactions.to_csv("data/tagged_synthetic_transaction_data.csv")
    
    print("\n--- Data Preview ---")
    rich.print(tagged_transactions.head())

    rich.print("--- Tagged Transactions ---")
    rich.print(tagged_transactions.head(1).to_dict( orient='records'))