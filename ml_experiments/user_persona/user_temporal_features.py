import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re

# --- 1. Setup & Sample Data ---
# In a real system, this data would come from your databases or data warehouse.

def create_sample_data():
    """Creates sample DataFrames for users, transactions, and interactions."""
    # User Profile Data
    users_data = {
        'user_id': [101, 102],
        'premium_card_indicator': [1, 0],
        'mortgage_indicator': [0, 1],
        'account_counts': [3, 5],
        'tenure_days': [730, 150],
        'risk_score': [650, 720],
        'delinquent_count': [0, 1],
        'utilization_ratio': [0.4, 0.8]
    }
    users_df = pd.DataFrame(users_data).set_index('user_id')

    # Transaction Data
    transactions_data = {
        'timestamp': pd.to_datetime([
            '2025-10-12', '2025-10-10', '2025-09-20', '2025-08-15', # User 101
            '2025-10-05', '2025-09-15'                             # User 102
        ]),
        'user_id': [101, 101, 101, 101, 102, 102],
        'amount': [50.5, 120.0, 75.2, 250.0, 30.0, 400.0],
        'category': ['groceries', 'electronics', 'groceries', 'travel', 'dining', 'travel']
    }
    transactions_df = pd.DataFrame(transactions_data)

    # Interaction (Clickstream) Data
    interactions_data = {
        'timestamp': pd.to_datetime([
            '2025-10-11', '2025-10-09', '2025-10-09', '2025-09-25', '2025-09-18', # User 101
            '2025-10-04'                                                         # User 102
        ]),
        'user_id': [101, 101, 101, 101, 101, 102],
        'event_type': ['nav_click', 'search', 'nav_click', 'nav_click', 'event', 'nav_click'],
        'detail': ['credit_cards', 'how to save', 'mortgage_calculator', 'investments', 'mortgage_calculator', 'credit_cards']
    }
    interactions_df = pd.DataFrame(interactions_data)

    return users_df, transactions_df, interactions_df

# --- 2. Refactored TemporalFeatureGenerator ---

class TemporalFeatureGenerator:
    """
    Generates time-aware features for a user based on their historical activity.
    """
    def __init__(self, transactions_df, interactions_df):
        """
        Initializes the generator with data sources.

        Args:
            transactions_df (pd.DataFrame): DataFrame with transaction data.
            interactions_df (pd.DataFrame): DataFrame with interaction/clickstream data.
        """
        self.transactions = transactions_df
        self.interactions = interactions_df
        # Ensure timestamps are in datetime format
        self.transactions['timestamp'] = pd.to_datetime(self.transactions['timestamp'])
        self.interactions['timestamp'] = pd.to_datetime(self.interactions['timestamp'])
        
        self.spending_categories = self.transactions['category'].unique()

    def _exponential_decay_weight(self, days_elapsed, half_life):
        """Calculates the weight of an event based on its recency."""
        if half_life <= 0:
            return 1.0 # No decay if half_life is zero or negative
        decay_rate = np.log(2) / half_life
        return np.exp(-decay_rate * days_elapsed)

    def generate_features(self, user_id, current_timestamp):
        """
        Generates a feature dictionary for a single user at a specific time.

        Args:
            user_id (int): The ID of the user.
            current_timestamp (datetime): The timestamp for feature calculation.
        """
        features = {}
        user_transactions = self.transactions[self.transactions['user_id'] == user_id]
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]

        windows = [1, 7, 30, 90]  # days
        
        # --- Multi-scale temporal aggregations ---
        for window in windows:
            start_date = current_timestamp - timedelta(days=window)
            
            # Filter data for the current time window
            tx_in_window = user_transactions[
                (user_transactions['timestamp'] >= start_date) & (user_transactions['timestamp'] < current_timestamp)
            ]
            int_in_window = user_interactions[
                (user_interactions['timestamp'] >= start_date) & (user_interactions['timestamp'] < current_timestamp)
            ]

            # Transaction-based features
            features[f'transaction_count_{window}d'] = len(tx_in_window)
            features[f'avg_transaction_amount_{window}d'] = tx_in_window['amount'].mean() if not tx_in_window.empty else 0
            # Velocity: count per day
            features[f'transaction_velocity_{window}d'] = len(tx_in_window) / window

            # Interaction features
            features[f'nav_clicks_{window}d'] = int_in_window[int_in_window['event_type'] == 'nav_click'].shape[0]
            features[f'search_queries_{window}d'] = int_in_window[int_in_window['event_type'] == 'search'].shape[0]
            features[f'unique_nav_links_{window}d'] = int_in_window[int_in_window['event_type'] == 'nav_click']['detail'].nunique()

            # Category spending with exponential decay
            for category in self.spending_categories:
                category_tx = tx_in_window[tx_in_window['category'] == category]
                if not category_tx.empty:
                    days_elapsed = (current_timestamp - category_tx['timestamp']).dt.total_seconds() / (24 * 3600)
                    weights = self._exponential_decay_weight(days_elapsed, half_life=window/2)
                    weighted_sum = (category_tx['amount'] * weights).sum()
                else:
                    weighted_sum = 0
                features[f'{category}_spending_decay_{window}d'] = weighted_sum
        
        # --- Trend features (comparing time windows) ---
        # Fixed the key name and added safe division
        count_7d = features['transaction_count_7d']
        count_30d = features['transaction_count_30d']
        features['transaction_trend_7d_vs_30d'] = count_7d / max(count_30d, 1)

        # --- Recency features ---
        for event_detail in ['mortgage_calculator', 'credit_cards']:
            last_event_time = user_interactions[user_interactions['detail'] == event_detail]['timestamp'].max()
            if pd.notna(last_event_time):
                days_since = (current_timestamp - last_event_time).days
            else:
                days_since = -1 # Use a special value for "never happened"
            features[f'days_since_last_{event_detail}'] = days_since
            
        return features


# --- 3. Refactored FeatureEncoder ---
# This class now follows scikit-learn's fit/transform pattern.

class FeatureEncoder:
    """
    Encodes static, contextual, and query features into a model-ready format.
    This encoder must be fitted on training data before use.
    """
    def __init__(self):
        # Define columns for different transformations
        self.profile_numeric_features = ['account_counts', 'tenure_days']
        self.risk_features = ['risk_score', 'delinquent_count', 'utilization_ratio']
        self.context_categorical_features = ['device_type', 'channel']
        
        # Initialize transformers
        # We will define the full pipeline in the 'fit' method
        self.pipeline = None

    def _cyclic_encode(self, data, cycle):
        """Encodes cyclic features like hour of day or day of week."""
        sin_feat = np.sin(2 * np.pi * data / cycle)
        cos_feat = np.cos(2 * np.pi * data / cycle)
        # Use np.column_stack to correctly handle both scalars and arrays
        return np.column_stack([sin_feat, cos_feat]) # <-- FIXED LINE
    
    def _mock_bert_encoder(self, query):
        """Mocks a BERT encoder by returning a fixed-size vector."""
        # In a real scenario, you would use a library like transformers
        np.random.seed(hash(query) % (2**32 - 1))
        return np.random.rand(1, 16) # Smaller embedding for example
    
    def fit(self, user_df, context_df):
        """
        Fits the encoders and scalers on the training data.

        Args:
            user_df (pd.DataFrame): DataFrame with user profile and risk features.
            context_df (pd.DataFrame): DataFrame with context features.
        """
        # Define preprocessing steps for different feature types
        profile_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
        risk_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        context_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        # Create a ColumnTransformer to apply different transformations to different columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('profile', profile_transformer, self.profile_numeric_features),
                ('risk', risk_transformer, self.risk_features),
                ('context', context_transformer, self.context_categorical_features)
            ],
            remainder='passthrough' # Keep other columns (like indicators)
        )
        
        self.pipeline = preprocessor.fit(pd.concat([user_df, context_df], axis=1))
        return self

    def transform_user_features(self, user_data, context_data):
        """
        Encodes a user's features using the pre-fitted pipeline.

        Args:
            user_data (pd.DataFrame): A single row DataFrame for one user.
            context_data (dict): Dictionary with contextual info like time and device.
        
        Returns:
            np.ndarray: The encoded feature vector for the user.
        """
        if self.pipeline is None:
            raise RuntimeError("Encoder has not been fitted yet. Call 'fit' first.")
            
        # --- Context features ---
        # Create a DataFrame from the context dict to match the schema
        context_df = pd.DataFrame([context_data])
        
        # Cyclic encoding for time features
        hour_cyclic = self._cyclic_encode(context_data['current_hour'], 24)
        day_of_week_cyclic = self._cyclic_encode(context_data['current_day_of_week'], 7)

        # --- Combine and transform ---
        # Combine user and context data for the pipeline
        full_user_context_df = pd.concat([user_data.reset_index(drop=True), context_df], axis=1)
        
        # Use the fitted pipeline to transform data
        encoded_profile_risk_context = self.pipeline.transform(full_user_context_df)

        # --- Behavioral sequence features (mocked) ---
        # In a real system, this would be an RNN, Transformer, or attention model
        sequence_features = np.random.rand(1, 10) # Mocked fixed-size vector

        # --- Concatenate all feature groups ---
        final_vector = np.concatenate([
            encoded_profile_risk_context,
            hour_cyclic,
            day_of_week_cyclic,
            sequence_features
        ], axis=1)
        
        return final_vector

    def transform_query_features(self, query):
        """Encodes a search query into a feature vector."""
        query_embedding = self._mock_bert_encoder(query)
        
        # Query characteristics
        query_features = np.array([[
            len(query.split()),                     # word count
            1 if re.search(r'\d', query) else 0,    # has number
            1 if 'save' in query or 'loan' in query else 0, # has financial keyword
            # Mock intent classification (0: navigate, 1: transact, 2: explore)
            hash(query) % 3
        ]])
        
        return np.concatenate([query_embedding, query_features], axis=1)

# --- 4. Example Usage ---

if __name__ == '__main__':
    # --- Setup ---
    users_df, transactions_df, interactions_df = create_sample_data()
    CURRENT_TIMESTAMP = datetime(2025, 10, 13, 19, 0, 0) # Current time for feature generation
    
    # --- Part A: Generate Temporal Features for a User ---
    print("--- Generating Temporal Features ---")
    temporal_generator = TemporalFeatureGenerator(transactions_df, interactions_df)
    user_101_temporal_features = temporal_generator.generate_features(user_id=101, current_timestamp=CURRENT_TIMESTAMP)
    
    print(f"Generated features for User 101 at {CURRENT_TIMESTAMP}:")
    for k, v in user_101_temporal_features.items():
        print(f"  {k}: {v:.2f}")

    # --- Part B: Encode User Features ---
    print("\n--- Encoding User & Context Features ---")
    # 1. Prepare data for fitting the encoder (using all users as training data)
    # In a real ML pipeline, this would be your training set.
    # We also need to create a sample context DataFrame for fitting.
    fit_context_df = pd.DataFrame([
        {'device_type': 'mobile', 'channel': 'app'},
        {'device_type': 'desktop', 'channel': 'web'}
    ])
    
    # Instantiate and fit the encoder
    encoder = FeatureEncoder()
    encoder.fit(users_df, fit_context_df)
    print("FeatureEncoder has been fitted.")

    # 2. Get data for the specific user we want to encode
    user_101_profile_data = users_df.loc[[101]]
    
    # 3. Define the current context
    current_context = {
        'device_type': 'mobile',
        'channel': 'app',
        'current_hour': CURRENT_TIMESTAMP.hour,
        'current_day_of_week': CURRENT_TIMESTAMP.weekday()
    }
    
    # 4. Transform the user's features
    user_101_encoded_vector = encoder.transform_user_features(user_101_profile_data, current_context)
    print(f"\nEncoded feature vector for User 101 (shape: {user_101_encoded_vector.shape}):")
    print(user_101_encoded_vector)

    # --- Part C: Encode a Query ---
    print("\n--- Encoding a Query Feature ---")
    search_query = "how to save for a mortgage"
    query_vector = encoder.transform_query_features(search_query)
    print(f"Encoded vector for query '{search_query}' (shape: {query_vector.shape}):")
    print(query_vector)