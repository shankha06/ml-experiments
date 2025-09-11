import os
import random
from datetime import datetime, timedelta, time
from typing import Any, Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
import rich

try:
    from faker import Faker
    fake = Faker('en_US') # Use a specific locale for consistency
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    # This initial print can remain standard as rich might not be imported yet
    print(
        "Faker library not found. Using generic data. "
        "To use Faker for richer data, run: pip install Faker"
    )

# --- Main Configuration ---
SF_LOCATION_CENTER: Dict[str, float] = {"lat": 37.7749, "lon": -122.4194}
SIMULATION_START_DATE: datetime = datetime.now() - timedelta(days=90)
SIMULATION_END_DATE: datetime = datetime.now()
NUM_USERS: int = 2500
AVG_TRANSACTIONS_PER_USER_PER_DAY: float = 5
ZIPF_EXPONENT: float = 1.1

# --- Merchant & Subscription Configuration ---
MERCHANT_DATA: Dict[str, List[str]] = {
    # SF-specific and popular merchants
    "Cafes": ["Blue Bottle Coffee", "Philz Coffee", "Ritual Coffee Roasters", "Starbucks"],
    "Restaurants": ["The Slanted Door", "La Taqueria", "Zuni Cafe", "State Bird Provisions", "Chipotle", "Sweetgreen"],
    "Groceries": ["Trader Joe's", "Whole Foods Market", "Safeway", "Gus's Community Market", "Costco"],
    "Transportation": ["Clipper Card Top-up", "Uber", "Lyft", "Bay Wheels", "BART"],
    "Shopping": ["Westfield SF Centre", "Union Square Shops", "Hayes Valley Boutiques", "Target"],
    "Luxury Goods": ["Gump's", "Shreve & Co.", "Cuyana", "Neiman Marcus"],
    "Gym": ["Equinox", "24 Hour Fitness", "Barry's Bootcamp", "Planet Fitness"],
    "Tech": ["Apple Store", "Best Buy", "Central Computers"],
    "Drugstore": ["Walgreens", "CVS"],
    "Streaming": ["Netflix", "Hulu", "Max"],
    "Work-Tools": ["Cloud-Storage", "Gym-Membership", "Subscription-Plan"],
}
# Define subscriptions that users can have
SUBSCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "Netflix": {"category": "Streaming", "amount_range": (15.49, 22.99)},
    "Spotify": {"category": "Streaming", "amount_range": (10.99, 16.99)},
    "Gym-Membership": {"category": "Gym", "amount_range": (40.0, 250.0)},
    "Cloud-Storage": {"category": "Work-Tools", "amount_range": (9.99, 19.99)},
}

# --- Scenario Configuration ---
# (Unchanged from previous version)
TRANSACTION_SCENARIOS: List[Dict[str, Any]] = [
    { "name": "essential_groceries_drugstore", "day_type": "any", "hour_range": (9, 20), "categories": ["Groceries", "Drugstore"], "pareto_shape": 3.0, "min_amount": 20, "max_amount": 350, "tags": {"Essential", "Recurring", "Mid-Range"}},
    { "name": "work_lunch", "day_type": "weekday", "hour_range": (12, 14), "categories": ["Restaurants", "Cafes"], "pareto_shape": 2.5, "min_amount": 12, "max_amount": 60, "tags": {"Work-Related", "Discretionary", "Mid-Range"}},
    { "name": "weekday_commute_coffee", "day_type": "weekday", "hour_range": (7, 10), "categories": ["Cafes", "Transportation"], "pareto_shape": 3.5, "min_amount": 4, "max_amount": 25, "tags": {"Weekday-Commute", "Essential", "Recurring", "Budget-Friendly"}},
    { "name": "weekend_leisure", "day_type": "weekend", "hour_range": (12, 22), "categories": ["Shopping", "Restaurants", "Tech"], "pareto_shape": 2.2, "min_amount": 40, "max_amount": 600, "tags": {"Weekend-Leisure", "Discretionary", "One-Time"}},
    { "name": "late_night_impulse", "day_type": "any", "hour_range": (22, 3), "categories": ["Restaurants", "Transportation"], "pareto_shape": 2.8, "min_amount": 8, "max_amount": 75, "tags": {"Late-Night", "Impulse-Buy", "One-Time", "Budget-Friendly"}},
    { "name": "health_kick", "day_type": "any", "hour_range": (6, 20), "categories": ["Gym", "Groceries"], "pareto_shape": 2.0, "min_amount": 25, "max_amount": 200, "tags": {"Health-Kick", "Discretionary", "Recurring", "Mid-Range"}},
    { "name": "weekend_splurge", "day_type": "weekend", "hour_range": (18, 23), "categories": ["Luxury Goods", "Restaurants", "Tech"], "pareto_shape": 1.8, "min_amount": 300, "max_amount": 5000, "tags": {"Weekend-Splurge", "Luxury", "High-Ticket", "One-Time"}},
]

# --- User Persona Configuration ---
USER_PERSONAS: Dict[str, Dict[str, Any]] = {
    "Tech Professional": {
        "scenario_weights": [0.2, 0.3, 0.2, 0.1, 0.05, 0.1, 0.05], # Higher weight on work lunch/leisure
        "favorite_merchants": {"Cafes": ["Blue Bottle Coffee", "Philz Coffee"], "Tech": ["Apple Store"]},
        "spending_multiplier": 1.5,
        "subscriptions": ["Netflix", "Spotify", "Gym-Membership", "Cloud-Storage"],
    },
    "Student": {
        "scenario_weights": [0.3, 0.1, 0.3, 0.1, 0.15, 0.05, 0.0], # Budget-conscious, more commute/late-night
        "favorite_merchants": {"Cafes": ["Starbucks"], "Groceries": ["Trader Joe's"]},
        "spending_multiplier": 0.7,
        "subscriptions": ["Netflix", "Spotify"],
    },
    "Family": {
        "scenario_weights": [0.4, 0.1, 0.1, 0.2, 0.05, 0.1, 0.05], # High weight on groceries/shopping
        "favorite_merchants": {"Groceries": ["Costco", "Safeway"], "Shopping": ["Target"]},
        "spending_multiplier": 1.2,
        "subscriptions": ["Netflix", "Gym-Membership"],
    }
}


def apply_dynamic_tags(transaction: Dict[str, Any]) -> None:
    """Adds or modifies tags based on transaction data after generation."""
    if transaction['amount'] > 1000:
        transaction['tags'].add("Very-High-Ticket")
    if transaction['timestamp'].hour < 5:
        transaction['tags'].add("Early-Hours")
    # More rules can be added here


def generate_synthetic_data() -> pd.DataFrame:
    """Main function to generate a full, realistic dataset based on user personas."""
    all_transactions: List[Dict[str, Any]] = []

    # Calculate Zipf probabilities for scenarios (used if no persona-specific weights)
    num_scenarios = len(TRANSACTION_SCENARIOS)
    ranks = np.arange(1, num_scenarios + 1)
    zipf_probs = 1.0 / (ranks**ZIPF_EXPONENT)
    zipf_probs /= np.sum(zipf_probs)

    # 1. Generate a pool of users
    users = []
    for user_id in range(NUM_USERS):
        persona_name = random.choice(list(USER_PERSONAS.keys()))
        persona = USER_PERSONAS[persona_name]
        user = {
            "user_id": f"U{user_id+1}",
            "persona": persona_name,
            "details": persona,
            "user_subscriptions": {}
        }
        # Assign specific subscriptions to this user
        for sub_name in persona.get("subscriptions", []):
            sub_details = SUBSCRIPTIONS[sub_name]
            user["user_subscriptions"][sub_name] = {
                "merchant": random.choice(MERCHANT_DATA[sub_details["category"]]),
                "amount": round(random.uniform(*sub_details["amount_range"]), 2)
            }
        users.append(user)

    # 2. Generate transactions for each user
    for user in users:
        # A. Generate recurring subscription payments
        for sub_name, sub_info in user["user_subscriptions"].items():
            first_payment_day = random.randint(1, 28)
            current_date = SIMULATION_START_DATE + timedelta(days=first_payment_day)
            while current_date < SIMULATION_END_DATE:
                ts = current_date + timedelta(hours=random.randint(3, 20))
                all_transactions.append({
                    "user_id": user["user_id"],
                    "timestamp": ts,
                    "amount": sub_info["amount"] + round(random.uniform(-0.5, 0.5), 2), #Slight noise
                    "merchant_name": sub_info["merchant"],
                    "merchant_category": SUBSCRIPTIONS[sub_name]['category'],
                    "tags": {"Subscription", "Recurring", "Essential"},
                    "scenario": "Subscription"
                })
                current_date += timedelta(days=random.randint(28, 32)) # monthly-ish


        # B. Generate regular, scenario-based transactions
        total_days = (SIMULATION_END_DATE - SIMULATION_START_DATE).days
        num_transactions_for_user = int(np.random.normal(total_days * AVG_TRANSACTIONS_PER_USER_PER_DAY, 13))
        
        scenario_weights = user['details'].get("scenario_weights", zipf_probs)

        for _ in range(num_transactions_for_user):
            scenario: Dict[str, Any] = random.choices(TRANSACTION_SCENARIOS, weights=scenario_weights, k=1)[0]
            
            # Simplified timestamp generation
            random_day_offset = timedelta(days=random.randint(0, total_days))
            ts = SIMULATION_START_DATE + random_day_offset # further refined below
            # ... (timestamp refinement logic from previous version can be inserted here if needed) ...

            # Choose merchant: prioritize favorites, then use Faker for variety
            category = random.choice(scenario['categories'])
            fav_merchants = user['details']['favorite_merchants'].get(category)
            if fav_merchants and random.random() < 0.7: # 70% chance to use a favorite
                merchant_name = random.choice(fav_merchants)
            elif FAKER_AVAILABLE:
                merchant_name = fake.company()
            else:
                merchant_name = random.choice(MERCHANT_DATA[category])
            
            # Special override for gym check-ins
            if category == "Gym" and "Gym-Membership" in user["user_subscriptions"]:
                 merchant_name = user["user_subscriptions"]["Gym-Membership"]["merchant"]


            # Generate amount with persona multiplier
            pareto_val = np.random.pareto(scenario['pareto_shape']) * scenario['min_amount'] + scenario['min_amount']
            amount = round(np.clip(pareto_val * user['details']['spending_multiplier'], scenario['min_amount'], scenario['max_amount']), 2)

            transaction = {
                "user_id": user["user_id"],
                "timestamp": ts,
                "amount": amount,
                "merchant_name": merchant_name,
                "merchant_category": category,
                "tags": set(scenario["tags"]), # use set for easy addition
                "scenario": scenario["name"]
            }
            apply_dynamic_tags(transaction)
            all_transactions.append(transaction)

    # 3. Finalize DataFrame
    df = pd.DataFrame(all_transactions)
    df['tags'] = df['tags'].apply(lambda x: ", ".join(sorted(list(x))))
    
    # Add location data
    df['lat'] = SF_LOCATION_CENTER['lat'] + np.random.normal(0, 0.09, len(df))
    df['lon'] = SF_LOCATION_CENTER['lon'] + np.random.normal(0, 0.09, len(df))
    df['iso_country_code'] = 'USA'
    df['payment_channel'] = np.random.choice(['Card', 'Online'], size=len(df), p=[0.8, 0.2])

    return df.sort_values(by="timestamp").reset_index(drop=True)


if __name__ == "__main__":
    rich.print("[bold green]Starting sophisticated data generation process...[/bold green]")
    
    generated_df: pd.DataFrame = generate_synthetic_data()

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_filename: str = os.path.join(output_dir, "synthetic_transaction_data.csv")
    generated_df.drop(["tags", "scenario"], axis=1, inplace=True)
    generated_df.to_csv(output_filename, index=False, float_format='%.2f')
    
    rich.print(f"\n[bold green]✅ Success! Generated [cyan]{len(generated_df)}[/cyan] transactions for [cyan]{NUM_USERS}[/cyan] users.[/bold green]")
    rich.print(f"File saved to: '[bold yellow]{output_filename}[/bold yellow]'")
    
    print("\n--- Data Preview ---")
    rich.print(generated_df.head())
    
    # print("\n--- Persona Distribution ---")
    # # Map user_id to persona for analysis
    # user_persona_map = {f"U{i+1}": random.choice(list(USER_PERSONAS.keys())) for i in range(NUM_USERS)}
    # generated_df['persona'] = generated_df['user_id'].map(user_persona_map)
    # rich.print(generated_df['persona'].value_counts())
    
    # print("\n--- Spending Power by Persona ---")
    # rich.print(generated_df.groupby('persona')['amount'].agg(['mean', 'median', 'max']).round(2))