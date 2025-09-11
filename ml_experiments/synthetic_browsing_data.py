import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import rich

try:
    from faker import Faker

    fake = Faker()
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    # This initial print can remain standard as rich might not be imported yet
    print(
        "Faker library not found. Using generic names. To use Faker, run: pip install Faker"
    )

try:
    from rich.console import Console
    from rich.live import Live
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

    # Fallback if rich is not installed
    class DummyConsole:
        def print(self, *args, **kwargs):
            print(*args)

        def rule(self, title=""):
            print(f"\n--- {title} ---" if title else "\n---")

        def input(self, prompt: str, **kwargs) -> str:
            return input(prompt)

    console = DummyConsole()
    Panel = lambda text, title="", **kwargs: console.print(
        f"\n--- {title} ---\n{text}\n---" if title else f"\n---\n{text}\n---"
    )
    Text = lambda text, style="": str(text)

    # Dummy Progress and Table for fallback
    class DummyProgress:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def add_task(self, description, total=None, **kwargs):
            console.print(f"Starting: {description}")
            return 0  # Dummy task_id

        def update(self, task_id, advance=1, description=None, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    class DummyTable:
        def __init__(self, title="", **kwargs):
            self.title = title
            self.columns = []
            self.rows = []
            if self.title:
                console.print(f"\n{Text(self.title, style='bold')}")

        def add_column(self, header, style=None, justify="left", **kwargs):
            self.columns.append(header)

        def add_row(self, *items, **kwargs):
            self.rows.append([str(item) for item in items])

        def __rich_console__(self, console_obj, options):  # For console.print(table)
            if (
                self.title and not RICH_AVAILABLE
            ):  # Already printed by constructor for dummy
                pass
            elif self.title and RICH_AVAILABLE:
                console_obj.print(Text(self.title, style="bold magenta"))

            if self.columns:
                console_obj.print(" | ".join(map(str, self.columns)))
                console_obj.print(
                    "-"
                    * (
                        sum(len(str(c)) for c in self.columns)
                        + 3 * (len(self.columns) - 1)
                    )
                )
            for row in self.rows:
                console_obj.print(" | ".join(map(str, row)))
            yield ""  # Must be a generator

    if not RICH_AVAILABLE:
        print(
            "Rich library not found. Output will be basic. For a better experience, run: pip install rich"
        )
        Progress = DummyProgress
        Table = DummyTable


# --- Configuration Parameters ---
# (Configuration parameters remain the same)
NUM_PRODUCTS: int = 30000
PRODUCT_ID_PREFIX: str = "P"
CATEGORIES_L1: List[str] = [
    "Electronics",
    "Fashion",
    "Home & Kitchen",
    "Books",
    "Sports",
    "Beauty",
    "Toys",
    "Grocery",
    "Health",
]
SUBCATEGORIES_L2: Dict[str, List[str]] = {
    "Electronics": [
        "Mobiles",
        "Laptops",
        "Accessories",
        "Cameras",
        "Audio",
        "Wearables",
        "Smart Home",
    ],
    "Fashion": [
        "Men's Apparel",
        "Women's Apparel",
        "Footwear",
        "Watches",
        "Jewelry",
        "Kids' Fashion",
    ],
    "Home & Kitchen": [
        "Furniture",
        "Decor",
        "Appliances",
        "Cookware",
        "Storage",
        "Bedding",
        "Lighting",
    ],
    "Books": [
        "Fiction",
        "Non-Fiction",
        "Sci-Fi",
        "Mystery",
        "Comics",
        "Children's Books",
        "Academic",
    ],
    "Sports": [
        "Fitness Gear",
        "Outdoor Equipment",
        "Team Sports Gear",
        "Cycling",
        "Racquet Sports",
        "Water Sports",
    ],
    "Beauty": [
        "Skincare",
        "Makeup",
        "Haircare",
        "Fragrance",
        "Men's Grooming",
        "Wellness",
    ],
    "Toys": [
        "Action Figures",
        "Dolls",
        "Board Games",
        "Puzzles",
        "Educational Toys",
        "Remote Control",
    ],
    "Grocery": [
        "Pantry Staples",
        "Snacks",
        "Beverages",
        "Organic Foods",
        "Frozen Foods",
        "Dairy & Eggs",
    ],
    "Health": [
        "Vitamins",
        "Supplements",
        "Personal Care",
        "Medical Supplies",
        "Wellness Devices",
    ],
}
TAG_POOL: List[str] = [
    "new arrival",
    "on sale",
    "bestseller",
    "eco-friendly",
    "premium quality",
    "budget-friendly",
    "handmade",
    "imported",
    "locally sourced",
    "vintage style",
    "modern design",
    "for kids",
    "for adults",
    "gift idea",
    "essential item",
    "limited edition",
    "smart device",
    "wireless",
    "organic certified",
    "gluten-free",
    "vegan friendly",
    "top rated",
    "customer favorite",
    "durable",
    "lightweight",
    "heavy-duty",
    "easy to use",
    "collectible",
]
ZIPF_PARAM_A: float = 1.7

NUM_USERS: int = 2500
USER_ID_PREFIX: str = "U"

SESSION_DURATION_MEAN_LOG: float = np.log(900)
SESSION_DURATION_SIGMA_LOG: float = 0.8

INTERACTIONS_PER_SESSION_MEAN_LOG: float = np.log(8)
INTERACTIONS_PER_SESSION_SIGMA_LOG: float = 0.9

interaction_probs: Dict[str, float] = {
    "search_results_click": 0.16,
    "click": 0.05,
    "click_recommendation_top5": 0.05,
    "click_recommendation_beyond": 0.03,
    "scroll": 0.26,
    "hover": 0.15,
    "view_details": 0.1,
    "quick_view": 0.1,
    "add_to_cart": 0.05,
    "remove_from_cart": 0.03,
    "purchase": 0.02,
}
INTERACTION_TYPES: List[str] = list(interaction_probs.keys())
INTERACTION_WEIGHTS: List[float] = list(interaction_probs.values())
# These prints are for validation, can be kept or removed
# print(f"Sum of INTERACTION_WEIGHTS: {sum(INTERACTION_WEIGHTS)}")
assert abs(sum(INTERACTION_WEIGHTS) - 1.0) < 1e-9, "INTERACTION_WEIGHTS must sum to 1.0"

MEAN_TIME_BETWEEN_INTERACTIONS_SECONDS: int = 75

HOUR_OF_DAY_PROBS: List[float] = [
    0.005,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.07,
    0.06,
    0.06,
    0.06,
    0.07,
    0.07,
    0.08,
    0.06,
    0.04,
    0.02,
    0.005,
]
# print(f"Sum of HOUR_OF_DAY_PROBS: {sum(HOUR_OF_DAY_PROBS)}")
assert abs(sum(HOUR_OF_DAY_PROBS) - 1.0) < 1e-9, "HOUR_OF_DAY_PROBS must sum to 1.0"

MIN_SAVINGS_AMOUNT: float = 3.0
MAX_SAVINGS_AMOUNT: float = 25.0
PARETO_SHAPE_FOR_HIGH_POPULARITY: float = 4.0
PARETO_SHAPE_FOR_LOW_POPULARITY: float = 1.5
SAVINGS_NOISE_RATIO: float = 0.30

# --- Generator Functions ---


def generate_product_data(
    num_products: int,
    categories_l1: List[str],
    subcategories_l2: Dict[str, List[str]],
    tag_pool: List[str],
    zipf_param_a: float,
) -> pd.DataFrame:
    console.print(Text(f"Generating {num_products} products...", style="cyan"))
    products: List[Dict[str, Any]] = []
    product_ids: List[str] = [f"{PRODUCT_ID_PREFIX}{i+1}" for i in range(num_products)]

    ranks: np.ndarray = np.arange(1, num_products + 1)
    if zipf_param_a == 0:
        probabilities_by_rank: np.ndarray = (
            np.ones(num_products) / num_products if num_products > 0 else np.array([])
        )
    else:
        probabilities_by_rank: np.ndarray = 1.0 / (ranks**zipf_param_a)
        probabilities_by_rank /= np.sum(probabilities_by_rank)

    shuffled_ranks: np.ndarray = np.copy(ranks)
    np.random.shuffle(shuffled_ranks)

    max_pop_score_possible: float
    min_pop_score_possible: float
    if num_products > 0:
        max_pop_score_possible = float(probabilities_by_rank[0])
        min_pop_score_possible = float(probabilities_by_rank[num_products - 1])
    else:
        max_pop_score_possible = 0.0
        min_pop_score_possible = 0.0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
        disable=not RICH_AVAILABLE,  # Disable if rich is not available
    ) as progress_bar:
        task = progress_bar.add_task("Processing products", total=num_products)
        for i in range(num_products):
            prod_id: str = product_ids[i]
            category_l1: str = random.choice(categories_l1)
            possible_subcategories: List[str] = subcategories_l2.get(
                category_l1, ["General SubCategory"]
            )
            if not possible_subcategories:
                possible_subcategories = ["General SubCategory"]
            subcategory_l2: str = random.choice(possible_subcategories)
            num_tags: int = random.randint(1, min(5, len(tag_pool)))
            tags: List[str] = random.sample(tag_pool, num_tags)
            current_product_rank: int = int(shuffled_ranks[i])
            popularity_score: float = float(
                probabilities_by_rank[current_product_rank - 1]
            )
            current_pareto_a: float = PARETO_SHAPE_FOR_LOW_POPULARITY
            if num_products == 1 or max_pop_score_possible == min_pop_score_possible:
                current_pareto_a = (
                    PARETO_SHAPE_FOR_HIGH_POPULARITY + PARETO_SHAPE_FOR_LOW_POPULARITY
                ) / 2
            else:
                normalized_popularity: float = (
                    popularity_score - min_pop_score_possible
                ) / (max_pop_score_possible - min_pop_score_possible)
                current_pareto_a = (
                    PARETO_SHAPE_FOR_LOW_POPULARITY
                    + normalized_popularity
                    * (
                        PARETO_SHAPE_FOR_HIGH_POPULARITY
                        - PARETO_SHAPE_FOR_LOW_POPULARITY
                    )
                )
            current_pareto_a = max(current_pareto_a, 0.1)
            base_savings: float = float(
                np.random.pareto(current_pareto_a) * MIN_SAVINGS_AMOUNT
            )
            max_noise_value: float = (
                MAX_SAVINGS_AMOUNT - MIN_SAVINGS_AMOUNT
            ) * SAVINGS_NOISE_RATIO
            noise: float = float(
                np.random.uniform(-max_noise_value / 2, max_noise_value / 2)
            )
            savings_with_noise: float = base_savings + noise
            final_savings: float = round(
                float(
                    np.clip(savings_with_noise, MIN_SAVINGS_AMOUNT, MAX_SAVINGS_AMOUNT)
                ),
                2,
            )
            products.append(
                {
                    "ProductID": prod_id,
                    "CategoryL1": category_l1,
                    "CategoryL2": subcategory_l2,
                    "Tags": ", ".join(tags),
                    "PopularityRank": current_product_rank,
                    "PopularityScore": popularity_score,
                    "SavingAmount": final_savings,
                }
            )
            progress_bar.update(task, advance=1)

    products_df: pd.DataFrame = pd.DataFrame(products)
    console.print(Text("Product data generation complete.", style="green"))
    return products_df


def generate_user_ids(num_users: int) -> List[str]:
    console.print(Text(f"Generating {num_users} user IDs...", style="cyan"))
    user_ids: List[str] = [f"{USER_ID_PREFIX}{i+1}" for i in range(num_users)]
    console.print(Text("User ID generation complete.", style="green"))
    return user_ids


def generate_session_interactions(
    session_id: str, user_id: str, products_df: pd.DataFrame, base_start_time: datetime
) -> List[Dict[str, Any]]:
    session_interactions: List[Dict[str, Any]] = []
    session_duration_seconds: int = max(
        30,
        int(
            np.random.lognormal(
                mean=SESSION_DURATION_MEAN_LOG, sigma=SESSION_DURATION_SIGMA_LOG
            )
        ),
    )
    num_interactions: int = max(
        1,
        int(
            np.random.lognormal(
                mean=INTERACTIONS_PER_SESSION_MEAN_LOG,
                sigma=INTERACTIONS_PER_SESSION_SIGMA_LOG,
            )
        ),
    )
    current_time: datetime = base_start_time
    product_ids_for_sampling: np.ndarray = products_df["ProductID"].values
    product_popularity_scores_for_sampling: np.ndarray = products_df[
        "PopularityScore"
    ].values.astype(
        float
    )  # Ensure float for np.random.choice

    sum_scores: float = np.sum(product_popularity_scores_for_sampling)
    if abs(sum_scores - 1.0) > 1e-5 and sum_scores != 0:
        product_popularity_scores_for_sampling = (
            product_popularity_scores_for_sampling / sum_scores
        )
    elif sum_scores == 0 and len(product_popularity_scores_for_sampling) > 0:
        product_popularity_scores_for_sampling = np.ones(
            len(product_ids_for_sampling)
        ) / len(product_ids_for_sampling)
    elif len(product_ids_for_sampling) == 0:
        return []

    session_cart: Set[str] = set()
    for i in range(num_interactions):
        if i > 0:
            time_delta_seconds: int = max(
                1,
                int(
                    np.random.exponential(scale=MEAN_TIME_BETWEEN_INTERACTIONS_SECONDS)
                ),
            )
            current_time += timedelta(seconds=time_delta_seconds)
        if (
            current_time - base_start_time
        ).total_seconds() > session_duration_seconds * 1.5 and i > 0:
            break
        interaction_type: str = str(
            np.random.choice(INTERACTION_TYPES, p=INTERACTION_WEIGHTS)
        )
        if len(product_ids_for_sampling) == 0:
            continue

        # Ensure p sums to 1 for np.random.choice
        if not np.isclose(np.sum(product_popularity_scores_for_sampling), 1.0):
            if (
                np.sum(product_popularity_scores_for_sampling) == 0
                and len(product_popularity_scores_for_sampling) > 0
            ):
                product_popularity_scores_for_sampling = np.ones(
                    len(product_ids_for_sampling)
                ) / len(product_ids_for_sampling)
            elif (
                np.sum(product_popularity_scores_for_sampling) > 0
            ):  # Normalize if sum is not 1 but > 0
                product_popularity_scores_for_sampling = (
                    product_popularity_scores_for_sampling
                    / np.sum(product_popularity_scores_for_sampling)
                )
            else:  # Fallback if scores are problematic (e.g. all negative, though unlikely here)
                chosen_product_id: str = str(
                    np.random.choice(product_ids_for_sampling)
                )  # Uniform choice
                product_details: pd.Series = products_df[
                    products_df["ProductID"] == chosen_product_id
                ].iloc[0]
                # ... rest of interaction logic ...
                continue  # Skip to next interaction if p is still bad

        chosen_product_id: str = str(
            np.random.choice(
                product_ids_for_sampling, p=product_popularity_scores_for_sampling
            )
        )
        product_details: pd.Series = products_df[
            products_df["ProductID"] == chosen_product_id
        ].iloc[0]

        if interaction_type == "add_to_cart":
            session_cart.add(chosen_product_id)
        elif interaction_type == "remove_from_cart":
            if chosen_product_id in session_cart:
                session_cart.remove(chosen_product_id)
            else:
                interaction_type = (
                    "view_details" if random.random() < 0.8 else "search_results_click"
                )
        elif interaction_type == "start_checkout":
            if not session_cart:
                interaction_type = "scroll"
        elif interaction_type == "purchase":
            if not session_cart and chosen_product_id not in session_cart:
                session_cart.add(chosen_product_id)

        session_interactions.append(
            {
                "SessionID": session_id,
                "UserID": user_id,
                "Timestamp": current_time,
                "InteractionType": interaction_type,
                "ProductID": chosen_product_id,
                "ProductCategoryL1": str(product_details["CategoryL1"]),
                "ProductCategoryL2": str(product_details["CategoryL2"]),
                "ProductTags": str(product_details["Tags"]),
                "SessionCartSize": len(session_cart),
            }
        )
        if interaction_type == "purchase" and chosen_product_id in session_cart:
            session_cart.clear()
    return session_interactions


def generate_synthetic_data(num_sessions_to_generate: int) -> pd.DataFrame:
    products_df: pd.DataFrame = generate_product_data(
        NUM_PRODUCTS, CATEGORIES_L1, SUBCATEGORIES_L2, TAG_POOL, ZIPF_PARAM_A
    )
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    product_file_path = os.path.join(output_dir, "synthetic_product_info.csv")
    products_df.to_csv(product_file_path, index=False)
    console.print(
        f"Product info saved to [link=file://{os.path.abspath(product_file_path)}]{product_file_path}[/link]",
        style="dim",
    )

    user_ids: List[str] = generate_user_ids(NUM_USERS)
    all_interactions_data: List[Dict[str, Any]] = []
    console.print(
        Text(
            f"\nGenerating {num_sessions_to_generate} session histories...",
            style="cyan",
        )
    )

    simulation_end_date: datetime = datetime.now()
    simulation_start_date: datetime = simulation_end_date - timedelta(days=365)
    total_days_in_window: int = (simulation_end_date - simulation_start_date).days

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
        disable=not RICH_AVAILABLE,
    ) as progress_bar:
        task = progress_bar.add_task(
            "Generating sessions", total=num_sessions_to_generate
        )
        for i in range(num_sessions_to_generate):
            session_id: str = f"S{i+1:07d}"
            current_user_id: str = random.choice(user_ids)
            days_offset: int = random.randint(0, total_days_in_window)
            session_date: datetime = simulation_start_date + timedelta(days=days_offset)
            hour: int = int(np.random.choice(np.arange(24), p=HOUR_OF_DAY_PROBS))
            minute: int = random.randint(0, 59)
            second: int = random.randint(0, 59)
            base_session_start_time: datetime = datetime(
                session_date.year,
                session_date.month,
                session_date.day,
                hour,
                minute,
                second,
            )
            if products_df.empty or products_df["PopularityScore"].sum() == 0:
                console.print(
                    Text(
                        f"Warning: Product data is empty or has no valid popularity scores. Skipping session {session_id}.",
                        style="yellow",
                    )
                )
                progress_bar.update(task, advance=1)
                continue

            session_interactions: List[Dict[str, Any]] = generate_session_interactions(
                session_id, current_user_id, products_df, base_session_start_time
            )
            all_interactions_data.extend(session_interactions)
            progress_bar.update(task, advance=1)

    console.print(Text("\nSynthetic data generation complete.", style="bold green"))
    if not all_interactions_data:
        console.print(
            Text(
                "Warning: No interactions were generated. Check parameters or number of sessions.",
                style="yellow bold",
            )
        )
        return pd.DataFrame()

    final_df: pd.DataFrame = pd.DataFrame(all_interactions_data)
    column_order: List[str] = [
        "SessionID",
        "UserID",
        "Timestamp",
        "InteractionType",
        "ProductID",
        "ProductCategoryL1",
        "ProductCategoryL2",
        "ProductTags",
        "SessionCartSize",
    ]
    final_df_columns: List[str] = [
        col for col in column_order if col in final_df.columns
    ]
    final_df = final_df[final_df_columns]
    return final_df.sort_values(by=["SessionID", "Timestamp"]).reset_index(drop=True)


if __name__ == "__main__":
    console.print(
        Panel(
            Text(
                "Synthetic User Session Data Generator",
                justify="center",
                style="bold blue",
            ),
            expand=False,
        )
    )
    num_sessions_input: int = 0
    while True:
        try:
            prompt_text = Text(
                "Enter the number of session histories to generate (e.g., 1000): ",
                style="input",
            )
            num_sessions_input_str = console.input(
                prompt_text if RICH_AVAILABLE else str(prompt_text)
            )
            num_sessions_input = int(num_sessions_input_str)
            if num_sessions_input > 0:
                break
            else:
                console.print(
                    Text(
                        "Please enter a positive integer greater than 0.",
                        style="yellow",
                    )
                )
        except ValueError:
            console.print(Text("Invalid input. Please enter an integer.", style="red"))
        except KeyboardInterrupt:
            console.print(Text("\nGeneration cancelled by user.", style="bold red"))
            exit()

    generated_df: pd.DataFrame = generate_synthetic_data(num_sessions_input)

    if not generated_df.empty:
        console.rule(
            Text(
                "Sample of Generated Data (First 10 interactions)", style="bold magenta"
            )
        )
        sample_table = Table(
            show_header=True, header_style="bold cyan", border_style="dim"
        )
        # Add columns dynamically from the DataFrame
        for col in generated_df.columns[
            :10
        ]:  # Show first 10 columns for brevity if too many
            sample_table.add_column(str(col))

        for index, row in generated_df.head(10).iterrows():
            sample_table.add_row(*[str(item) for item in row.values[:10]])
        console.print(Padding(sample_table, (1, 0)))

        console.rule(Text("Data Summary", style="bold magenta"))
        summary_table = Table(
            show_header=False,
            border_style="dim",
            box=None if not RICH_AVAILABLE else ...,
        )  # rich.box.ROUNDED
        if RICH_AVAILABLE and hasattr(rich.box, "ROUNDED"):
            summary_table.box = rich.box.ROUNDED

        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", style="bold green")
        summary_table.add_row("Total interactions generated:", str(len(generated_df)))
        summary_table.add_row(
            "Number of unique sessions:", str(generated_df["SessionID"].nunique())
        )
        summary_table.add_row(
            "Number of unique users:", str(generated_df["UserID"].nunique())
        )
        summary_table.add_row(
            "Number of unique products:", str(generated_df["ProductID"].nunique())
        )
        console.print(Padding(summary_table, (1, 0)))

        output_dir = "data"  # Ensure this is consistent
        os.makedirs(output_dir, exist_ok=True)
        output_filename: str = os.path.join(
            output_dir, "synthetic_session_data.csv"
        )
        try:
            generated_df.to_csv(output_filename, index=False)
            console.print(
                Text(
                    f"\nFull dataset saved to [link=file://{os.path.abspath(output_filename)}]{output_filename}[/link]",
                    style="bold green",
                )
            )
        except Exception as e:
            console.print(Text(f"\nCould not save data to CSV: {e}", style="bold red"))
            console.print(
                Text(
                    "Displaying a larger sample instead (first 100 rows):",
                    style="yellow",
                )
            )
            # Fallback to simple print for large data if table formatting is too slow or complex without rich
            if RICH_AVAILABLE:
                large_sample_table = Table(
                    show_header=True, header_style="bold cyan", border_style="dim"
                )
                for col in generated_df.columns:
                    large_sample_table.add_column(str(col))
                for index, row in generated_df.head(100).iterrows():
                    large_sample_table.add_row(*[str(item) for item in row.values])
                with console.pager():
                    console.print(large_sample_table)
            else:
                print(generated_df.head(100).to_string())

    else:
        console.print(Text("No data was generated.", style="bold yellow"))

    console.print(
        Panel(Text("Done!", justify="center", style="bold blue"), expand=False)
    )
