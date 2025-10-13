"""
This plan implements a two-stage ranking cascade with two-tower retrieval followed by multi-task ranking, achieving sub-100ms latency while capturing evolving user preferences.
The architecture combines FIT-inspired two-tower models for candidate generation, self-normalized IPS for unbiased training, and HoME-style multi-gate experts for final ranking.
Expected improvements: 15-20% CTR increase, 25-30% task completion rate improvement, and 10-15% cross-sell conversion uplift based on industry benchmarks.
"""

import json
import math
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# --- Configuration & Data Pools ---

NUM_SAMPLES = 10000
MAX_RESULTS_PER_QUERY = 4  # Total number of links shown for a query

# Pools of data to make the logs more realistic. Queries based on bankin queries for things retail customers would want to access
QUERIES = [
    "transaction history", "spending trend",
    "lost my credit card", "how to change password",
    "card benefits", "increase account limit", "how to apply for credit card",
]


DOMAINS = [
    "https://www.bank-abc.com", "https://www.bank-xyz.com"

]

URL_PATHS = [
    "articles", "products", "guides", "reviews", "listings", "posts", "items"
]

SKIP_TYPES = ["impression", "skip_after_examination", "user_scrolled_past"]


def generate_session(session_index: int, user_id_pool: list) -> dict:
    """
    Generates a single, randomized multi-engagement session log.

    Args:
        session_index: The index of the session, used for creating a unique session ID.
        user_id_pool: A list of user IDs to sample from, allowing users to have multiple sessions.

    Returns:
        A dictionary representing the session log.
    """

    # 1. Basic Session Information
    session_query = random.choice(QUERIES)
    session_user_id = random.choice(user_id_pool)
    session_id = f"s{session_index + 1}"

    # Generate a random timestamp within the last 30 days
    now = datetime.now(timezone.utc)
    session_start_dt = now - timedelta(seconds=random.randint(0, 30 * 24 * 3600))
    session_start_epoch = int(session_start_dt.timestamp())

    # 2. Simulate Search Results
    # Generate a list of unique URLs for this session's search results
    base_domain = random.choice(DOMAINS)
    results_urls = [
        f"{base_domain}/{random.choice(URL_PATHS)}/{session_query.replace(' ', '-')}-{i}"
        for i in range(1, MAX_RESULTS_PER_QUERY + 1)
    ]

    # 3. Simulate User Engagement
    # Users are more likely to click 1-3 results.
    num_clicks = random.randint(1, 4)

    # Users are biased towards clicking higher-ranked results.
    # We use a weighted choice to simulate this.
    possible_ranks = list(range(1, MAX_RESULTS_PER_QUERY + 1))
    # Weights are skewed towards the top results (e.g., rank 1 is 10x more likely to be picked than rank 10)
    rank_weights = [1 / r for r in possible_ranks]

    clicked_ranks = sorted(random.choices(possible_ranks, weights=rank_weights, k=num_clicks))
    # Ensure clicked ranks are unique
    clicked_ranks = sorted(list(set(clicked_ranks)))

    shown_but_not_clicked_ranks = [r for r in possible_ranks if r not in clicked_ranks]

    # 4. Build Clicked and Not-Clicked Lists
    clicked_links = []
    current_time_epoch = session_start_epoch

    for rank in clicked_ranks:
        # Time passes between clicks
        click_delay = random.uniform(3.5, 15.0)
        current_time_epoch += int(click_delay)

        dwell_time = round(random.uniform(10.0, 180.0), 2)  # Dwell time on the page

        # Randomize key names to match the example's inconsistency
        click = {
            random.choice(["url", "nav_link", "link"]): results_urls[rank - 1],
            random.choice(["rank", "position"]): rank,
            random.choice(["click_timestamp", "click_time", "timestamp"]): current_time_epoch,
            random.choice(["dwell_seconds", "dwell", "time_on_page"]): dwell_time
        }
        clicked_links.append(click)

    shown_but_not_clicked = []
    for rank in shown_but_not_clicked_ranks:
        impression = {
            random.choice(["url", "link"]): results_urls[rank - 1],
            random.choice(["rank", "position"]): rank,
            random.choice(["impression_time", "timestamp"]): session_start_epoch,
        }
        # Occasionally add a skip_type
        if random.random() > 0.6:  # 40% chance
            impression["skip_type"] = random.choice(SKIP_TYPES)
        shown_but_not_clicked.append(impression)

    # 5. Assemble Final Session Object with Randomized Schema
    session = {
        "query": session_query,
        "user_id": session_user_id,
        "session_id": session_id,
    }

    # Randomly choose the timestamp format
    if random.random() > 0.5:
        session["timestamp"] = session_start_dt
    else:
        session["timestamp"] = session_start_epoch

    # Randomly choose the key names for the lists
    if random.random() > 0.5:
        session["clicked_links"] = clicked_links
        session["shown_but_not_clicked"] = shown_but_not_clicked
    else:
        session["clicks"] = clicked_links
        session["impressions"] = shown_but_not_clicked

    return session


class ClickLogProcessor:
    """
    Processes search/navigation sessions into a training dataset with:
      - Positive samples (clicked)
      - Negative samples (shown but not clicked) and optionally sampled negatives from a large pool
      - Per-example position-based propensity scores
      - IPS (inverse propensity) and self-normalized weights (SNIPS)

    New in this version:
      - Mixed negative sampling: 70% hard negatives sampled from ranks [hard_min_rank..hard_max_rank], 30% uniform
        across a provided candidate pool (session["candidates"] | session["all_nav_links"]).
      - Power-law propensity model: propensity[position] = 1 / (1 + position) ** gamma (default gamma=0.5).
      - Configurable IPS computation with optional target policy callback and SNIPS normalization mode.

    The resulting dataset can be written in JSON or JSON Lines (JSONL) format.

    Assumed session schema (robust to slight variations in key names):
      session = {
        "query": str,
        "user_id": str/int,
        "session_id": str/int (optional),
        "timestamp": datetime/int/float/str (optional),
        "clicked_links" | "clicks": [
          {
            "nav_link" | "url" | "link": str,
            "position" | "rank": int (1-based),
            "click_time" | "click_timestamp" | "timestamp": datetime/int/float/str (optional),
            "dwell_time" | "dwell_seconds" | "dwell" | "time_on_page": float (optional)
          }, ...
        ],
        "shown_but_not_clicked" | "impressions": [
          {
            "nav_link" | "url" | "link": str,
            "position" | "rank": int (1-based),
            "shown_time" | "impression_time" | "timestamp" | "time": datetime/int/float/str (optional),
            "skip_type": "impression" | "skip_after_examination" | "user_scrolled_past" (optional)
          }, ...
        ],

        # Optional candidate pool for negative sampling (large list):
        "candidates" | "all_nav_links": [
          {
            "nav_link" | "url" | "link": str,
            "position" | "rank": int (1-based, larger than top-K allowed),
            # ... any other metadata
          }, ...
        ]
      }

    Notes:
      - Replace calculate_position_bias() with calibrated estimates from logs when available.
      - Mixed negative sampling only activates if a candidate pool is provided; otherwise, logs-only negatives are used.
      - IPS/SNIPS weights are provided to support debiased training.
    """

    def __init__(
        self,
        min_propensity: float = 0.05,
        epsilon: float = 1e-6,
        position_propensity_map: Optional[Dict[int, float]] = None,
        # Position bias model: "inverse_log" (legacy) or "power_law"
        propensity_model: str = "power_law",
        power_law_gamma: float = 0.5,
        # IPS target policy; if None, assumes π_target(action|context) = 1.0
        target_policy: Optional[Callable[[Dict[str, Any]], float]] = None,
        # SNIPS mode: if True, SNIPS = w_i * (N / sum_j w_j); if False, normalized = w_i / sum_j w_j
        snips_include_n_factor: bool = True,
        # Mixed negative sampling config
        enable_mixed_negative_sampling: bool = True,
        negatives_per_positive: int = 3,
        hard_fraction: float = 0.7,
        hard_min_rank: int = 101,
        hard_max_rank: int = 1000,
    ) -> None:
        """
        Args:
          min_propensity: Lower bound to avoid extreme IPS weights.
          epsilon: Small constant to avoid division by zero.
          position_propensity_map: Optional mapping from position (1-based) to propensity.
          propensity_model: "inverse_log" or "power_law".
          power_law_gamma: Exponent for the power-law propensity model.
          target_policy: Optional callable: rec -> probability under the target policy.
          snips_include_n_factor: Use classical SNIPS scaling (N / sum w) if True, else normalize by sum w.
          enable_mixed_negative_sampling: Enable 70/30 hard/random negative sampling when a candidate pool is present.
          negatives_per_positive: Target number of sampled negatives per positive example.
          hard_fraction: Fraction of sampled negatives from hard pool [hard_min_rank..hard_max_rank].
          hard_min_rank: Lower bound (inclusive) for hard negative rank sampling (1-based).
          hard_max_rank: Upper bound (inclusive) for hard negative rank sampling (1-based).
        """
        self.min_propensity = float(min_propensity)
        self.epsilon = float(epsilon)
        self.position_propensity_map = dict(position_propensity_map) if position_propensity_map else None

        self.propensity_model = propensity_model
        self.power_law_gamma = float(power_law_gamma)

        self.target_policy = target_policy
        self.snips_include_n_factor = bool(snips_include_n_factor)

        self.enable_mixed_negative_sampling = bool(enable_mixed_negative_sampling)
        self.negatives_per_positive = int(negatives_per_positive)
        self.hard_fraction = float(hard_fraction)
        self.hard_min_rank = int(hard_min_rank)
        self.hard_max_rank = int(hard_max_rank)

    # -----------------
    # Public API
    # -----------------
    def process_session(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert a single session into a list of labeled examples with propensity/IPS weights.

        Returns:
          A list of dictionaries suitable for JSON serialization.
        """
        query = self._get(session, ["query"])
        user_id = self._get(session, ["user_id"])
        session_id = self._get(session, ["session_id", "id"], default=None)
        session_time = self._to_iso8601(self._get(session, ["timestamp", "session_time", "start_time"], default=None))

        positives_src = self._get(session, ["clicked_links", "clicks"], default=[]) or []
        negatives_src = self._get(session, ["shown_but_not_clicked", "impressions"], default=[]) or []

        records: List[Dict[str, Any]] = []

        # Positives (clicked)
        positive_links: Set[str] = set()
        for item in positives_src:
            nav_link = self._get(item, ["nav_link", "url", "link"])
            position = self._get(item, ["position", "rank"], default=None)
            click_time = self._to_iso8601(self._get(item, ["click_time", "click_timestamp", "timestamp"], default=None))
            dwell_time = self._get(item, ["dwell_time", "dwell_seconds", "dwell", "time_on_page"], default=None)

            propensity = self.calculate_position_bias(position)
            ips_weight = self._ips_weight(position, base_record={"type": "positive", "query": query, "nav_link": nav_link})

            rec = {
                "type": "positive",
                "label": 1,
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "session_time": session_time,
                "nav_link": nav_link,
                "position": position,
                "click_time": click_time,
                "dwell_time": dwell_time,
                "propensity": propensity,
                "ips_weight": ips_weight,
            }
            records.append(rec)
            if isinstance(nav_link, str):
                positive_links.add(nav_link)

        # Negatives from logs (shown but not clicked)
        existing_negative_links: Set[str] = set()
        for item in negatives_src:
            nav_link = self._get(item, ["nav_link", "url", "link"])
            position = self._get(item, ["position", "rank"], default=None)
            shown_time = self._to_iso8601(self._get(item, ["shown_time", "impression_time", "timestamp", "time"], default=None))
            skip_type = self._get(item, ["skip_type"], default="impression")

            propensity = self.calculate_position_bias(position)
            ips_weight = self._ips_weight(position, base_record={"type": "negative", "query": query, "nav_link": nav_link})

            rec = {
                "type": "negative",
                "label": 0,
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "session_time": session_time,
                "nav_link": nav_link,
                "position": position,
                "shown_time": shown_time,
                "skip_type": skip_type,
                "propensity": propensity,
                "ips_weight": ips_weight,
            }
            if isinstance(nav_link, str) and nav_link not in positive_links:
                records.append(rec)
                existing_negative_links.add(nav_link)

        # Mixed negative sampling (70% hard, 30% random) from candidate pool, if available
        if self.enable_mixed_negative_sampling:
            sampled_negatives = self._maybe_sample_negatives(session, positive_links, existing_negative_links)
            for nav_link, position, source in sampled_negatives:
                propensity = self.calculate_position_bias(position)
                ips_weight = self._ips_weight(position, base_record={"type": "negative", "query": query, "nav_link": nav_link})

                rec = {
                    "type": "negative",
                    "label": 0,
                    "query": query,
                    "user_id": user_id,
                    "session_id": session_id,
                    "session_time": session_time,
                    "nav_link": nav_link,
                    "position": position,
                    "shown_time": session_time,  # synthetic; align to session time
                    "skip_type": source,         # "sampled_hard" or "sampled_random"
                    "propensity": propensity,
                    "ips_weight": ips_weight,
                }
                # Avoid duplicates with existing records
                if nav_link not in existing_negative_links and nav_link not in positive_links:
                    records.append(rec)
                    existing_negative_links.add(nav_link)

        # Self-normalized IPS weights per session (or simple normalization by sum)
        self._attach_snips_weights(records)

        return records

    def build_dataset(
        self,
        sessions: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple sessions into a flat list of training examples.
        """
        dataset: List[Dict[str, Any]] = []
        for session in sessions:
            dataset.extend(self.process_session(session))
        return dataset

    def write_json(
        self,
        records: Sequence[Dict[str, Any]],
        output_path: str,
        jsonl: bool = True,
        indent: Optional[int] = None,
    ) -> None:
        """
        Write records to disk in JSON or JSON Lines format.

        Args:
          records: List of dicts from process_session/build_dataset.
          output_path: Destination file path.
          jsonl: True to write a JSONL file (one JSON object per line).
          indent: Pretty-print indent for standard JSON. Ignored for JSONL.
        """
        if jsonl:
            with open(output_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(list(records), f, ensure_ascii=False, indent=indent)

    def create_dataset_json(
        self,
        sessions: Iterable[Dict[str, Any]],
        output_path: str,
        jsonl: bool = True,
        indent: Optional[int] = None,
    ) -> None:
        """
        Convenience method: build dataset and write to JSON/JSONL.
        """
        data = self.build_dataset(sessions)
        self.write_json(data, output_path, jsonl=jsonl, indent=indent)

    # -----------------
    # Position propensity
    # -----------------
    def calculate_position_bias(self, position: Optional[int]) -> float:
        """
        Estimate a position-based propensity (probability of click due to position).

        Priority:
          1) If position_propensity_map is provided, use that directly.
          2) Else use the selected model:
             - inverse_log (legacy): 1 / log2(position + 1)
             - power_law: 1 / (1 + position) ** gamma

        Returns:
          A float in (min_propensity, 1.0], clipped to [min_propensity, 1.0].
        """
        if position is None:
            # Unknown position; fall back to a conservative default
            return float(max(0.2, self.min_propensity))

        p: float
        if self.position_propensity_map is not None:
            p = float(self.position_propensity_map.get(int(position), self.min_propensity))
        else:
            try:
                pos = max(1, int(position))
            except (TypeError, ValueError):
                pos = 1

            if self.propensity_model == "inverse_log":
                try:
                    p = 1.0 / math.log2(pos + 1.0)
                except (ValueError, ZeroDivisionError):
                    p = 0.5
            else:
                # power_law (default)
                gamma = self.power_law_gamma
                p = 1.0 / pow(1.0 + float(pos), float(gamma))

        # Clip to safe range
        return float(min(1.0, max(self.min_propensity, p)))

    # -----------------
    # Helpers
    # -----------------
    @staticmethod
    def _get(obj: Any, keys: List[str], default: Any = None) -> Any:
        """
        Robustly extract a value from a dict-like or object-like structure using a list of candidate keys.
        Returns the first found value or default if none are present.
        """
        for key in keys:
            # dict-like
            if isinstance(obj, dict) and key in obj:
                return obj[key]
            # object-like
            if hasattr(obj, key):
                return getattr(obj, key)
        return default

    @staticmethod
    def _to_iso8601(value: Any) -> Optional[str]:
        """
        Convert various time representations to ISO-8601 string in UTC, if possible.
        Accepts:
          - datetime
          - int/float epoch seconds or milliseconds
          - string (returned as-is)
          - None
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()

        if isinstance(value, (int, float)):
            # Heuristic: treat values >= 1e11 as milliseconds
            ts = float(value)
            if ts >= 1e11:
                ts /= 1000.0
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.isoformat()

        if isinstance(value, str):
            # Assume already a timestamp string; caller's responsibility for format
            return value

        # Fallback to string conversion
        return str(value)

    def _ips_weight(self, position: Optional[int], base_record: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute IPS weight = π_target(action|context) / π_logging(action|context)
        where π_logging is proxied by position-based propensity, and π_target defaults to 1.0.
        """
        propensity = self.calculate_position_bias(position)
        target_prob = 1.0
        if self.target_policy and base_record is not None:
            try:
                target_prob = float(self.target_policy(base_record))
            except Exception:
                target_prob = 1.0
        return float(target_prob) / max(float(propensity), self.epsilon)

    def _attach_snips_weights(self, records: List[Dict[str, Any]]) -> None:
        """
        Compute self-normalized weights:
          - If snips_include_n_factor: SNIPS normalizes IPS weights to reduce variance:
              snips_weight_i = ips_weight_i * (N / sum_j ips_weight_j)
          - Else: normalized_weight = ips_weight_i / sum_j ips_weight_j
        """
        ips_values = [r.get("ips_weight") for r in records if isinstance(r.get("ips_weight"), (int, float))]
        n = len(ips_values)
        s = float(sum(ips_values)) if n > 0 else 0.0

        if n == 0 or s <= 0.0:
            for r in records:
                r["snips_weight"] = 1.0
            return

        if self.snips_include_n_factor:
            scale = n / s
            for r in records:
                ips = r.get("ips_weight")
                r["snips_weight"] = float(ips) * scale if isinstance(ips, (int, float)) else 1.0
        else:
            for r in records:
                ips = r.get("ips_weight")
                r["snips_weight"] = float(ips) / s if isinstance(ips, (int, float)) else 1.0

    # ----- Mixed negative sampling -----
    def _iter_candidates(self, session: Dict[str, Any]) -> List[Tuple[str, Optional[int]]]:
        """
        Returns list of (nav_link, position) from an optional candidate pool provided in the session.
        Accepts keys: "candidates" or "all_nav_links". Robust to key variations inside items.
        """
        pool = self._get(session, ["candidates", "all_nav_links"], default=[]) or []
        candidates: List[Tuple[str, Optional[int]]] = []
        for item in pool:
            nav_link = self._get(item, ["nav_link", "url", "link"])
            if not isinstance(nav_link, str):
                continue
            pos = self._get(item, ["position", "rank"], default=None)
            if pos is not None:
                try:
                    pos = int(pos)
                except (TypeError, ValueError):
                    pos = None
            candidates.append((nav_link, pos))
        return candidates

    def _maybe_sample_negatives(
        self,
        session: Dict[str, Any],
        positive_links: Set[str],
        existing_negative_links: Set[str],
    ) -> List[Tuple[str, Optional[int], str]]:
        """
        If a candidate pool is provided, sample additional negatives using 70/30 hard/random mixing:
          - Hard pool: candidates with position in [hard_min_rank, hard_max_rank]
          - Random pool: all remaining candidates (excluding positives and already-existing negatives)

        Returns:
          List of tuples (nav_link, position, source) where source ∈ {"sampled_hard", "sampled_random"}.
        """
        candidates = self._iter_candidates(session)
        if not candidates:
            return []

        # Deduplicate by link, keep the best (lowest) position if duplicates present
        best_pos: Dict[str, Optional[int]] = {}
        for link, pos in candidates:
            if link not in best_pos or (isinstance(pos, int) and (best_pos[link] is None or pos < best_pos[link])):
                best_pos[link] = pos

        # Filter out positives and existing negatives
        pool_items: List[Tuple[str, Optional[int]]] = [
            (link, pos)
            for link, pos in best_pos.items()
            if link not in positive_links and link not in existing_negative_links
        ]
        if not pool_items:
            return []

        # Partition into hard vs random pools based on position bounds
        hard_pool: List[Tuple[str, Optional[int]]] = []
        rand_pool: List[Tuple[str, Optional[int]]] = []
        for link, pos in pool_items:
            if isinstance(pos, int) and self.hard_min_rank <= pos <= self.hard_max_rank:
                hard_pool.append((link, pos))
            else:
                rand_pool.append((link, pos))

        # Determine how many negatives to sample
        num_pos = max(1, len(positive_links))  # at least scale to 1 if positives exist
        target_k = max(0, int(round(self.negatives_per_positive * num_pos)))

        if target_k == 0:
            return []

        hard_k = min(len(hard_pool), int(round(self.hard_fraction * target_k)))
        rand_k = max(0, target_k - hard_k)

        rng = random.Random()
        sampled: List[Tuple[str, Optional[int], str]] = []

        if hard_pool and hard_k > 0:
            hard_sample = rng.sample(hard_pool, k=hard_k) if len(hard_pool) >= hard_k else hard_pool
            sampled.extend([(link, pos, "sampled_hard") for link, pos in hard_sample])

        # Build a uniform pool across all remaining items (both pools) for the random share
        remaining_pool = [(l, p) for (l, p) in pool_items if l not in {x[0] for x in sampled}]
        if remaining_pool and rand_k > 0:
            rand_sample = rng.sample(remaining_pool, k=rand_k) if len(remaining_pool) >= rand_k else remaining_pool
            sampled.extend([(link, pos, "sampled_random") for link, pos in rand_sample])

        return sampled


# Optional: quick usage example (remove or adapt as needed)
if __name__ == "__main__":

    print(f"Generating {NUM_SAMPLES} log samples...")

    # Create a pool of user IDs so that users can have multiple sessions
    num_unique_users = NUM_SAMPLES // 4  # e.g., 2500 unique users for 10k sessions
    user_id_pool = [f"u{random.randint(1000, 9999)}" for _ in range(num_unique_users)]

    all_sessions = [generate_session(i, user_id_pool) for i in range(NUM_SAMPLES)]
    print(all_sessions[1])

    """
    {
        "query": "best python courses for beginners",
        "user_id": "u8884",
        "session_id": "s2",
        "timestamp": 1758922149,
        "clicks": [
            {
                "link": "https://bakers-delight.com/products/best-python-courses-for-beginners-1",
                "rank": 1,
                "click_timestamp": 1758922160,
                "dwell_seconds": 27.02
            },
            {
                "link": "https://bakers-delight.com/listings/best-python-courses-for-beginners-2",
                "rank": 2,
                "timestamp": 1758922171,
                "dwell_seconds": 146.25
            }
        ],
        "impressions": [
            {
                "link": "https://bakers-delight.com/listings/best-python-courses-for-beginners-3",
                "position": 3,
                "timestamp": 1758922149,
                "skip_type": "user_scrolled_past"
            },
            {
                "link": "https://bakers-delight.com/guides/best-python-courses-for-beginners-4",
                "position": 4,
                "impression_time": 1758922149,
                "skip_type": "impression"
            }
        ]
    }
    """

    # --- Output ---

    processor = ClickLogProcessor(
        # Example: keep defaults, which enable the power-law propensity and mixed negative sampling
        # power_law_gamma=0.5,
        # enable_mixed_negative_sampling=True,
        # negatives_per_positive=3,
        # hard_fraction=0.7,
        # hard_min_rank=101,
        # hard_max_rank=1000,
        # snips_include_n_factor=True,
    )
    dataset = processor.build_dataset(all_sessions)
    # Or write as a single JSON array
    processor.write_json(dataset, "data/personalization/training_dataset.json", jsonl=False, indent=2)