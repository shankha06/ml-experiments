import numpy as np
from typing import List, Tuple, Optional

def get_adaptive_k(
    scores: List[float], 
    min_k: int = 1, 
    max_k: Optional[int] = None
) -> Tuple[int, List[float]]:
    
    # 1. Safety Checks
    if not scores:
        return 0, []
    
    scores_arr = np.array(scores)
    
    if max_k and len(scores_arr) > max_k:
        scores_arr = scores_arr[:max_k]
    
    if len(scores_arr) <= min_k:
        return len(scores_arr), scores_arr.tolist()

    # 2. Calculate Standard Gaps
    # Gap[i] = Score[i] - Score[i+1]
    raw_gaps = scores_arr[:-1] - scores_arr[1:]
    
    # 3. Apply Weighting (The Fix)
    # We weight the gap by the score itself. 
    # A 0.2 drop at 0.9 confidence is more significant than a 0.3 drop at 0.5 confidence.
    weighted_gaps = raw_gaps * scores_arr[:-1]

    # 4. Identify the Elbow
    # Consider only gaps starting from min_k - 1
    valid_weighted_gaps = weighted_gaps[min_k-1:]
    
    if len(valid_weighted_gaps) == 0:
        return min_k, scores_arr[:min_k].tolist()

    # Find the index of the largest weighted gap
    elbow_index = np.argmax(valid_weighted_gaps) + (min_k - 1)
    
    # 5. Cut off results
    optimal_k = elbow_index + 1
    
    return optimal_k, scores_arr[:optimal_k].tolist()

# --- Verification ---
# Your scenario:
scores = [0.95, 0.94, 0.92, 0.88, 0.65, 0.64, 0.60, 0.55, 0.20, 0.15]

k, top_scores = get_adaptive_k(scores, min_k=1)

print(f"Adaptive k: {k}")
print(f"Top Scores: {top_scores}")