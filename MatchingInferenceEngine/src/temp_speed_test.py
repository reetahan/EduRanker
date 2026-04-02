import time
import numpy as np
from mallows import mallows_insertion_sampling, slow_mallows_insertion_sampling, _build_position_prob_cache, _build_numba_prob_cache

def run_benchmark():
    # --- Parameters ---
    n_schools = 200
    n_students = 5000  # A small batch to test
    k = 10
    phi = 0.8
    
    # Dummy central ranking (e.g., [0, 1, 2, ... 199])
    central_ranking = np.arange(n_schools, dtype=np.int32)
    rng = np.random.default_rng(42)

    print(f"--- Benchmarking {n_students} students (Schools: {n_schools}, Top-K: {k}, Phi: {phi}) ---")

    # --- 1. Pre-build Caches (Not timed, as this happens once per district) ---
    _, old_cache = _build_position_prob_cache(n_schools, phi)
    new_cache = _build_numba_prob_cache(n_schools, phi)

    # --- 2. WARMUP RUN ---
    # We call the new function once so Numba compiles it to C before we start the timer
    _ = mallows_insertion_sampling(central_ranking, phi, rng=rng, cum_cache_2d=new_cache, k_ranking_length=k)

    # --- 3. Time the OLD Method ---
    print("Running Old Method...")
    start_old = time.perf_counter()
    for _ in range(n_students):
        _ = slow_mallows_insertion_sampling(central_ranking, phi, rng=rng, position_prob_cache=old_cache)
    time_old = time.perf_counter() - start_old

    # --- 4. Time the NEW Method ---
    print("Running New Method...")
    start_new = time.perf_counter()
    for _ in range(n_students):
        _ = mallows_insertion_sampling(central_ranking, phi, rng=rng, cum_cache_2d=new_cache, k_ranking_length=k)
    time_new = time.perf_counter() - start_new

    # --- 5. Results ---
    print("\n--- RESULTS ---")
    print(f"Old Method: {time_old:.4f} seconds")
    print(f"New Method: {time_new:.4f} seconds")
    
    speedup = time_old / time_new if time_new > 0 else float('inf')
    print(f"Speedup: {speedup:.1f}x faster!")

# Run it!
if __name__ == "__main__":
    run_benchmark()