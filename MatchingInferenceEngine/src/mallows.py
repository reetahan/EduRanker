import numpy as np
from numba import njit
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from util import log_and_print

def _build_numba_prob_cache(max_positions, phi):
    """
    Builds a 2D array for fast C-level CDF lookups.
    """
    cum_cache_2d = np.ones((max_positions, max_positions), dtype=np.float64)
    for i in range(max_positions):
        num_positions = i + 1
        probs = np.array([phi ** (num_positions - 1 - j) for j in range(num_positions)])
        cum_cache_2d[i, :num_positions] = np.cumsum(probs / probs.sum())
    return cum_cache_2d

@njit
def _mallows_top_k_numba(central_ranking, u_draws, cum_cache_2d, k):
    """Build only the top-k of a Mallows permutation."""
    n = len(central_ranking)
    if n == 0:
        return np.empty(0, dtype=central_ranking.dtype)
    if k > n:
        k = n
    
    top = np.empty(k, dtype=central_ranking.dtype)
    size = 0
    
    for i in range(n):
        num_positions = i + 1
        cdf = cum_cache_2d[i, :num_positions]
        pos = np.searchsorted(cdf, u_draws[i])
        
        if pos >= k:
            continue
        
        if size < k:
            # Shift right to make room
            for j in range(size, pos, -1):
                top[j] = top[j - 1]
            top[pos] = central_ranking[i]
            size += 1
        else:
            # Full: shift right, drop last
            for j in range(k - 1, pos, -1):
                top[j] = top[j - 1]
            top[pos] = central_ranking[i]
    
    return top[:size]

def mallows_insertion_sampling(central_ranking, phi, rng=None, cum_cache_2d=None, k_ranking_length=10):
    n = len(central_ranking)
    chooser = rng if rng is not None else np.random
    u_draws = chooser.random(n)
    if cum_cache_2d is None:
        cum_cache_2d = _build_numba_prob_cache(n, phi)
    return _mallows_top_k_numba(np.array(central_ranking), u_draws, cum_cache_2d, k_ranking_length)

def slow_mallows_insertion_sampling(central_ranking, phi, rng=None, position_prob_cache=None):
    n = len(central_ranking)
    ranking = []
    chooser = rng if rng is not None else np.random
    
    for i in range(n):
        item = central_ranking[i]
        
        if len(ranking) == 0:
            ranking.append(item)
        else:
            positions = len(ranking) + 1
            if position_prob_cache is not None:
                probs = position_prob_cache[positions]
            else:
                probs = np.array([phi ** (positions - 1 - j) for j in range(positions)])
                probs = probs / probs.sum()
            pos = chooser.choice(positions, p=probs)
            ranking.insert(pos, item)
    
    return np.array(ranking)

def compute_sigma_cutoff(phi, k_ranking_length=10, min_prob=1e-5):
    if phi >= 0.99:
        return None
    cutoff = int(np.ceil(1 + np.log(min_prob / (1 - phi)) / np.log(phi)))
    return max(cutoff, k_ranking_length)

def _build_position_prob_cache(max_positions, phi):
    cache = {1: np.array([1.0])}
    cumcache = {1: np.array([1.0])}
    for positions in range(2, max_positions + 1):
        probs = np.array([phi ** (positions - 1 - j) for j in range(positions)])
        probs = probs / probs.sum()
        cache[positions] = probs
        cumcache[positions] = np.cumsum(probs)
    return cache, cumcache


def _sample_students_chunk(sigma_indices, phis, component_indices, seed, k_ranking_length=10):
    rng = np.random.default_rng(seed)
    cum_caches = {phi_idx: _build_numba_prob_cache(len(sigma_indices), phis[phi_idx]) for phi_idx in range(len(phis))}
    rankings = []
    for k in component_indices:
        rankings.append(
            mallows_insertion_sampling(
                sigma_indices,
                phis[k],
                rng=rng,
                cum_cache_2d=cum_caches[k],
                k_ranking_length=k_ranking_length,
            )
        )
    return rankings


def sample_students_global_mixture(
    params,
    district,
    n_students,
    n_jobs=1,
    chunk_size=1000,
    random_seed=None,
    log_progress=False,
    progress_every=5000,
    log_file=None,
):
    """
    Sample students from global mixture with district-specific sigma.

    Args:
        params: Parameter dictionary with global and district-specific settings.
        district: District identifier.
        n_students: Number of synthetic students to generate.
        n_jobs: Number of processes to use. Use 1 for sequential execution.
        chunk_size: Number of students per parallel chunk.
        random_seed: Optional integer seed for reproducibility.
        log_progress: Whether to print progress while generating students.
        progress_every: Print progress every N completed students.
        log_file: Optional path used by log_and_print for persistent logging.
    """
    
    # Global parameters
    phis = params['global_phis']
    weights = params['global_weights']
    K = len(phis)
    
    # District-specific parameters
    sigma_d = params['districts'][district]['central_ranking']
    schools = params['districts'][district]['schools']
    school_to_idx = {s: i for i, s in enumerate(schools)}
    sigma_indices = np.array([school_to_idx[s] for s in sigma_d])
    
    rng = np.random.default_rng(random_seed)
    component_indices = rng.choice(K, size=n_students, p=weights)
    progress_every = max(1, int(progress_every))

    def maybe_log(completed):
        if log_progress and (completed == n_students or completed % progress_every == 0):
            pct = 100.0 * completed / n_students if n_students else 100.0
            log_and_print(
                f"[sample_students_global_mixture] Completed {completed}/{n_students} students ({pct:.1f}%)",
                log_file=log_file,
            )

    if n_jobs <= 1 or n_students <= 1:
        cum_caches = {phi_idx: _build_numba_prob_cache(len(sigma_indices), phis[phi_idx]) for phi_idx in range(K)}
        rankings = []
        for i, k in enumerate(component_indices, start=1):
            ranking = mallows_insertion_sampling(
                sigma_indices,
                phis[k],
                rng=rng,
                cum_cache_2d=cum_caches[k],
            )
            rankings.append(ranking)
            maybe_log(i)
        return rankings

    max_workers = min(max(1, int(n_jobs)), os.cpu_count() or 1, n_students)
    chunk_size = max(1, int(chunk_size))
    chunks = [component_indices[start:start + chunk_size] for start in range(0, n_students, chunk_size)]
    child_seeds = np.random.SeedSequence(random_seed).spawn(len(chunks))
    
    rankings = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk_meta = {}
        for chunk_idx, (chunk, seed) in enumerate(zip(chunks, child_seeds)):
            future = executor.submit(
                _sample_students_chunk,
                sigma_indices,
                phis,
                chunk,
                seed,
            )
            future_to_chunk_meta[future] = (chunk_idx, len(chunk))

        completed = 0
        rankings_by_chunk = [None] * len(chunks)
        for future in as_completed(future_to_chunk_meta):
            chunk_idx, chunk_len = future_to_chunk_meta[future]
            rankings_by_chunk[chunk_idx] = future.result()
            completed += chunk_len
            maybe_log(completed)

        for chunk_rankings in rankings_by_chunk:
            rankings.extend(chunk_rankings)

    return rankings