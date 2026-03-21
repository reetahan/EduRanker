import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from analysis import log_and_print

def mallows_insertion_sampling(central_ranking, phi, rng=None, position_prob_cache=None):
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

def _build_position_prob_cache(max_positions, phi):
    cache = {1: np.array([1.0])}
    for positions in range(2, max_positions + 1):
        probs = np.array([phi ** (positions - 1 - j) for j in range(positions)])
        cache[positions] = probs / probs.sum()
    return cache


def _sample_students_chunk(sigma_indices, phis, component_indices, seed):
    rng = np.random.default_rng(seed)
    prob_caches = {phi_idx: _build_position_prob_cache(len(sigma_indices), phis[phi_idx]) for phi_idx in range(len(phis))}
    rankings = []
    for k in component_indices:
        rankings.append(
            mallows_insertion_sampling(
                sigma_indices,
                phis[k],
                rng=rng,
                position_prob_cache=prob_caches[k],
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
        prob_caches = {phi_idx: _build_position_prob_cache(len(sigma_indices), phis[phi_idx]) for phi_idx in range(K)}
        rankings = []
        for i, k in enumerate(component_indices, start=1):
            ranking = mallows_insertion_sampling(
                sigma_indices,
                phis[k],
                rng=rng,
                position_prob_cache=prob_caches[k],
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