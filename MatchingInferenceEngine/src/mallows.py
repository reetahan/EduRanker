import numpy as np


def mallows_insertion_sampling(central_ranking, phi):
    n = len(central_ranking)
    ranking = []
    
    for i in range(n):
        item = central_ranking[i]
        
        if len(ranking) == 0:
            ranking.append(item)
        else:
            positions = len(ranking) + 1
            probs = np.array([phi ** (positions - 1 - j) for j in range(positions)])
            probs = probs / probs.sum()
            pos = np.random.choice(positions, p=probs)
            ranking.insert(pos, item)
    
    return np.array(ranking)

def sample_students_global_mixture(params, district, n_students):
    """
    Sample students from global mixture with district-specific sigma
    """
    
    # Global parameters
    phis = params['global_phis']
    weights = params['global_weights']
    K = len(phis)
    
    # District-specific parameters
    sigma_d = params['districts'][district]['central_ranking']
    schools = params['districts'][district]['schools']
    
    rankings = []
    
    for _ in range(n_students):
        # Choose type from global mixture
        k = np.random.choice(K, p=weights)
        
        # Sample ranking from Mallows(σ_d, φ_k)
        school_to_idx = {s: i for i, s in enumerate(schools)}
        sigma_indices = np.array([school_to_idx[s] for s in sigma_d])
        
        ranking = mallows_insertion_sampling(sigma_indices, phis[k])
        rankings.append(ranking)
    
    return rankings