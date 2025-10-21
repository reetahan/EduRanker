import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.special import comb
from tqdm import tqdm

def rejection_probability(ell, pi_r, c):
    """
    Compute P(Binomial(ell-1, pi_r) >= c)
    """
    if ell <= 1:
        return 0.0
    
    # Use survival function for numerical stability
    return 1 - binom.cdf(c-1, ell-1, pi_r)

def prob_unmatched_weighted(ell, pi_values, c, k):
    """
    Compute P(unmatched | ell) with weighted average
    """
    rejection_probs = np.array([rejection_probability(ell, pi_r, c) 
                                 for pi_r in pi_values])
    
    # Weight by probability of including each school
    weighted_avg_rejection = np.sum(pi_values * rejection_probs) / np.sum(pi_values)
    
    return weighted_avg_rejection ** k

def sample_mallows_ranking(m, phi):
    """Sample a full ranking from Mallows(1,2,...,m, phi)"""
    remaining = list(range(1, m+1))
    ranking = []
    
    for pos in range(m):
        # Compute weights: phi^(rank-1) for each remaining school
        weights = np.array([phi**(remaining.index(r)) for r in remaining])
        probs = weights / weights.sum()
        
        # Sample
        chosen_idx = np.random.choice(len(remaining), p=probs)
        ranking.append(remaining[chosen_idx])
        remaining.pop(chosen_idx)
    
    return ranking

def compute_pi_r(phi, k, m, n_samples=10000):
    """
    Compute pi_r(phi, k) for all r via Monte Carlo

    Returns: array of length m with pi_1, pi_2, ..., pi_m
    """
    counts = np.zeros(m)

    for _ in tqdm(range(n_samples), desc=f"Computing π_r (φ={phi:.1f}, k={k})", leave=False):
        ranking = sample_mallows_ranking(m, phi)
        top_k = ranking[:k]
        for school in top_k:
            counts[school-1] += 1

    return counts / n_samples


def advise_parents(ell, n, m, c, phi, k_max=20):
    """
    Generate personalized advice for parents
    
    Args:
        ell: Student's lottery number (1 = best)
        n: Total students
        m: Total schools
        c: Capacity per school
        phi: Estimated preference correlation (0 = consensus, 1 = diverse)
        k_max: Maximum schools they can rank
    
    Returns:
        Dictionary with advice
    """
    
    advice = {}
    
    # Compute pi_r values
    pi_values = compute_pi_r(phi, k_max, m, n_samples=10000)
    
    # Compute P(unmatched) for different k values
    k_range = range(1, k_max+1)
    probs = [prob_unmatched_weighted(ell, pi_values[:k], c, k) for k in k_range]
    
    advice['lottery_percentile'] = 100 * ell / n
    advice['prob_unmatched_by_k'] = list(zip(k_range, probs))
    
    # Find minimum k to achieve 95% match probability
    target_prob = 0.05
    k_recommended = next((k for k, p in zip(k_range, probs) if p < target_prob), k_max)
    advice['k_recommended'] = k_recommended
    advice['prob_at_recommended_k'] = probs[k_recommended-1]
    
    # Categorize lottery quality
    if ell <= 0.2 * n:
        category = "Excellent"
        strategy = "You have a top lottery number. You can safely include your true top choices. Even popular schools are realistic options for you."
    elif ell <= 0.5 * n:
        category = "Good"
        strategy = "You have a decent lottery number. Include a mix of reach schools (popular) and match schools (moderate demand)."
    elif ell <= 0.8 * n:
        category = "Moderate"
        strategy = "Your lottery number is in the middle range. Focus on match and safety schools, but still include 1-2 reach schools."
    else:
        category = "Challenging"
        strategy = "Your lottery number is toward the lower end. Maximize your list length and focus on safety schools (less popular but good quality)."
    
    advice['lottery_category'] = category
    advice['general_strategy'] = strategy
    
    # School distribution recommendation
    if ell <= 0.3 * n:
        distribution = {"reach": 0.4, "match": 0.4, "safety": 0.2}
    elif ell <= 0.7 * n:
        distribution = {"reach": 0.2, "match": 0.5, "safety": 0.3}
    else:
        distribution = {"reach": 0.1, "match": 0.3, "safety": 0.6}
    
    advice['school_distribution'] = distribution
    
    # Specific numbers
    reach_count = int(k_recommended * distribution['reach'])
    match_count = int(k_recommended * distribution['match'])
    safety_count = k_recommended - reach_count - match_count
    
    advice['recommended_counts'] = {
        'reach': reach_count,
        'match': match_count,
        'safety': safety_count
    }
    
    return advice

def print_parent_advice(ell, n=1000, m=50, c=25, phi=0.5, k_max=20):
    """Print formatted advice for parents"""
    
    advice = advise_parents(ell, n, m, c, phi, k_max)
    
    print("=" * 60)
    print("PERSONALIZED SCHOOL APPLICATION ADVICE")
    print("=" * 60)
    print(f"\nYour lottery number: {ell} (top {advice['lottery_percentile']:.1f}%)")
    print(f"Lottery quality: {advice['lottery_category']}")
    print(f"\n{advice['general_strategy']}")
    
    print(f"\n📊 RECOMMENDED LIST LENGTH: {advice['k_recommended']} schools")
    print(f"   → Your match probability: {100*(1-advice['prob_at_recommended_k']):.1f}%")
    
    print(f"\n🎯 SCHOOL MIX RECOMMENDATION:")
    rec = advice['recommended_counts']
    print(f"   • {rec['reach']} REACH schools (top-tier, competitive)")
    print(f"   • {rec['match']} MATCH schools (realistic good options)")
    print(f"   • {rec['safety']} SAFETY schools (high confidence)")
    
    print(f"\n📈 PROBABILITY OF MATCHING BY LIST LENGTH:")
    for k, p in advice['prob_unmatched_by_k'][:15]:  # Show first 15
        match_prob = 100 * (1 - p)
        bar = "█" * int(match_prob / 5)
        print(f"   k={k:2d}: {match_prob:5.1f}% {bar}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Market has φ={phi:.2f} (0=consensus, 1=diverse preferences)")
    
    if phi < 0.3:
        print(f"   • Strong consensus exists - popular schools very competitive")
        print(f"   • Consider looking beyond the 'obvious' top choices")
    elif phi > 0.7:
        print(f"   • Diverse preferences - opportunities across many schools")
        print(f"   • Research schools that match YOUR specific values")
    else:
        print(f"   • Moderate consensus - balance reach and safety carefully")
    
    print(f"\n   • Total capacity: {m*c} seats for {n} students")
    surplus = (m*c - n) / n * 100
    print(f"   • System has {surplus:.1f}% surplus capacity")
    
    if surplus > 20:
        print(f"   • Good news: plenty of seats available overall")
    elif surplus > 0:
        print(f"   • Moderate surplus - competition exists but manageable")
    else:
        print(f"   • ⚠️  Not enough total seats - maximize your list length!")
    
    print("\n" + "=" * 60)

# Example usage
print_parent_advice(ell=150, n=1000, m=50, c=25, phi=0.5, k_max=20)
print("\n" + "="*60 + "\n")
print_parent_advice(ell=850, n=1000, m=50, c=25, phi=0.5, k_max=20)