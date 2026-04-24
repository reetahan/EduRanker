import numpy as np
import pandas as pd
from util import log_and_print
from numba import njit

def compute_aggregates(student_rankings, matches, district_assignments, schools_list):
    n_students = len(student_rankings)
    n_schools = len(schools_list)
    districts = np.unique(district_assignments)
    n_districts = len(districts)
    
    district_to_idx = {d: i for i, d in enumerate(districts)}
    school_to_idx = {s: i for i, s in enumerate(schools_list)}
    
    total_app = np.zeros((n_districts, n_schools))
    true_app = np.zeros((n_districts, n_schools))
    match_stats = np.zeros((n_districts, 4))
    filled = np.zeros(n_schools)
    
    for student_id in range(n_students):
        district_idx = district_to_idx[district_assignments[student_id]]
        ranking = student_rankings[student_id]
        if isinstance(ranking, np.ndarray):
            ranking = ranking.tolist()
        match = matches[student_id]
        
        for school in ranking:
            school_idx = school_to_idx[school]
            total_app[district_idx, school_idx] += 1
        
        if match != '-1':
            match = str(match)  
            match_school_idx = school_to_idx[match]
            try:
                match_position = ranking.index(match)  
            except ValueError:
                log_and_print(f"Warning: Student matched to {match} not in ranking: {ranking}")
                continue
            
            for school in ranking[match_position:]:
                school_idx = school_to_idx[school]
                true_app[district_idx, school_idx] += 1
            
            filled[match_school_idx] += 1
            
            if match_position < 3:
                match_stats[district_idx, 0] += 1
            if match_position < 5:
                match_stats[district_idx, 1] += 1
            if match_position < 10:
                match_stats[district_idx, 2] += 1
        else:
            for school in ranking:
                school_idx = school_to_idx[school]
                true_app[district_idx, school_idx] += 1
            
            match_stats[district_idx, 3] += 1
    

    for d in range(n_districts):
        # Count total students in this district
        district_total = np.sum(district_assignments == districts[d])
        
        if district_total > 0:
            match_stats[d, :] = (match_stats[d, :] / district_total) * 100
    
    return {
        'total_app': total_app,
        'true_app': true_app,
        'match_stats': match_stats,
        'filled': filled
    }

def gale_shapley_per_school_numba_wrapper(student_rankings, school_lottery_numbers, school_capacities):
    print(f"Invoked gale_shapley_per_school_numba_wrapper")
    max_len = max(len(r) for r in student_rankings)
    n = len(student_rankings)
    padded = np.full((n, max_len), -1, dtype=np.int32)
    lengths = np.empty(n, dtype=np.int32)
    for i, r in enumerate(student_rankings):
        L = len(r)
        padded[i, :L] = r
        lengths[i] = L
    return gale_shapley_per_school_numba(
        padded, lengths,
        school_lottery_numbers.astype(np.float64),
        np.asarray(school_capacities, dtype=np.int32),
    )

def gale_shapley(student_rankings, student_lottery_numbers, school_capacities):
    n_students = len(student_rankings)
    n_schools = len(school_capacities)
    
    student_order = np.argsort(student_lottery_numbers)
    
    matches = np.full(n_students, -1)
    school_filled = np.zeros(n_schools, dtype=int)
    
    for student in student_order:
        for school in student_rankings[student]:
            if school_filled[school] < school_capacities[school]:
                school_filled[school] += 1
                matches[student] = school
                break
    
    return matches

@njit
def gale_shapley_per_school_numba(
    padded_rankings,
    ranking_lengths,
    school_lottery,
    school_capacities,
):
    n_students = padded_rankings.shape[0]
    n_schools = school_capacities.shape[0]
    max_cap = np.max(school_capacities)

    matches = np.full(n_students, -1, dtype=np.int32)
    next_proposal = np.zeros(n_students, dtype=np.int32)

    school_held = np.full((n_schools, max_cap), -1, dtype=np.int32)
    school_count = np.zeros(n_schools, dtype=np.int32)
    # Cache the worst lottery number among held students per school
    school_worst_lottery = np.full(n_schools, -1.0, dtype=np.float64)

    stack = np.empty(n_students, dtype=np.int32)
    for i in range(n_students):
        stack[i] = i
    stack_size = n_students

    while stack_size > 0:
        stack_size -= 1
        student = stack[stack_size]

        if next_proposal[student] >= ranking_lengths[student]:
            continue

        school = padded_rankings[student, next_proposal[student]]
        next_proposal[student] += 1
        s_lot = school_lottery[school, student]
        cnt = school_count[school]
        cap = school_capacities[school]

        if cnt < cap:
            # Room available — accept directly
            school_held[school, cnt] = student
            school_count[school] = cnt + 1
            matches[student] = school
            if s_lot > school_worst_lottery[school]:
                school_worst_lottery[school] = s_lot
        elif s_lot < school_worst_lottery[school]:
            # Proposer beats the cached worst — find and replace
            worst_idx = 0
            worst_lot = school_lottery[school, school_held[school, 0]]
            for j in range(1, cap):
                lot_j = school_lottery[school, school_held[school, j]]
                if lot_j > worst_lot:
                    worst_lot = lot_j
                    worst_idx = j

            rejected = school_held[school, worst_idx]
            school_held[school, worst_idx] = student
            matches[rejected] = -1
            matches[student] = school

            # Recompute cached worst
            new_worst_lot = -1.0
            for j in range(cap):
                lot_j = school_lottery[school, school_held[school, j]]
                if lot_j > new_worst_lot:
                    new_worst_lot = lot_j
            school_worst_lottery[school] = new_worst_lot

            stack[stack_size] = rejected
            stack_size += 1
        else:
            # Proposer is worse than everyone held — instant reject, O(1)
            stack[stack_size] = student
            stack_size += 1

    return matches

def gale_shapley_per_school(student_rankings, school_lottery_numbers, school_capacities):
    n_students = len(student_rankings)
    
    free = set(range(n_students))
    next_proposal = [0] * n_students
    matches = np.full(n_students, -1)
    school_held = [[] for _ in range(len(school_capacities))]
    
    while free:
        student = free.pop()
        if next_proposal[student] >= len(student_rankings[student]):
            continue
        school = student_rankings[student][next_proposal[student]]
        next_proposal[student] += 1
        
        school_held[school].append(student)
        if len(school_held[school]) > school_capacities[school]:
            rejected = max(school_held[school],
                          key=lambda s: school_lottery_numbers[school, s])
            school_held[school].remove(rejected)
            matches[rejected] = -1
            free.add(rejected)
        
        if student in school_held[school]:
            matches[student] = school
    
    return matches

def boston_algorithm(student_rankings, student_lottery_numbers, school_capacities):
    n_students = len(student_rankings)
    n_schools = len(school_capacities)
    
    student_order = np.argsort(student_lottery_numbers)
    
    matches = np.full(n_students, -1)
    school_assignments = [[] for _ in range(n_schools)]
    
    max_rank_length = max(len(ranking) for ranking in student_rankings)
    
    for rank in range(max_rank_length):
        applications = [[] for _ in range(n_schools)]
        
        for student in student_order:
            if matches[student] == -1 and rank < len(student_rankings[student]):
                school = student_rankings[student][rank]
                applications[school].append(student)
        
        for school in range(n_schools):
            remaining_seats = school_capacities[school] - len(school_assignments[school])
            
            if remaining_seats > 0:
                accepted = applications[school][:remaining_seats]
                
                for student in accepted:
                    school_assignments[school].append(student)
                    matches[student] = school

def boston_algorithm_per_school(student_rankings, school_lottery_numbers, school_capacities):
    n_students = len(student_rankings)
    
    matches = np.full(n_students, -1)
    school_assignments = [[] for _ in school_capacities]
    
    max_rank_length = max(len(ranking) for ranking in student_rankings)
    
    for rank in range(max_rank_length):
        applications = [[] for _ in school_capacities]
        
        for student in range(n_students):
            if matches[student] == -1 and rank < len(student_rankings[student]):
                school = student_rankings[student][rank]
                applications[school].append(student)
        
        for school, capacity in enumerate(school_capacities):
            remaining_seats = capacity - len(school_assignments[school])
            
            if remaining_seats > 0:
                priorities = school_lottery_numbers[school]
                accepted = sorted(applications[school],
                                  key=lambda s: priorities[s])[:remaining_seats]
                
                school_assignments[school].extend(accepted)
                for student in accepted:
                    matches[student] = school
    
    return matches

def top_trading_cycles(student_rankings, school_rankings, school_capacities):
    n_students = len(student_rankings)
    n_schools = len(school_capacities)

    matches = [-1] * n_students
    remaining_capacities = list(school_capacities)

    # Pointeur vers la prochaine école encore disponible pour chaque élève
    next_school_idx = [0] * n_students

    # Pointeur vers le prochain élève encore actif / non affecté pour chaque école
    next_student_idx = [0] * n_schools

    while True:
        active_students = []
        active = [False] * n_students
        student_points = [-1] * n_students

        # Chaque élève non affecté pointe vers sa meilleure école encore disponible
        for student in range(n_students):
            if matches[student] != -1:
                continue

            prefs = student_rankings[student]
            i = next_school_idx[student]

            while i < len(prefs) and remaining_capacities[prefs[i]] == 0:
                i += 1

            next_school_idx[student] = i

            if i < len(prefs):
                school = prefs[i]
                student_points[student] = school
                active_students.append(student)
                active[student] = True

        if not active_students:
            break

        school_points = [-1] * n_schools

        # Chaque école pointe vers son meilleur élève encore actif et non affecté
        for school in range(n_schools):
            if remaining_capacities[school] <= 0:
                continue

            ranking = school_rankings[school]
            i = next_student_idx[school]

            while i < len(ranking):
                student = ranking[i]
                if matches[student] == -1 and active[student]:
                    break
                i += 1

            next_student_idx[school] = i

            if i < len(ranking):
                school_points[school] = ranking[i]

        # Détection de tous les cycles disjoints du tour
        processed = set()
        cycle_students = []

        for start_student in active_students:
            start_node = ("student", start_student)

            if start_node in processed:
                continue

            visited = {}
            path = []
            node = start_node

            while node is not None and node not in processed and node not in visited:
                visited[node] = len(path)
                path.append(node)

                node_type, idx = node

                if node_type == "student":
                    school = student_points[idx]
                    node = None if school == -1 else ("school", school)
                else:
                    student = school_points[idx]
                    node = None if student == -1 else ("student", student)

            # Si on retombe sur un noeud déjà vu dans ce parcours, on a trouvé un cycle
            if node is not None and node in visited:
                cycle = path[visited[node]:]
                for node_type, idx in cycle:
                    if node_type == "student":
                        cycle_students.append(idx)

            processed.update(path)

        if not cycle_students:
            break

        # Exécution simultanée de tous les cycles disjoints du tour
        for student in cycle_students:
            matches[student] = student_points[student]

        for student in cycle_students:
            school = student_points[student]
            remaining_capacities[school] -= 1

    return matches


