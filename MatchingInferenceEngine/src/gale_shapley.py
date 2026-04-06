import numpy as np
import pandas as pd
from analysis import log_and_print

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