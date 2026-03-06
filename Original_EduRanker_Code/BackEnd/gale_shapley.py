from heapq import heappop, heappush, heappushpop
from pathlib import Path
import numpy as np

class Student:
	def __init__(self, identifier, ranking, lottery_number):
		self.identifier = identifier
		self.ranking = ranking
		self.lottery_number = lottery_number
		self.last_proposal = -1
		self.matched = False

	def propose(self):
		"""
		Moves down the preference list by one step, then returns the DBN of the next school the student will propose to
		(namely, the highest ranked school the student is yet to propose to).
		"""

		self.last_proposal += 1
		return self.ranking[self.last_proposal]

	def update_match_status(self, matched: bool):
		self.matched = matched

	def can_propose(self):
		"""
		Returns `True` if and only if the student can propose to a new school, namely if the student does not have a
		temporary allocation and still has some schools they haven't proposed to in their ranking.
		"""
		return not self.matched and self.last_proposal < len(self.ranking)-1

	def get_result(self):
		"""
		Returns the DBN of the school the student was matched with and the position of the school in the preference
		list (0-based), or `(None, None)` if the student was unmatched.
		"""
		if not self.matched:
			return None, None
		else:
			return self.ranking[self.last_proposal], self.last_proposal

class School:
	def __init__(self, dbn, ranking, total_seats):
		self.dbn = dbn

		self.ranking_list = ranking
		self.ranking = {student_id: rank for rank, student_id in enumerate(ranking)}

		self.non_priority_list = []
		self.total_seats = total_seats
		self.applications_received = 0

	def check_match(self, student_id) -> bool:
		"""
		Returns `True` if a proposal received by the specified student will be accepted, `False` otherwise.

		Parameters:
		- student_id: the identifier of the proposing student.
		"""

		self.applications_received += 1

		# If the school has residual capacity, the proposal will be accepted regardless of rank.
		if len(self.non_priority_list) < self.total_seats:
			return True

		# If the school has provisionally accepted at least one student that ranks worse than the proposing student, then
		# the proposal will be accepted, otherwise it will be rejected.
		rank = self.ranking.get(student_id)
		return rank < -self.non_priority_list[0][0]

	def handle_proposal(self, student_id) -> Student | None:
		"""
		Handles the proposal received by the specified student and returns the identifier of the student that is rejected
		as a result, if applicable. If no student is rejected as a consequence of this proposal, returns `None`.

		Parameters:
		- student_id: the identifier of the proposing student.

		Returns: the identifier of the rejected student, or `None` if no student was rejected.
		"""

		rank = self.ranking.get(student_id)
		item = (-rank, student_id)  # student_rank is negated since heapq implements a min heap

		# If the school has residual capacity, accept the proposing student and reject no one.
		if len(self.non_priority_list) < self.total_seats:
			heappush(self.non_priority_list, item)
			return None

		# If all seats have been filled, accept the proposing student on a provisional basis, then reject the lowest-ranking
		# among the provisionally accepted students.
		reject = heappushpop(self.non_priority_list, item)
		return reject[1]

	def prefers_to_matches(self, student_id):
		"""
		Returns `True` if the school prefers the student with the specified ID to at least one of the students it has
		accepted, namely if the student in question is ranked higher than the lowest-ranked accepted student.
		"""
		rank = self.ranking.get(student_id)
		worst_match_rank = self.non_priority_list[0][0]
		return rank < -worst_match_rank

	def get_result(self) -> tuple[list, int, int, int]:
		"""
		Returns the students matched with the school, their count, the number of available spots, and the number of applicants.
		"""
		res = [v[1] for v in self.non_priority_list]
		return res, len(res), self.total_seats, self.applications_received

class PrioritySchool(School):
	def __init__(self, dbn, ranking, priority_students, priority_seats, total_seats):
		super().__init__(dbn, ranking, total_seats)

		self.priority_dict = {student_id: student_id in priority_students for student_id in ranking}
		self.priority_list = []
		self.priority_seats = priority_seats

	def check_match(self, student_id) -> bool:
		"""
		Returns `True` if a proposal received by the specified student will be accepted, `False` otherwise.

		Parameters:
		- student_id: the identifier of the proposing student.
		"""

		# If the school has residual capacity, the proposal will be accepted regardless of rank.
		if len(self.priority_list) + len(self.non_priority_list) < self.total_seats:
			return True

		rank = self.ranking.get(student_id)
		has_priority = self.priority_dict.get(student_id)

		# If the proposing student has priority and the priority list has residual capacity, the proposal will be accepted.
		if has_priority and len(self.priority_list) < self.priority_seats:
			return True

		# If the school has provisionally accepted at least one competing student that ranks worse than the proposing
		# student, then the proposal will be accepted, otherwise it will be rejected.
		# Competing students are restricted to students in the non-priority list for non-priority proponents.
		outranks_priority = rank < -self.priority_list[0][0]
		outranks_non_priority = len(self.non_priority_list) > 0 and rank < -self.non_priority_list[0][0]
		return (has_priority and outranks_priority) or outranks_non_priority

	def handle_proposal(self, student_id) -> Student | None:
		"""
		Handles the proposal received by the specified student and returns the identifier of the student that is rejected
		as a result, if applicable. If no student is rejected as a consequence of this proposal, returns `None`.

		Parameters:
		- student_id: the identifier of the proposing student.

		Returns: the identifier of the rejected student, or `None` if no student was rejected.
		"""

		rank = self.ranking.get(student_id)
		item = (-rank, student_id)  # student_rank is negated since heapq implements a min heap
		has_priority = self.priority_dict.get(student_id)

		# If the school has residual capacity, accept the proposing student and reject no one. Non-priority students will
		# be placed in the non-priority list, while priority students can be placed anywhere, depending on availability.
		if len(self.priority_list) + len(self.non_priority_list) < self.total_seats:
			if not has_priority:
				heappush(self.non_priority_list, item)
				return None

			# Note that a priority student can be only placed in the priority list in two cases:
			# 1. if there are any remaining priority seats, or
			if len(self.priority_list) < self.priority_seats:
				heappush(self.priority_list, item)
				return None

			# 2. if they outrank at least one student currently in the priority list, who will be bumped to non-priority.
			bumped = heappushpop(self.priority_list, item)
			heappush(self.non_priority_list, bumped)
			return None

		# If all seats have been filled but the proposing student has priority and there are any remaining priority seats,
		# accept the proposing student and reject the lowest-ranking non-priority student.
		if has_priority and len(self.priority_list) < self.priority_seats:
			heappush(self.priority_list, item)
			reject = heappop(self.non_priority_list)[1]
			return reject[1]

		# If all seats have been filled and there are no remaining priority seats, accept the proposing student on a
		# provisional basis, then reject the lowest-ranking among the qualifying competing students.
		if has_priority:

			# If the proposing student has priority, every student qualifies as competition; if the initially rejected student
			# was in the priority list, then the proponent will take their place in the priority list, and the initially
			# rejected student will be compared to students in the non-priority list to determine the actual rejected student.
			reject_priority = heappushpop(self.priority_list, item)
			reject_overall = heappushpop(self.non_priority_list, reject_priority)
			return reject_overall[1]

		# If the proposing student does not have priority, only students in the non-priority list qualify as competition,
		# even if they have priority (this means all priority seats have been filled by better ranked priority students).
		reject = heappushpop(self.non_priority_list, item)
		return reject[1]

	def get_result(self) -> tuple[list, int, int, int]:
		"""
		Returns the students matched with the school, their count, the number of available spots, and the number of applicants.
		"""
		res = [v[1] for v in self.priority_list + self.non_priority_list]
		return res, len(res), self.priority_seats + self.total_seats, self.applications_received

class Matching:
	def __init__(self, students, student_info, schools, school_info):
		lottery_idx = 1  # In student_info, the lottery number is found at index 1 of the attribute array
		capacity_idx = 1  # In school_info, the capacity is found at index 1 of the attribute array

		self.students = { student_id:
			Student(student_id, ranking, student_info[student_id][lottery_idx]) for student_id, ranking in students.items()
		}
		self.schools = { school_dbn:
			School(school_dbn, ranking, school_info[school_dbn][capacity_idx]) for school_dbn, ranking in schools.items()
		}

	def run(self):
		"""
		Runs the Gale-Shapley matching algorithm between the provided students and schools.
		"""

		# This FIFO queue contains all applicants that do not currently have a match, but can still propose to some schools
		# on their ranking. If it's empty, then all students are either matched to a school or definitively unmatched.
		students_to_match = list(self.students.copy().items())

		while students_to_match:
			identifier, current_student = students_to_match.pop(0)

			matched = False
			# The current student will keep proposing, moving down their school ranking, until one of the following occurs:
	 		# - They are accepted by one of the schools, so they find a (provisional) match, in which case matched == True.
			# - They propose to all schools on their ranking and never get accepted, so they are unmatched, in which case
	 		#   matched == False.
			while not matched and current_student.can_propose():
				target_dbn = current_student.propose()  # Get the DBN of the next school to propose to
				target_school = self.schools.get(target_dbn)  # Fetch the school object
				matched = target_school.check_match(identifier)  # Check if the school would accept the proposal

			# If the student stopped proposing because they found a match, it is possible that the school that accepted them
	 		# had to reject a student who was previously holding a seat. If that is the case, the rejected student will be
			# added back to the queue to try and find a new match.
			if matched:
				current_student.update_match_status(True)
				# Add the current student to the school accepting them and fetch the ID of the rejected student, if any
				rejected_id = target_school.handle_proposal(identifier)
				if rejected_id is not None:
					rejected_student = self.students.get(rejected_id)
					rejected_student.update_match_status(False)
					students_to_match.append((rejected_id, rejected_student))

	def check_stability(self):
		"""
		Checks the stability of the outcome of the matching algorithm.
		"""

		print()
		print('*** Stability check ***')

		# Verify that the outcome is stable, i.e. there are no applicant-school pairs (a_j, s_i) such that both of the
		# following conditions apply:
		# (1) The applicant a_j would rather be matched with s_i than with their assigned match.
		# (2) The school s_i would rather accept a_j than its lowest-ranked accepted student.

		unstable_pairs = []
		for student_id, student in self.students.items():
			# For all students, fetch the school they were matched to and the school's placement on their ranking
			matched_school, rank = student.get_result()

			# If rank == 0 then the student is matched with their top choice, thus the pair is stable
			if rank is None or rank > 0:
				# If rank > 0 then the student is not matched with their top choice; fetching all schools the student ranked
				# higher than their match satisfies condition (1)
				preferred_schools = student.ranking if rank is None else student.ranking[:rank]
				i = 0
				found_unstable = False

				# Iterating over the schools this student would prefer over their match
				while not found_unstable and i < len(preferred_schools):
					current_school_dbn = preferred_schools[i]
					current_school = self.schools[current_school_dbn]  # Fetch school object from DBN
					i += 1

					# The school prefers this student if the student is ranked higher than the lowest-ranked among the applicants
		 			# already accepted by the school: if that happens, then condition (2) is satisfied and the match is unstable
					if current_school.prefers_to_matches(student_id):
						found_unstable = True
						unstable_pairs.append((student_id, (matched_school, rank+1 if rank else None), (current_school_dbn, i)))

		num_unstable = len(unstable_pairs)
		print()
		print(f'The outcome is {"stable" if num_unstable == 0 else "unstable"} ({num_unstable} unstable pairs).')

	def get_results(self, rs=None, save_to_disk=True) -> tuple[dict, dict, dict]:
		"""
		Returns the outcome of the matching algorithm, saving it to disk if requested.

		Parameters:
		- rs: the random state used to generate the input (only used to save the outcome to disk).
		- save_to_disk: if `True`, the outcome will be saved to disk.

		Returns:
		- bins: list of lists comprising the lottery numbers of all the applicants, grouped by the position in the preference
		list of the school they were matched to.
		- matches: dictionary whose keys are the identifiers of each applicant and whose values are dictionaries including
		the DBN of the school the applicant was matched to (at index `dbn`) and the school's position in their preference
		list (at index `rank`). Unmatched applicants have the DBN and rank set to `None`.
		- school_outcome: dictionary whose keys are the DBNs of each school and whose values are dictionaries including the
		list of the students matched with the school (index `matches`) and their number (index `match_count`), the school's
		total capacity (index `total_seats`), and the number of true applicants to the school (index `true_applicants`).
		"""

		bins = {i: [] for i in range(13)}
		matches = {}
		school_data = {}

		for id, student in self.students.items():
			school, rank = student.get_result()
			matches[id] = {
				'dbn': school,
				'rank': rank+1 if rank is not None else None
			}
			if rank is None:
				bins[12].append(student.lottery_number)
			else:
				bins[rank].append(student.lottery_number)

		for dbn, school in self.schools.items():
			# format: number of accepted students, number of seats, number of true applicants
			matched_students, count, seats, apps = school.get_result()
			school_data[dbn] = {
				'matches': matched_students,
				'match_count': count,
				'total_seats': seats,
				'true_applicants': apps
			}

		if save_to_disk:
			path = f'BackEnd/Data/Simulation/results_rs{rs}/'
			Path(path).mkdir(parents=True, exist_ok=True)
			np.save(path + 'bins.npy', bins, allow_pickle=True)
			np.save(path + 'matches.npy', matches, allow_pickle=True)
			np.save(path + 'school_outcome.npy', school_data, allow_pickle=True)

		return bins, matches, school_data

def run_matching(students, student_info, schools, school_info):
	"""
	Instantiates the students and schools based on the provided information, then runs the matching algorithm and returns
	the outcome.

	Parameters:
	- students: dictionary indexed by the students' identifiers reporting each student's preference profile.
	- student_info: dictionary indexed by the students' identifiers reporting each student's representation in list form.
	- schools: dictionary indexed by the schools' DBNs reporting each schools's ranking of the applicants.
	- school_info: dictionary indexed by the schools' DBNs reporting each school's representation in list form.
	"""

	match = Matching(students, student_info, schools, school_info)
	match.run()
	return match.get_results(save_to_disk=False)

def run_matching_offline(random_state):
	"""
	Instantiates the students and schools based on the provided random state, then runs the matching algorithm and returns
	the outcome.
	"""

	path = f'BackEnd/Data/Generated/simulation_results_rs{random_state}/'

	students = np.load(path + 'student_rankings.npy', allow_pickle=True).item()
	student_info = np.load(path + 'student_info.npy', allow_pickle=True).item()
	schools = np.load(path + 'school_rankings.npy', allow_pickle=True).item()
	school_info = np.load(path + 'school_info.npy', allow_pickle=True).item()

	match = Matching(students, student_info, schools, school_info)
	match.run()

	match.get_results(random_state, save_to_disk=True)

if __name__ == '__main__':
	for random_state in range(1, 51):
		run_matching_offline(random_state)
		print(f'Random state {random_state}: done')