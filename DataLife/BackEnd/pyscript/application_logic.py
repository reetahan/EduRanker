from pyscript import window, when
from pyscript.js_modules import script as js
from pyodide.ffi import to_js

import numpy as np

import gale_shapley as gs
from oneshot import oneshot_with_input

STUDENT_ID = 'current_user'

def is_valid(lottery) -> bool:
	if lottery is None:
		return False

	lottery = lottery.replace('-', '').lower()
	if len(lottery) != 32:
		return False

	for c in lottery:
		if c not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']:
			return False

	return True

def dict_to_list(dictionary, sort=True) -> list:
	"""
	Given a dictionary of lists of different lenghts, returns the corresponding list-of-lists representation. If `sort` is
	`True`, then each list is sorted.
	"""

	lst = [None] * len(dictionary.keys())
	for i in range(len(lst)):
		val = dictionary.get(i)
		if len(val) == 0:
			val = ['0000']
		lst[i] = sorted(val) if sort else val
	return lst

def load_simulation_results(has_data, student_lottery, student_gpa, student_list, rs=1):
	"""
	Runs the Gale-Shapley matching algorithm over the generated students and schools, returning the outcome along with
	data about the students and schools. If `has_data` is `True`, then a new student is generated and included in the
	matching.

	Parameters:
	- has_data: if `False`, then the outcome of the matching is directly loaded from disk, otherwise a new student is
	generated based on the other arguments and added to the matching, which will run in real time.
	- student_lottery: the lottery number of the new student (cannot be `None`). Only applicable if `has_data` is `True`.
	- student_gpa: the grade average of the new student; if `None`, then the grade average will be selected at random.
	Only applicable if `has_data` is `True`.
	- student_list: the preference profile of the new student (cannot be `None` or empty). Only applicable if `has_data`
	is `True`.
	- rs: the random state for the simulation. Must be between 1 and 50, defaults to 1.

	Returns:
	- student_dict: dictionary whose keys are the identifiers of each applicant and whose values are dictionaries including
	the applicant's lottery number (`lottery`), school selection policy (`selection`), school ranking policy (`ranking`),
	preference profile length (`list_length`), and GPA (`gpa`).
	- school_dict: dictionary whose keys are the DBNs of each school and whose values are dictionaries including the
	school's admission policy (`policy`), popularity (`popularity`), and likeability (`likeability`).
	- bins: list of lists comprising the lottery numbers of all the applicants, grouped by the position in the preference
	list of the school they were matched to.
	- matches: dictionary whose keys are the identifiers of each applicant and whose values are dictionaries including
	the DBN of the school the applicant was matched to (index `dbn`) and the school's position in their preference
	list (index `rank`). Unmatched applicants have the DBN and rank set to `None`.
	- school_outcome: dictionary whose keys are the DBNs of each school and whose values are dictionaries including the
	list of the students matched with the school (index `matches`) and their number (index `match_count`), the school's
	total capacity (index `total_seats`), and the number of true applicants to the school (index `true_applicants`).
	"""

	# If no custom data is provided, load results from previous simulation for the specified random state
	if not has_data:
		# Load student and school data
		path_gen = f'gen_{rs}/'
		path_sim = f'sim_{rs}/'

		student_info = np.load(path_gen + 'student_info.npy', allow_pickle=True).item()
		school_info = np.load(path_gen + 'school_info.npy', allow_pickle=True).item()

		student_dict = {
			k: { 'lottery': v[1], 'selection': v[2], 'ranking': v[3], 'list_length': v[4], 'gpa': v[5] }
			for k, v in student_info.items()
		}
		school_dict = {
			k: { 'policy': v[2], 'popularity': v[3], 'likeability': v[4] }
			for k, v in school_info.items()
		}

		# Load matching outcome
		bins = dict_to_list(np.load(path_sim + 'bins.npy', allow_pickle=True).item())
		matches = np.load(path_sim + 'matches.npy', allow_pickle=True).item()
		school_outcome = np.load(path_sim + 'school_outcome.npy', allow_pickle=True).item()

	# Otherwise, execute the one-shot pipeline to obtain the input, then run the simulation
	else:
		# Run the one-shot pipeline to generate students and schools, adding the student whose data was provided
		students, schools, student_info, school_info = \
			oneshot_with_input(rs, student_lottery, student_list, float(student_gpa or -1), student_name=STUDENT_ID, return_list=True)

		student_dict = {
			k: { 'lottery': v[1], 'selection': v[2], 'ranking': v[3], 'list_length': v[4], 'gpa': v[5] }
			for k, v in student_info.items()
		}
		school_dict = {
			k: { 'policy': v[2], 'popularity': v[3], 'likeability': v[4] }
			for k, v in school_info.items()
		}

		# Run the matching and return the outcome
		bins, matches, school_outcome = gs.run_matching(students, student_info, schools, school_info)
		bins = dict_to_list(bins)

	return student_dict, school_dict, bins, matches, school_outcome

@when('click', '#run-simulation')
def on_click(event):
	"""
	When the "Run simulation" button is clicked, fetches the user data from JavaScript and runs the matching simulation,
	adding a new student with the provided data if applicable. Then, passes the outcome to JavaScript by setting global
	variables on the `window` object and dispatching a custom event to notify the script that the simulation is done running.

	The `event` parameter is unused, but it needs to be in the function's signature to match the requirements for PyScript
	event listener callbacks.
	"""

	res = js.userData.getValues(True).to_py()
	has_data = res.get('hasData')
	lottery = res.get('lottery')
	gpa = res.get('gpa')
	preferences = res.get('preferences')
	random_state = res.get('rs')

	if has_data and (not is_valid(lottery) or preferences is None):
		js.displayAlert('Please enter a valid lottery number and at least one school choice to run the simulation')
		window.py_error = to_js(True)
		window.dispatchEvent(js.pyDoneEvent)

	# Load simulation results based on the provided data
	students, schools, bins, matches, school_outcome = load_simulation_results(has_data, lottery, gpa, preferences, rs=random_state)

	window.py_students = to_js(students)
	window.py_schools = to_js(schools)
	window.py_bins = to_js(bins)
	window.py_matches = to_js(matches)
	window.py_school_outcome = to_js(school_outcome)
	window.dispatchEvent(js.pyDoneEvent)