import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';

const STUDENT_ID = 'current_user';  // Used to retrieve the custom data provided in input, when applicable
const removeIcon = 'cancel';
const pyDoneEvent = new Event('py-backend-done');
let userData = null;

const lotteryNumbers = [
	'3a4bdc7f-8e91-0c2a-6f5d-1b9e6a38d047',
	'0c68f153-12e5-3e7f-2234-6517592f1763',
	'9568a214-f388-402d-4145-45e8138643e',
	'7833cfd2-93c5-b281-407b-c3b96160ea3f',
	'd8b0965e-9e71-c53b-f849-953a82e164d1',
	'211cee28-3c31-bf81-4a16-6f39344b9f97',
	'cb6e6546-cfd6-0134-d85a-5e8041f8b06c',
	'4f6bd3ef-2929-d7c0-8215-c4065d0109b4'
];

function displayAlert(s) {
	alert(s);
}

/**
 * Returns the ordinal string for a given integer.
 */
function getOrdinal(num) {
	if (num % 10 == 0 || num % 10 > 3 || [11, 12, 13].includes(num))
		return num + 'th';
	if (num % 10 == 1)
		return num + 'st';
	if (num % 10 == 2)
		return num + 'nd';
	if (num % 10 == 3)
		return num + 'rd';
}

/**
 * Returns the numerical value of the first 8 digits of a hexadecimal string.
 */
function getHexValue(string) {
	return Number('0x' + string.replaceAll('-', '').substring(0, 8));
}

/**
 * Returns the first `numDigits` digits of the hexadecimal representation of the given number, padded to reach 8 digits if needed.
 */
function getHexDigits(num, numDigits = 1) {
	num = Math.floor(num);
	return num.toString(16).padStart(8, '0').substring(0, numDigits).toUpperCase();
}

/**
 * This class animates the lottery number examples element in the website. The `iterate` function updates the lottery
 * number shown in the page every 3 seconds.
 */
class LotteryNumberExamples {
	constructor() {
		this.el = document.getElementById('lottery-example');
		this.point = 0;
	}

	updateLotteryNumber() {
		this.el.innerText = lotteryNumbers[this.point];
		if (this.point < lotteryNumbers.length - 1)
			this.point++;
		else
			this.point = 0;
	}

	iterate() {
		const self = this;
		setInterval(self.updateLotteryNumber.bind(self), 3000);
	}
}

/**
 * This class handles the expandable explainers for the lottery numbers.
 */
class LotteryExplainers {
	constructor() {
		this.expand1 = document.getElementById('expand-1');
		this.expand2 = document.getElementById('expand-2');
		this.expand3 = document.getElementById('expand-3');

		this.blob1 = document.getElementById('blob-1');
		this.blob2 = document.getElementById('blob-2');
		this.blob3 = document.getElementById('blob-3');
	}

	initializeListeners() {
		const self = this;

		this.expand1.addEventListener('click', () => {
			if (self.expand1.children[0].innerText === 'open_in_full') {
				self.blob1.classList.remove('hide');
				self.expand1.children[0].innerText = 'close_fullscreen';
			} else if (self.expand1.children[0].innerText === 'close_fullscreen') {
				self.blob1.classList.add('hide');
				self.expand1.children[0].innerText = 'open_in_full';
			}
		});

		this.expand2.addEventListener('click', () => {
			if (self.expand2.children[0].innerText === 'open_in_full') {
				self.blob2.classList.remove('hide');
				self.expand2.children[0].innerText = 'close_fullscreen';
			} else if (self.expand2.children[0].innerText === 'close_fullscreen') {
				self.blob2.classList.add('hide');
				self.expand2.children[0].innerText = 'open_in_full';
			}
		});

		this.expand3.addEventListener('click', () => {
			if (self.expand3.children[0].innerText === 'open_in_full') {
				self.blob3.classList.remove('hide');
				self.expand3.children[0].innerText = 'close_fullscreen';
			} else if (self.expand3.children[0].innerText === 'close_fullscreen') {
				self.blob3.classList.add('hide');
				self.expand3.children[0].innerText = 'open_in_full';
			}
		});
	}
}

/**
 * This class is used to instantiate items of the school choice list.
 */
class SchoolChoice {
	constructor(num) {
		this.num = num;

		this.el = document.createElement('li');
		this.el.id = 'choice-' + num;
		this.el.classList.add('flex-row', 'input');
		this.el.draggable = true;

		const icon_drag = document.createElement('span');
		icon_drag.classList.add('material-symbols-outlined', 'draggable');
		icon_drag.innerHTML = 'drag_indicator';
		this.el.appendChild(icon_drag);

		const input = document.createElement('input');
		input.name = 'choice-' + num;
		input.placeholder = 'Select a school';
		input.size = 56;
		input.setAttribute('list', 'school-list');
		this.el.appendChild(input);

		const icon_del = document.createElement('span');
		icon_del.classList.add('material-symbols-outlined', 'clickable');
		icon_del.innerHTML = removeIcon;
		icon_del.addEventListener('click', this.removeSchoolChoice);
		this.el.appendChild(icon_del);
	}

	getElement() {
		return this.el;
	}
}

/**
 * This class regulates the behavior of the school choice list, handling the listeners regarding the "Add choice" button
 * and enabling drag-and-drop between list items.
 */
class SchoolList {
	constructor() {
		this.el = document.getElementById('choices').children[0];
		this.addChoiceItem = document.getElementById('add-choice-li');
		this.addChoiceBtn = document.getElementById('add-choice');

		this.attachDeleteEventListener();
		this.attachButtonEventListener();

		this.dragged = null;
		this.attachInputEventListeners();
	}

	attachDeleteEventListener() {
		const self = this;

		this.el.addEventListener('click', e => {
			if (e.target.innerText !== removeIcon)
				return;

			self.el.removeChild(e.target.parentNode);

			let hasEmpty = false;
			for (let item of self.el.children) {
				if (item.children[0].id !== 'add-choice')
					hasEmpty = hasEmpty || (item.children[1].value === '');
			}

			if (!hasEmpty)
				self.addChoiceBtn.classList.remove('disabled');
		});
	}

	attachButtonEventListener() {
		const self = this;

		this.addChoiceBtn.addEventListener('click', () => {
			if (self.addChoiceBtn.classList.contains('disabled'))
				return;

			const numChoices = self.el.childElementCount - 1;
			const newChoice = new SchoolChoice(numChoices + 1).getElement();

			self.el.insertBefore(newChoice, self.addChoiceItem);
			self.addChoiceBtn.classList.add('disabled');

			self.attachInputEventListeners();
		});
	}

	attachInputEventListeners() {
		const self = this;
		let dragged = null;

		for (let item of this.el.children) {
			if (item.children[0].id !== 'add-choice') {

				item.addEventListener('change', () => {
					const numChoices = self.el.childElementCount - 1;
					if (numChoices >= 12)
						return;

					let hasEmpty = false;
					for (let item of self.el.children) {
						if (item.children[0].id !== 'add-choice')
							hasEmpty = hasEmpty || (item.children[1].value === '');
					}

					if (!hasEmpty)
						self.addChoiceBtn.classList.remove('disabled');
				});

				item.addEventListener('dragstart', e => dragged = e.target);
				item.addEventListener('dragover', e => e.preventDefault());

				item.addEventListener('drop', e => {
					let rows = Array.from(self.el.children);
					let destination = e.target.closest('li');

					if (dragged == null || destination == null)
						return;

					if (rows.indexOf(destination) > rows.indexOf(dragged))
						destination.after(dragged);
					else
						destination.before(dragged);
				});
			}
		}
	}

	/**
	 * Returns an array containing the DBNs of the chosen schools, in the order specified by the user, or `null` if the
	 * list is empty.
	 */
	getValues() {
		let res = [];

		for (let item of this.el.children) {
			if (item.children[0].id !== 'add-choice') {  // For each list item except for the one with the "add" button
				let val = item.children[1].value;
				if (val !== '')
					res.push(val.substring(0, 6));  // Only select the DBN of the school
			}
		}

		return (res.length === 0 ? null : res);
	}

	getSchoolByIndex(index) {
		return this.el.children[index].children[1].value;
	}
}

/**
 * This class handles all of the user-provided data to be passed to the backend for running the matching simulation.
 */
class UserData {
	constructor(schoolList) {
		// This datalist includes all the NYC high schools that took part in the matching process in 2023
		this.datalist = document.getElementById('school-list');

		this.addDataBtn = document.getElementById('btn-add-data');
		this.removeDataBtn = document.getElementById('btn-remove-data');

		// Shuffle controls whether the random state for the simulation is updated at each execution (shuffle on) or kept stable (shuffle off)
		this.shuffleOff = document.getElementById('shuffle-off');
		this.shuffleOn = document.getElementById('shuffle-on');
		this.randomState = null;  // Default: shuffle on, random state set to null, dynamically assigned when needed

		this.customData = document.getElementById('custom-data');
		this.lotteryInput = document.getElementById('custom-lottery-input');
		this.gpaInput = document.getElementById('custom-gpa-input');
		this.choicesList = new SchoolList();

		// This property keeps track of whether the user is providing any custom data to add to the simulation
		this.hasCustomData = false;

		this.addDataBtn.addEventListener('click', () => {
			this.customData.classList.remove('hide');
			this.addDataBtn.classList.add('hide');
			this.removeDataBtn.classList.remove('hide');
			this.hasCustomData = true;
		});

		this.removeDataBtn.addEventListener('click', () => {
			this.customData.classList.add('hide');
			this.removeDataBtn.classList.add('hide');
			this.addDataBtn.classList.remove('hide');
			this.hasCustomData = false;
		});

		this.shuffleOff.addEventListener('click', () => {
			this.shuffleOff.classList.add('hide');
			this.shuffleOn.classList.remove('hide');
			this.randomState = null;  // Shuffle on: random state set to null, will be computed every time simulation is run
		});

		this.shuffleOn.addEventListener('click', () => {
			this.shuffleOn.classList.add('hide');
			this.shuffleOff.classList.remove('hide');
			this.randomState = 1 + Date.now() % 50;  // Shuffle off: random state set to a value that will be passed directly
		});

		this.initializeDatalist(schoolList);
	}

	/**
	 * This function initializes the options for the datalist containing NYC high schools that took part in the matching
	 * in 2023. The options are parsed from a JSON file and passed as a parameter.
	 */
	initializeDatalist(schoolList) {
		schoolList.forEach(opt => {
			let option = document.createElement('option');
			option.value = opt.dbn + ' - ' + opt.name;
			this.datalist.appendChild(option);
		});
	}

	/**
	 * This function returns an object containing the custom data passed by the user (lottery number, ranking,
	 * GPA) along with a security check, whose value is `False` only if no data has been provided, and the random state.
	 */
	getValues(updateRandomState = false) {
		let rs = this.randomState;
		// As getValue is also called after the simulation, updateRandomState is only true when a new simulation has been launched
		if (updateRandomState)
			// If rs is null, then shuffle is on: compute a new random state based on the current timestamp
			// Otherwise, shuffle is off: keep the previous random state to ensure stability
			if (rs === null)
				rs = 1 + Date.now() % 50;

		if (!this.hasCustomData)
			return { hasData: false, lottery: null, gpa: null, preferences: null, rs: rs };

		const userLottery = this.lotteryInput.value === '' ? null : this.lotteryInput.value;
		const userGPA = this.gpaInput.value === '' ? null : this.gpaInput.value;
		const userPreferences = this.choicesList.getValues();

		const securityCheck = (userLottery === null) && (userGPA === null) && (userPreferences === null);

		return { hasData: !securityCheck, lottery: userLottery, gpa: userGPA, preferences: userPreferences, rs: rs };
	}

	getSchoolByIndex(index) {
		return this.choicesList.getSchoolByIndex(index);
	}
}

/**
 * This class handles all of the data visualization to be performed after a simulation is executed. It manages both the
 * nutritional label and the focus sections shown below it.
 */
class DataVisualizer {
	constructor(students, schools, bins, matches, schoolOutcome) {
		// { student_id: { lottery: ..., selection: ..., ranking: ..., list_length: ..., gpa: ... } }
		this.studentInfo = students;
		// { school_dbn: { policy: ..., popularity: (propto total apps), likeability: (true apps/total apps) } }
		this.schoolInfo = schools;

		this.bins = bins.map(bin => bin.map(num => num.replaceAll('-', '')));
		// { student_id: { dbn: (matched school or null), rank: (matched position or null) } }
		this.matches = matches;
		// { school_dbn: { matches: [ ... ], match_count: ..., total_seats: ..., true_applicants: ... } }
		this.schoolOutcome = schoolOutcome;

		this.el = document.getElementById('simulation-results');

		// Nutritional label: outcome from custom data (only if input is provided)
		this.customOutcomeSection = document.getElementById('custom-outcome');
		this.pCustomOutcome = document.getElementById('school-match').childNodes[0];

		// Nutritional label: input data
		this.pTotalStudents = document.getElementById('total-applicants').childNodes[0];
		this.pTotalSchools = document.getElementById('total-schools').childNodes[0];
		this.pTotalCapacity = document.getElementById('total-capacity').childNodes[0];
		this.pPreferenceStrategy = document.getElementById('preference-strategy').childNodes[0];
		this.pListLength = document.getElementById('average-list-length').childNodes[0];
		this.pAdmissionPolicy = document.getElementById('admission-policies').childNodes[0];

		// Nutritional label: student outcome
		this.pStudentsTop = document.getElementById('students-top').childNodes[0];
		this.pStudentsFive = document.getElementById('students-top-5').childNodes[0];
		this.pStudentsUnmatched = document.getElementById('students-unmatched').childNodes[0];

		// Nutritional label: school outcome
		this.pSchoolsUnfilled = document.getElementById('schools-unfilled').childNodes[0];
		this.pSeatsUnfilled = document.getElementById('seats-unfilled').childNodes[0];

		// Focus card 1
		this.focusSection = document.getElementById('focus');
		this.focusHeading = document.getElementById('focus-heading');
		this.focusNumber = document.getElementById('focus-number');
		this.focusSubtitle = document.getElementById('focus-subtitle');
		this.focusPar1 = document.getElementById('focus-par-1');
		this.focusPar2 = document.getElementById('focus-par-2');
		this.focusPar3 = document.getElementById('focus-par-3');
		this.plotPrimary = document.getElementById('plot-primary');

		// Focus card 2
		this.spotlightSection = document.getElementById('spotlight');
		this.spotlightHeading = document.getElementById('spotlight-heading');
		this.spotlightPar1 = document.getElementById('spotlight-par-1');
		this.spotlightPar2 = document.getElementById('spotlight-par-2');
		this.plotSecondary = document.getElementById('plot-secondary');
		this.selectFirstDigit = document.getElementById('first-digit');

		this.computeAggregateStats();
		this.attachListeners();
	}

	/**
	 * Compute aggregate statistics that will be used to compile the nutritional label.
	 */
	computeAggregateStats() {
		this.counts = this.bins.map(bin => bin.length);
		this.totalStudents = this.counts.reduce((a, b) => a + b, 0);
		this.ratios = this.bins.map(bin => bin.length / this.totalStudents);

		this.totalSchools = this.schoolOutcome.size;

		let sumListLength = 0;
		let policyMatrix = [[0, 0], [0, 0]];
		let sel, rnk;
		for (let val of this.studentInfo.values()) {
			sumListLength += val.get('list_length');  // Add the list length (used to compute average)
			sel = val.get('selection');
			rnk = val.get('ranking');
			if (sel >= 0 && rnk >= 0)
				policyMatrix[sel][rnk] += 1;  // Update student policy matrix
		}
		this.avgListLength = sumListLength / this.matches.size;
		this.studentPolicyMatrix = policyMatrix;

		let admissions = [0, 0, 0];
		for (let val of this.schoolInfo.values())
			admissions[val.get('policy')-1] += 1;  // Update school admission policy counts

		this.admissionPolicies = admissions;

		let totalCapacity = 0;
		let seatsUnfilled = 0;
		let schoolsUnfilled = [];
		for (let val of this.schoolOutcome.entries()) {
			let offers = val[1].get('match_count');
			let seats = val[1].get('total_seats');
			totalCapacity += seats;
			seatsUnfilled += (seats - offers);
			if (seats > offers)
				schoolsUnfilled.push(val[0]);
		}
		this.totalCapacity = totalCapacity;
		this.seatsUnfilled = seatsUnfilled;
		this.schoolsUnfilled = schoolsUnfilled;
		this.numSchoolsUnfilled = schoolsUnfilled.length;

		// Compute cumulative statistics for students matched with any of their top 5 choices
		this.ratioFive = [0, 1, 2, 3, 4].map(idx => this.ratios[idx]).reduce((a, b) => a + b, 0);
		this.countFive = [0, 1, 2, 3, 4].map(idx => this.counts[idx]).reduce((a, b) => a + b, 0);
	}

	attachListeners() {
		const self = this;

		this.pCustomOutcome.nextSibling.addEventListener('click', () => self.showFocus('custom'));

		this.pPreferenceStrategy.nextSibling.addEventListener('click', () => self.showFocus('input', 'strategy'));
		this.pAdmissionPolicy.nextSibling.addEventListener('click', () => self.showFocus('input', 'policy'));

		this.pStudentsTop.nextSibling.addEventListener('click', () => self.showFocus('student', 'top'));
		this.pStudentsFive.nextSibling.addEventListener('click', () => self.showFocus('student', 'top-5'));
		this.pStudentsUnmatched.nextSibling.addEventListener('click', () => self.showFocus('student', 'unmatched'));

		this.pSchoolsUnfilled.nextSibling.addEventListener('click', () => self.showFocus('school', 'unfilled'));
		this.pSeatsUnfilled.nextSibling.addEventListener('click', () => self.showFocus('school', 'seats'));

		this.selectFirstDigit.addEventListener('change', e => self.plotCounts(undefined, e.target.value));
	}

	computeCustomOutcome() {
		let heading, number, subtitle, par1, par2, par3;

		console.log(this.matches.get(STUDENT_ID));
		const outcomeRank = this.matches.get(STUDENT_ID).get('rank');
		const schools = userData.getValues().preferences;  // User's preference list
		const userLottery = this.studentInfo.get(STUDENT_ID).get('lottery').replaceAll('-', '');
		const userLotteryHex = getHexValue(userLottery);
		const userGrade = this.studentInfo.get(STUDENT_ID).get('gpa');

		// Info about the admission policy of the schools selected by the user (always applicable)
		const policies = [0, 0, 0];
		schools.map(dbn => this.schoolInfo.get(dbn))  // Retrieve the schoolInfo object for each school
			.forEach(s => {	policies[s.get('policy') - 1] += 1 });

		par3 = 'You applied to '
			+ policies[0] + ' open ' + (policies[0] == 1 ? 'school' : 'schools') + ', '
			+ policies[1] + ' EdOpt ' + (policies[1] == 1 ? 'school' : 'schools') + ' and '
			+ policies[2] + ' screen ' + (policies[2] == 1 ? 'school' : 'schools') + '.';

		// Info about students matched to the same school as the user (only if user is not unmatched)
		if (outcomeRank != undefined) {
			const outcomeDBN = schools[outcomeRank - 1];
			const studentsOther = this.schoolOutcome.get(outcomeDBN).get('matches');
			const lotteriesOther = studentsOther.map(s => this.studentInfo.get(s).get('lottery')).map(getHexValue);
			const gradesOther = studentsOther.map(s => this.studentInfo.get(s).get('gpa'));

			// Evaluate lottery number and GPA w.r.t. quartiles among the students matched to this school
			let qualityLottery, qualityGrade;
			let lq1 = d3.quantile(lotteriesOther, .25), lq2 = d3.quantile(lotteriesOther, .5), lq3 = d3.quantile(lotteriesOther, .75);
			if (userLotteryHex < lq1)
				qualityLottery = 'good';
			else if (userLotteryHex < lq2)
				qualityLottery = 'above average';
			else if (userLotteryHex < lq3)
				qualityLottery = 'below average';
			else
				qualityLottery = 'bad';

			let gq1 = d3.quantile(gradesOther, .25), gq2 = d3.quantile(gradesOther, .5), gq3 = d3.quantile(gradesOther, .75);
			if (userGrade < gq1)
				qualityGrade = 'low';
			else if (userGrade < gq2)
				qualityGrade = 'below average';
			else if (userGrade < gq3)
				qualityGrade = 'above average';
			else
				qualityGrade = 'high';

			par2 = 'Compared to students matched with the same school as you, your lottery number is '
				+ qualityLottery + ' and your GPA is ' + qualityGrade + '.';
		}

		// Info about students matched to higher ranked schools (only if the match is not the top choice)
		if (outcomeRank != 1) {
			let numAhead = 0, numLottery = 0, numGrade = 0;

			// Retrieve all the schools the user ranked higher than his actual match
			const schoolsAhead = schools.slice(outcomeRank).map(dbn => this.schoolOutcome.get(dbn));
			schoolsAhead.forEach(school => {
				numAhead += school.get('match_count');

				school.get('matches')  // Retrieve all students matched with the school
					.map(m => this.studentInfo.get(m))  // Retrieve the studentInfo object for each student
					.forEach(student => {
						// Count students with a better lottery number than the user
						if (student.get('lottery').replaceAll('-', '') < userLottery)
							numLottery += 1;
						// Count students with a worse lottery number but a better GPA than the user
						else if (student.get('gpa') > userGrade)
							numGrade += 1;
					});
			});

			if (outcomeRank == undefined)
				heading = 'Among the applicants matched with the schools you selected,';
			else
				heading = 'Among the applicants matched with schools you ranked higher than ' + getOrdinal(outcomeRank) +',';
			number = (100 * numLottery / numAhead).toFixed(1) + '%';
			subtitle = 'had a better lottery number than yours.';
			par1 = 'Among those who matched with those schools having a worse lottery number, '
				+ (100 * numGrade / (numAhead - numLottery)).toFixed(1) + '% had a higher GPA than you.';
		} else {
			// In case the student was matched to their top choice
			heading = 'You were matched with your top choice.'
		}

		return { heading, number, subtitle, par1, par2, par3 };
	}

	computeStudentOutcome(variant) {
		let number, subtitle, par1, par2, par3, plot, index;

		// Compute stats about students matched with their top choice and display the choice counts on a histogram
		if (variant === 'top') {
			const medianTop = this.bins[0][Math.floor(this.bins[0].length / 2)];

			let sumGPA = 0;
			for (let match of this.matches.entries()) {
				if (match[1].get('rank') == 1)
					sumGPA += this.studentInfo.get(match[0]).get('gpa');
			}
			const avgGPA = sumGPA / this.counts[0];

			number = (100 * this.ratios[0]).toFixed(1) + '%';
			subtitle = 'of the applicants were matched to their top choice.';
			par1 = 'That\'s ' + this.counts[0].toLocaleString() + ' students out of ' + this.totalStudents.toLocaleString() + '.';
			par2 = 'Their median lottery number starts with the digits ' + medianTop.substring(0, 4).toUpperCase()
				+ ' and their average GPA is ' + avgGPA.toFixed(1) + '.';
			par3 = 'Click on any bar to learn more about the applicants matched with a specific choice.';
			plot = 'counts';
			index = 1;

		// Compute cumulative stats about students matched within their top 5 choices and display the tree map
		} else if (variant === 'top-5') {

			const combinedBins = [0, 1, 2, 3, 4].map(idx => this.bins[idx]).reduce((x, y) => x.concat(y), []).sort();
			const medianTopFive = combinedBins[Math.floor(combinedBins.length / 2)];

			let sumGPA = 0;
			for (let match of this.matches.entries()) {
				if (match[1].get('rank') <= 5)
					sumGPA += this.studentInfo.get(match[0]).get('gpa');
			}
			const avgGPA = sumGPA / this.countFive;

			number = (100 * this.ratioFive).toFixed(1) + '%';
			subtitle = 'of the applicants were matched to one of their top 5 choices.';
			par1 = 'That\'s ' + this.countFive.toLocaleString() + ' students out of ' + this.totalStudents.toLocaleString() + '.';
			par2 = 'Their median lottery number starts with the digits ' + medianTopFive.substring(0, 4).toUpperCase()
				+ ' and their average GPA is ' + avgGPA.toFixed(1) + '.';
			plot = 'counts-cumulative';

		// Compute stats about unmatched students and display the choice counts on a histogram
		} else if (variant === 'unmatched') {

			const medianUnmatched = this.bins[12][Math.floor(this.bins[12].length / 2)];

			let sumGPA = 0;
			for (let match of this.matches.entries()) {
				if (match[1].get('rank') == undefined)
					sumGPA += this.studentInfo.get(match[0]).get('gpa');
			}
			const avgGPA = sumGPA / this.counts[12];

			number = (100 * this.ratios[12]).toFixed(1) + '%';
			subtitle = 'of the applicants were unmatched to their choices.';
			par1 = 'That\'s ' + this.counts[12].toLocaleString() + ' students out of ' + this.totalStudents.toLocaleString() + '.';
			par2 = 'Their median lottery number starts with the digits ' + medianUnmatched.substring(0, 4).toUpperCase()
				+ ' and their average GPA is ' + avgGPA.toFixed(1) + '.';
			par3 = 'Click on any bar to learn more about the applicants matched with a specific choice.';
			plot = 'counts';
			index = 13;
		}

		return { number, subtitle, par1, par2, par3, plot, index };
	}

	/**
	 * This function displays the focus section with data relative to the nutritional label field expanded by the user.
	 */
	showFocus(target, variant) {
		let heading, number, subtitle, par1, par2, par3, plot, index;

		if (target === 'custom') {
			let res = this.computeCustomOutcome();
			heading = res.heading;
			number = res.number;
			subtitle = res.subtitle;
			par1 = res.par1;
			par2 = res.par2;
			par3 = res.par3;

		} else if (target === 'input') {

			if (variant === 'strategy') {
				par1 = (this.studentPolicyMatrix[0][0] + this.studentPolicyMatrix[0][1]).toLocaleString()
					+ ' applicants selected the schools at random. Among them, ' + this.studentPolicyMatrix[0][0].toLocaleString()
					+ ' left the ranking unordered, while ' + this.studentPolicyMatrix[0][1].toLocaleString()
					+ ' sorted the selected schools by likeability.';
				par2 = (this.studentPolicyMatrix[1][0] + this.studentPolicyMatrix[1][1]).toLocaleString()
					+ ' applicants selected the schools based on their popularity. Among them, '
					+ this.studentPolicyMatrix[1][0].toLocaleString() + ' left the ranking unordered, while '
					+ this.studentPolicyMatrix[0][1].toLocaleString() + ' sorted the selected schools by likeability.';
			} else if (variant === 'policy') {
				par1 = 'In this simulation, ' + this.admissionPolicies[0] + ' schools were open, ' + this.admissionPolicies[1]
					+ ' schools had EdOpt admissions, and ' + this.admissionPolicies[2] + ' schools had screen admissions.';
			}

		} else if (target === 'student') {
			let res = this.computeStudentOutcome(variant);
			number = res.number;
			subtitle = res.subtitle;
			par1 = res.par1;
			par2 = res.par2;
			par3 = res.par3;
			plot = res.plot;
			index = res.index;

		} else if (target === 'school') {
			let totalPopularity = 0, partialPopularity = 0;
			for (let e of this.schoolInfo.entries()) {
				totalPopularity += e[1].get('popularity');
				if (this.schoolsUnfilled.includes(e[0]))
					partialPopularity += e[1].get('popularity');
			}

			let popularityDifferential = (totalPopularity / this.totalSchools) / (partialPopularity / this.numSchoolsUnfilled) - 1;
			let popularityWeightRatio = (this.studentPolicyMatrix[1][0] + this.studentPolicyMatrix[1][0]) / this.totalStudents;

			number = (100 * this.numSchoolsUnfilled / this.totalSchools).toFixed(1) + '%';
			subtitle = 'of the schools (' + this.numSchoolsUnfilled + ' out of ' + this.totalSchools + ') are not at full capacity.';
			par1 = 'This amounts to a total of ' + this.seatsUnfilled.toLocaleString() + ' seats left unfilled, for an average of '
				+ (this.seatsUnfilled / this.numSchoolsUnfilled).toFixed(1) + ' unfilled seats per school.';
			par2 = 'Overall, the average school was ' +
				(100 * popularityDifferential * popularityWeightRatio).toFixed(1)
				+ '% more likely to be added to a student\'s ranking compared to these schools.'

		}

		this.refreshFocus(heading, number, subtitle, par1, par2, par3, plot, index);
	}

	show() {
		this.el.classList.remove('hide');
		this.el.scrollIntoView(true);
	}

	/**
	 * Plot the range of lottery numbers for students matched with their `index`-th choice (or unmatched, if `index` is 13).
	 * Displays a summary of the 90% range, inter-quartile range, and median.
	 */
	plotLotteryRange(index) {
		// Clear any existing plots
		this.plotSecondary.innerHTML = '';

		const hexDigits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f'];
		const bin = this.bins[index - 1].map(getHexValue);
		const data = {
			index: index,
			min: d3.quantile(bin, .05),
			max: d3.quantile(bin, .95),
			q1: d3.quantile(bin, .25),
			q3: d3.quantile(bin, .75),
			med: d3.quantile(bin, .5)
		};

		const margin = { top: 25, right: 25, bottom: 25, left: 25 };
		const width = 400;
		const height = 100;
		const boxCenter = 50;
		const boxWidth = 30;

		// Define the X axis scale
		let x = d3.scaleLinear()
			.domain([0, getHexValue('ffffffff')])
			.range([margin.left, width - margin.right]);

		// Append the SVG container
		const svg = d3.select('#plot-secondary')
			.append('svg')
			.attr('width', width)
			.attr('height', height)
			.attr('viewBox', [0, 0, width, height]);

		// Add the X axis
		svg.append('g')
			.attr('transform', 'translate(0,' + (height - margin.bottom) + ')')
			.call(d3.axisBottom(x)
				.tickValues(hexDigits.map(n => getHexValue(n + '0000000')))
				.tickSizeOuter(0)
				.tickFormat(x => getHexDigits(x)))
			.attr('font-size', '14px')
			.call(g => g.append('text')
				.attr('x', margin.left)
				.attr('y', -50)
				.attr('fill', '#000000')
				.attr('text-anchor', 'start')
				.attr('font-size', '14px')
				.text('First digit'));

		// Append the min-max line
		svg.append('line')
			.attr('x1', x(data.min))
			.attr('x2', x(data.max))
			.attr('y1', boxCenter)
			.attr('y2', boxCenter)
			.attr('stroke', 'black');

		// Append the box
		svg.append('rect')
			.attr('fill', '#4036ed')
			.attr('x', x(data.q1))
			.attr('y', boxCenter - boxWidth / 2)
			.attr('width', x(data.q3) - x(data.q1))
			.attr('height', boxWidth);

		// Append the horizontal lines
		svg.selectAll('toto')
			.data([data.med])
			.enter()
			.append('line')
			.attr('x1', d => x(d))
			.attr('x2', d => x(d))
			.attr('y1', boxCenter - boxWidth / 2)
			.attr('y2', boxCenter + boxWidth / 2)
			.attr('stroke', 'orange')
			.attr('stroke-width', '2px');

		// Tooltip
		svg.selectAll('rect')
			.append('title')
			.text(() => {
				let prefix = 'Most of the applicants matched with\ntheir ' + getOrdinal(data.index) + ' choice ';
				if (data.index == 13)
					prefix = 'Most of the unmatched applicants\n';

				return prefix + 'have a lottery number\nstarting with a digit between '
					+ getHexDigits(data.min) + ' and ' + getHexDigits(data.max) + '.';
			});

		let studentPrefix = (index == 13) ? 'unmatched applicants' : 'applicants matched with their ' + getOrdinal(index) + ' choice';
		let binAtIndex = this.bins[index - 1];
		let medianAtIndex = binAtIndex[Math.floor(binAtIndex.length / 2)];

		this.spotlightHeading.innerText = 'Lottery numbers of ' + studentPrefix;
		this.spotlightPar1.innerText = 'Most of the ' + studentPrefix + ' have a lottery number starting with a digit between '
			+ getHexDigits(data.min) + ' and ' + getHexDigits(data.max) + '. Specifically, for half of them the first digit is between '
			+ getHexDigits(data.q1) + ' and ' + getHexDigits(data.q3) + '.';
		this.spotlightPar2.innerText = 'The median lottery number for ' + studentPrefix + ' starts with the digits '
			+ medianAtIndex.replaceAll('-', '').substring(0, 4).toUpperCase() + '.';

		this.spotlightSection.classList.remove('hide');
	}

	/**
	 * Plots a tree-map displaying the distribution of students based on the position of their matched school on the
	 * ranking. Students are grouped between those matched with schools ranked 1-5, those matched with schools
	 * ranked 6-12, and those unmatched.
	 */
	plotTreeMap() {
		// Clear any existing plots
		this.plotPrimary.innerText = '';

		// Set the dimensions of the graph
		const width = 600;
		const height = 500;

		// Map the data to the correct format
		let hierarchyData = [
			{ name: 'root', parent: null, value: null },
			{ name: '1-5', parent: 'root', value: null },
			{ name: '6-12', parent: 'root', value: null },
			{ name: 'Unmatched', parent: 'root', value: this.counts[12] }
		];
		this.counts.forEach((count, index) => {
			if (index < 12) {
				hierarchyData.push({
					name: getOrdinal(index + 1),
					parent: index < 5 ? '1-5' : '6-12',
					value: count
				});
			}
		});

		// Define color and opacity scales
		const color = d3.scaleOrdinal()
			.domain(['1-5', '6-12', 'root'])
			.range(['#4036ed', '#a4a3f1', '#edecfc']);

		const opacity = d3.scaleLinear()
			.domain([0, d3.max[this.counts]])
			.range([.5, 1]);

		// Append the SVG container
		const svg = d3.select('#plot-primary')
			.append('svg')
			.attr('width', width)
			.attr('height', height)
			.attr('viewBox', [0, 0, width, height]);

		// Define the hierarchy data
		const root = d3.stratify()
			.id(d => d.name)
			.parentId(d => d.parent)
			(hierarchyData);
		root.sum(d => d.value);

		// Create the tree map
		d3.treemap()
			.tile(d3.treemapBinary)
			.size([width, height])
			.padding(4)
			(root);

		// Add the rectangles
		svg.selectAll('rect')
			.data(root.leaves())
			.enter()
			.append('rect')
			.attr('x', d => d.x0)
			.attr('y', d => d.y0)
			.attr('width', d => d.x1 - d.x0)
			.attr('height', d => d.y1 - d.y0)
			.style('stroke', 'black')
			.style('fill', d => color(d.parent.data.name))
			.style('opacity', d => opacity(d.parent.data.value));

		// Add labels with name and value
		svg.selectAll('text')
			.data(root.leaves())
			.enter()
			.filter(d => d.data.value >= 500)
			.append('text')
			.attr('x', d => d.x0 + 5)    // +10 to adjust position (more right)
			.attr('y', d => d.y0 + 15)    // +20 to adjust position (lower)
			.text(d => d.data.name)
			.attr('font-size', '12px')
			.attr('font-weight', 600)
			.attr('fill', d => d.data.name === 'Unmatched' ? 'black' : 'white');

		svg.selectAll('vals')
			.data(root.leaves())
			.enter()
			.filter(d => d.data.value >= 500)
			.append('text')
			.attr('x', d => d.x0 + 5)
			.attr('y', d => d.y0 + 30)
			.text(d => d.data.value.toLocaleString())
			.attr('font-size', '12px')
			.attr('fill', d => d.data.name === 'Unmatched' ? 'black' : 'white');

		// Tooltip
		svg.selectAll('rect')
			.append('title')
			.text(d => {
				if (d.data.name === 'Unmatched')
					return d.data.value.toLocaleString() + ' applicants were unmatched';
				return d.data.value.toLocaleString() + ' applicants were matched to their ' + d.data.name.toLowerCase() + ' choice';
			});
	}

	/**
	 * Plots a histogram displaying the number of students matched with each position on their ranking,
	 * highlighting those matched with their `indexFocused`-th choice.
	 */
	plotCounts(indexFocused, firstDigit = '') {
		// Clear any existing plots
		if (indexFocused != undefined)
			this.plotPrimary.innerHTML = '';
		else
			document.getElementById('plot-tertiary').innerHTML = '';

		// Set the dimensions and margins of the graph
		const margin = { top: 30, right: 25, bottom: 25, left: 50 };
		const width = 600;
		const height = 400;

		// Map the data to the correct format
		let ratios;
		if (firstDigit === '')
			ratios = this.ratios;
		else {
			 let subsetStudents = [];
			// Select all students whose lottery number starts with the specified digit
			for (let e of this.studentInfo.entries()) {
				let studentId = e[0], studentLottery = e[1].get('lottery');
				if (studentLottery.substring(0, 1).toUpperCase() === firstDigit)
					subsetStudents.push(studentId);
			}

			ratios = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
			subsetStudents.map(id => this.matches.get(id).get('rank')).forEach(r => ratios[(r || 13) - 1] += 1);
			ratios = ratios.map(c => c / subsetStudents.length);
		}

		const ratiosMap = ratios.keys().toArray().map(k => { return { index: k + 1, ratio: ratios[k] } });

		const self = this;
		const clickHandler = idx => self.plotLotteryRange(idx);

		// Define the X axis scale
		let x = d3.scaleBand()
			.domain(d3.range(1, 14))
			.range([margin.left, width - margin.right])
			.padding(0.2);

		// Define the Y axis scale
		let y = d3.scaleLinear()
			.domain([0, d3.max(ratiosMap, d => d.ratio)])
			.range([height - margin.bottom, margin.top]);

		// Append the SVG container
		const svg = d3.select(indexFocused == undefined ? '#plot-tertiary' : '#plot-primary')
			.append('svg')
			.attr('width', width)
			.attr('height', height)
			.attr('viewBox', [0, 0, width, height]);

		// Add a rect for each bar
		svg.append('g')
			.selectAll()
			.data(ratiosMap)
			.join('rect')
			.attr('x', d => x(d.index))
			.attr('y', d => y(0))
			.attr('width', x.bandwidth())
			.attr('fill', d => (indexFocused == undefined || d.index === indexFocused) ? '#4036ed' : '#a4a3f1')
			.on('click', (e, d) => clickHandler(d.index))
			.transition()
			.duration(1000)
				.attr('y', d => y(d.ratio))
				.attr('height', d => y(0) - y(d.ratio));

		// Add the X axis
		svg.append('g')
			.attr('transform', 'translate(0,' + (height - margin.bottom) + ')')
			.call(d3.axisBottom(x).tickSizeOuter(0).tickFormat(x => x <= 12 ? x : 'U'))
			.attr('font-size', '14px');

		// Add the Y axis
		svg.append('g')
			.attr('transform', 'translate(' + margin.left + ',0)')
			.call(d3.axisLeft(y).tickFormat(y => (100 * y).toFixed(0) + '%'))
			.attr('font-size', '14px')
			.call(g => g.append('text')
				.attr('x', -margin.left)
				.attr('y', 15)
				.attr('fill', '#000000')
				.attr('text-anchor', 'start')
				.attr('font-size', '14px')
				.text('Matched students' + (indexFocused == undefined ? ' - First digit ' + firstDigit : '')));

		// Tooltip
		svg.selectAll('rect')
			.append('title')
			.text(d => {
				if (d.index === 13)
					return (100 * d.ratio).toFixed(0) + '% of the applicants were unmatched';
				return (100 * d.ratio).toFixed(0) + '% of the applicants were matched to their ' + getOrdinal(d.index) + ' choice';
			});
	}

	/**
	 * Updates the nutritional label with aggregate statistics computed on the outcome of the last executed simulation.
	 */
	refreshNutritionalLabel() {
		this.customOutcomeSection.classList.add('hide');

		if (userData.getValues().hasData) {
			const rank = this.matches.get(STUDENT_ID).get('rank');
			this.pCustomOutcome.nodeValue =
				(rank == undefined) ? 'you were unmatched' :
				getOrdinal(rank) + ' choice (' + userData.getSchoolByIndex(rank - 1).substring(9) + ')';

			this.customOutcomeSection.classList.remove('hide');
		}

		this.pTotalStudents.nodeValue = this.totalStudents.toLocaleString();
		this.pTotalSchools.nodeValue = this.totalSchools.toLocaleString();
		this.pTotalCapacity.nodeValue = this.totalCapacity.toLocaleString();
		this.pListLength.nodeValue = this.avgListLength.toFixed(1);

		this.pStudentsTop.nodeValue = (100 * this.ratios[0]).toFixed(0) + '% (' + this.counts[0].toLocaleString() + ')';
		this.pStudentsFive.nodeValue = (100 * this.ratioFive).toFixed(0) + '% (' + this.countFive.toLocaleString() + ')';
		this.pStudentsUnmatched.nodeValue = (100 * this.ratios[12]).toFixed(0) + '% (' + this.counts[12].toLocaleString() + ')';

		this.pSchoolsUnfilled.nodeValue = (100 * this.numSchoolsUnfilled / this.totalSchools).toFixed(0) + '% (' + this.numSchoolsUnfilled + ')';
		this.pSeatsUnfilled.nodeValue = this.seatsUnfilled.toLocaleString();

		this.focusSection.classList.add('hide');
		this.spotlightSection.classList.add('hide');
	}

	/**
	 * Updates the focus section with the provided data.
	 */
	refreshFocus(heading = '', number = '', subtitle = '', par1 = '', par2 = '', par3 = '', plot = '', index = -1) {
		this.focusHeading.innerText = heading;
		this.focusNumber.innerText = number;
		this.focusSubtitle.innerText = subtitle;
		this.focusPar1.innerText = par1;
		this.focusPar2.innerText = par2;
		this.focusPar3.innerText = par3;

		// Clear plots
		this.plotPrimary.innerHTML = '';
		this.plotSecondary.innerHTML = '';
		this.spotlightSection.classList.add('hide');

		if (plot === 'counts') {
			this.plotCounts(index);
			this.plotPrimary.scrollIntoView(true);
		} else if (plot === 'counts-cumulative') {
			this.plotTreeMap();
			this.plotPrimary.scrollIntoView(true);
		}

		this.focusSection.classList.remove('hide');
	}
}

window.addEventListener('load', () => {
	// Load the NYC school list and use it to populate the school choice datalist
	fetch('./resources/school_list.json')
		.then((response) => response.json())
		.then((json) => userData = new UserData(json));

	// Instantiate and start the animation of the lottery number examples
	const lotteryNums = new LotteryNumberExamples();
	lotteryNums.iterate();

	// Instantiate the expandable explainers
	const lotteryExplainers = new LotteryExplainers();
	lotteryExplainers.initializeListeners();

	document.getElementById('run-simulation').addEventListener('click', () => {
		document.getElementById('simulation-results').classList.add('hide');  // Hide simulation results (if applicable)
		document.getElementById('simulation-loader').classList.remove('hide');  // Display loader
		document.getElementById('plot-primary').innerHTML = '';
	});
});

window.addEventListener('py:all-done', () => {
	// The "Run simulation" button is disabled until this event is captured, meaning PyScript is loaded and ready to run.
	// This is needed since the click event listener for the button is in PyScript and will not work until it's fully loaded.
	document.getElementById('run-simulation').classList.remove('disabled');
});

window.addEventListener('py-backend-done', () => {
	if (window.py_error) {
		document.getElementById('simulation-loader').classList.add('hide');  // Hide loader
		return;
	}

	// After the simulation is executed, fetch the results from the global variables and initialize the visualizer.
	const visualizer =
		new DataVisualizer(window.py_students, window.py_schools, window.py_bins, window.py_matches, window.py_school_outcome);

	visualizer.refreshNutritionalLabel();
	visualizer.show();

	document.getElementById('simulation-loader').classList.add('hide');  // Hide loader
})

export { displayAlert, pyDoneEvent, userData };