# DataLife
A current research project partnership between NYU, Politecnico di Milano, and Rutgers aimed at improving NYC highschool admissions efficiency and transparency.

## Tour of Important Documents:
- FrontEnd
  - index.html = main page
    - style.css = visual definitions for the whole frontend
  - script.js = takes the results of the simulation and generates the plots
- BackEnd:
  - oneshot.py = main simulation code which creates the student/school objects/rankings
    - load.ipynb = playground for oneshot.py
  - gale_shapley.py = main matching code that returns which students are admitted into which schools
  - application_logic.py = calls oneshot function + gale-shapely matching and passes results to frontend

## Simulation Information: (variable configurations for each stage)
- Student Strategy = [a,b] : a = student.selection_policy, b = student.ranking_policy (defined around oneshot.py lines 50-54)
  - stage 1 = [0,0] = students choose schools randomly with no order
  - stage 2 = [0,1] = students choose schools randomly but order by the school's likeability index, descending
  - stage 3 = [1,0] = students choose schools via popularity-weighted selection and placed in a randomized order
  - stage 4 = [1,1] = students choose schools via popularity-weighted selection and rank them based on the likeability index, descending
  - stage 5+ = [randint(0,1),randint(0,1)]
- School Strategy = [c] : c = school.policy (defined around oneshot.py line 113)
  - stages 1-5 = [1] = school sorts students that applied based on lottery number descending
  - stage 6 = [2] = school sorts students that applied based on seat number descending with lottery number as a tiebreaker
  - stage 7 = [3] = school sorts students that applied based on screen number descending with lottery number as a tiebreaker
  - stage 8+ = randint(1,3)
 - Student List size [d] : d = student.num_schools = len(student.schools) (defined around oneshot.py line 59)
   - stage 1-8 = [12]
   - stage 9+ = [self.get_num_pick()] = normal distribution selection with mean 7 & std 2, cast to int, bounded 1 to 12 (defined around oneshot.py line 73)
> Note: By default, the simulation has random student policy and random school policy with random student list, aka stage 9. Also, the seat and screen distributions are defined in oneshot.py lines 28-29 and implemented in lines 199-208. Check the official write-up for more details.

## Requirements:
- uuid(1.30) for lottery numbers
- random, numpy(1.23.1), and pandas(1.4.3) for data handling
- SequenceMatcher from difflib(0.3.5)
> Note: this probably works with most versions but these are what we are currently using

Optional imports:
- for the load.ipynb visualizations: plotly express (5.9.0) & matplotlib (3.6.0)
- for transparency: tqdm(4.64.1)
- additional distance metrics for school matching: editdistance(0.8.1)


## Links:
- Code: https://github.com/KoraSHughes/DataLife
- Drive: https://drive.google.com/drive/u/2/folders/1Yb50q6S4WFuu1fUl2jDsDsLQXnBjJukL
    - Timeline: https://docs.google.com/document/d/1pvjEKBPctUSVmkJagOqbjg75urHlMuKdKaXYgLeUFzA/edit#heading=h.m193mcqm81xi


## Reference for Generating Simulated Student's Data:
- School Admissions Main Data: https://infohub.nyced.org/reports/government-reports/student-applications-admissions-and-offers
- Admissions Outcomes Data: https://infohub.nyced.org/reports/admissions-and-enrollment/fall-2023-admissions-outcomes
- MiddleSchool Demographics Data: https://data.cityofnewyork.us/Education/2017-18-2021-22-Demographic-Snapshot/c7ru-d68s/about_data
- Test Scores Data: https://infohub.nyced.org/reports/academics/test-results
- Reference Github: https://github.com/yehudagale/NYCSchoolDataGenerator

## Running the web server:

- Using VSCode: install the **Live Server** extension, then right-click on index.html and select 'Open with Live Server' from the context menu.

- Using other IDEs: open a terminal from the FrontEnd directory and run `python3 -m http.server`, then navigate to `localhost:8000`. If the simulation does not work properly, run the command in a terminal whose working directory is the project root: `localhost:8000` will show the directory listing, navigate to `/FrontEnd/index.html` to display the website.


# Description
A nutritional label for NYC high-school admissions: Explaining ranking and matching to middle school students.

The goal of this project is to develop an interactive simulation of two-sided matching [1], grounded in a specific use case – high-school admissions in New York City [2-4]. The simulation will be accompanied by an adaptive “nutritional label” [5,6] that explains results of the simulation, and highlights changes when simulation parameters change, to applicants – middle-school students and their families.

The project will be supervised by Prof. Julia Stoyanovich, Associate Professor of Computer Science and Engineering, Associate Professor of Data Science, and Director of the Center for Responsible AI at New York University and Prof. Amélie Marian, Associate Professor of Computer Science at Rutgers, the State University of New Jersey.

The project will be carried out in partnership with the New York Hall of Science (NYSCI), an educational institution in Queens, NY that described itself as “a hub for researchers, developers, and producers in education, to create and study new ways of delivering equitable learning opportunities that are anchored in our Design Make Play approach to learning and STEM engagement.”  NYSCI develops and hosts full-scale science exhibits as well as hands-on smaller-scale interactive activities that aim to teach visitors - primarily children of different ages - about a range of Science, Technology, Engineering, and Mathematics (STEM) topics.

Project participants – graduate students in data science and design from NYU and Politecnico di Milano – will collaboratively implement the Gale-Shapley Stable Marriage matching algorithm.  They will implement a graphical user interface that allows applicants – middle-school students and their families – to run the simulation according to specified parameters and understand the results.  To make the user interface understandable to applicants and their families, the team will engage in collaborative design and evaluation activities with the “customer” – NYSCI staff with experience in educating middle-school students and, if appropriate IRB permissions are received on time - with a sample of middle-school students who attend NYSCI.

Supplementary reading
https://en.wikipedia.org/wiki/Stable_marriage_problem
https://www.schools.nyc.gov/enrollment/enroll-grade-by-grade/how-students-get-offers-to-doe-public-schools
“Decoding the NYC admissions lottery numbers”, Amelie Marian, Medium (2021-23), https://medium.com/algorithms-in-the-wild/decoding-the-nyc-school-admission-lottery-numbers-bae7148e337d
“Algorithmic Transparency and Accountability through Crowdsourcing: A Study of the NYC School Admission Lottery“, Amelie Marian, ACM FAccT (2023)  https://dl.acm.org/doi/10.1145/3593013.3594009
“Nutritional labels for data and models”, Stoyanovich & Howe, IEEE Data Engineering Bulletin (2019),  http://sites.computer.org/debull/A19sept/p13.pdf
“Introducing contextual transparency for automated decision systems”, Sloane, Solano-Kamaiko, Yuan, Dasgupta, Stoyanovich, Nature Machine Intelligence (2023) https://doi.org/10.1038/s42256-023-00623-7
Project team:
NYU supervisor: Prof. Julia Stoyanovich
Collaborating partner: Dorothy Benett, New York Hall of Science
Collaborating partner: Prof. Amelie Marian, Rutgers University, computer scientist with domain expertise in NYC public school admissions
Graduate students: 2 from computer science / data science (NYU) + 2 from design / human computer interaction (PoliMi)

## Development team

- Tommaso Bonetti (Computer Science, Politecnico di Milano)
- Kora Stewart Hughes (Computer Science, New York University)
- Martina Balducci (Communication Design, Politecnico di Milano)