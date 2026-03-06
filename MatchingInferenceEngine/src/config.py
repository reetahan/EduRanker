import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')

EXP_OUT_FOLDER = os.getenv('EXP_OUT_FOLDER', '/scratch/rm6609/EduRanker/experiment-results/')
DATA_GENERATION_SEED = int(os.getenv('DATA_GENERATION_SEED', '44'))