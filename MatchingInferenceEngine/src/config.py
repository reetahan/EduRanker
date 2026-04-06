import os
from pathlib import Path
from dotenv import load_dotenv

# Project root for MatchingInferenceEngine (where .env lives)
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / '.env')

EXP_OUT_FOLDER = os.getenv('EXP_OUT_FOLDER', '/scratch/rm6609/EduRanker/MatchingInferenceEngine/experiment-results/')
RAW_DATA_DIR = os.getenv(
    'RAW_DATA_DIR',
    '/scratch/rm6609/EduRanker/MatchingInferenceEngine/sample-data/raw-data'
)
POLISHED_DATA_DIR = os.getenv(
    'POLISHED_DATA_DIR',
    '/scratch/rm6609/EduRanker/MatchingInferenceEngine/sample-data/data'
)
CHILEAN_DATA_DIR = os.getenv(
    'CHILEAN_DATA_DIR',
    '/scratch/rm6609/EduRanker/MatchingInferenceEngine/sample-data/data/chilean_data_processed'
)
DATA_GENERATION_SEED = int(os.getenv('DATA_GENERATION_SEED', '44'))