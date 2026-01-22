import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(file_path):
    """
    Reads Excel data from the given file path and returns a pandas DataFrame.
    """
    data = pd.read_excel(file_path)
    return data



df = read_data('data/master_data_03_zip_code.xlsx')
print(len(df))