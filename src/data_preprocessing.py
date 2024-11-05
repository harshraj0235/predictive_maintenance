import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Perform any necessary preprocessing steps here
    # For demonstration purposes, we'll just return the data as is
    return data