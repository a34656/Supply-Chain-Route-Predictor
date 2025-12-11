import os
import pandas as pd

from src.utils.import_helper import setup_imports
setup_imports()

def preprocess_wgi(input_path, output_path):
    """
    Clean the World Governance Indicators dataset.
    Keeps Series Code for pivot, removes Series Name later,
    and saves metadata + cleaned file to a destination directory.
    """
    df = pd.read_csv(input_path)

    # if "series_name" in df.columns:
    #     df = df.drop(columns = "series_name")

    year_cols = [col for col in df.columns if '[YR' in col]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors = 'coerce')

    df['Indicator_mean'] = df[year_cols].mean(axis = 1)

    df = df[['Country Code', 'Series Code', 'Indicator_mean']]

    final = df.pivot_table(index = 'Country Code', columns = 'Series Code', values = 'Indicator_mean').reset_index()
    final.to_csv(output_path, index = False)
    return final

if __name__ == "__main__":
    input_path = r"C:\Users\KIIT0001\ai_engine\data\raw\wgi.csv"
    output_path = r"C:\Users\KIIT0001\ai_engine\data\processed\wgi.csv"
    preprocess_wgi(input_path, output_path)
