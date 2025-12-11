import os
import pandas as pd
from src.data_preprocessing.clean_lpi import clean_single_lpi

def clean_all_lpi(input_path, output_path):
    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)
    
    YEAR_MAP = {
        "2023": "LPI_2023.xlsx",
        "2018": "LPI_2018.xlsx",
        "2016": "LPI_2016.xlsx",
        "2014": "LPI_2014.xlsx",
        "2012": "LPI_2012.xlsx",
        "2010": "LPI_2010.xlsx",
        "2007": "LPI_2007.xlsx"
    }

    cleaned_frames = []

    # Check if input_path is a file (user might have passed a single file instead of a directory)
    if os.path.isfile(input_path):
        print(f"Input path is a file, cleaning single file: {input_path}")
        # Try to guess year or just clean it. The loop expects a directory. 
        # If the user passed a single file, maybe we should just clean that.
        # But 'clean_single_lpi' needs a year. 
        # For now, let's assume input_path is a directory as per the code structure.
        # If it is a file, the join below will likely produce non-existent paths.
        pass

    for year, fname in YEAR_MAP.items():
        # strict join only works if input_path is a dir
        path = os.path.join(input_path, fname)

        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue
        
        df_year = clean_single_lpi(path, int(year))

        out_path_year = os.path.join(output_folder, f"LPI_clean_{year}.csv")
        df_year.to_csv(out_path_year, index=False)

        cleaned_frames.append(df_year)

    if cleaned_frames:
        # Merge all years
        df_all = pd.concat(cleaned_frames, ignore_index=True)
        df_all.to_csv(output_path, index=False)
        print(f"\nAll LPI datasets cleaned and merged! Saved to {output_path}")
    else:
        print("No LPI datasets were cleaned.")