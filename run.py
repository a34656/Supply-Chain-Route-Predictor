from src.utils.import_helper import setup_imports
setup_imports()

from src.data_preprocessing.clean_wgi import preprocess_wgi
from src.data_preprocessing.clean_trade import clean_trade
from src.data_preprocessing.clean_lpi_all import clean_all_lpi

def main():
    # preprocess_wgi(input_path="data/raw/wgi.csv", output_path="data/processed/wgi_clean.csv")
    # clean_trade(input_path="data/raw/TradeData_11_30_2025_20_59_49.csv", output_path="data/processed/TradeData.csv")
    clean_all_lpi(input_path="data/raw/International_LPI_from_2007_to_2023_0.xlsx", output_path="data/processed/lpi_clean.csv")

if __name__ == "__main__":
    main()