import pandas as pd
import chardet

# def detect_encoding(file_path):
#     with open(file_path, 'rb') as f:
#         raw = f.read(5000)
    
#     return chardet.detect(raw)['encoding']

import pandas as pd

def load_with_fallback(path):
    encodings = ["utf-8", "latin1", "cp1252"]

    last_error = None
    
    for enc in encodings:
        try:
            print(f"Trying encoding: {enc}")
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"Loaded successfully with encoding: {enc}")
            return df
        except Exception as e:
            print(f"Failed with {enc}: {e}")
            last_error = e
            continue

    raise ValueError(f"All encodings failed. Last error: {last_error}")


def clean_trade(input_path, output_path):
    print("Detecting File Encoding")
    df = load_with_fallback(input_path)
    # print(f"Detected Encoding: {encoding}")

    # if encoding is None:
    #     print("Failed to detect encoding. Defaulting to latin1.")
    #     encoding = 'latin1'

    # df = pd.read_csv(input_path, low_memory = False)
    print(df.head())

    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex = False)

    numeric_cols = ["primaryValue", "fobvalue", "cifvalue", "netWgt", "grossWgt", "qty", "altQty"]
    text_cols = ["reporterISO", "partnerISO", "reporterDesc", "partnerDesc", "cmdCode", "cmdDesc", "motDesc"]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.encode("ascii", "ignore").str.decode("ascii")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace(["..", "---", "--", "-", "N/A", "na", ""], None).apply(lambda x: str(x).replace(",", "") if isinstance(x, str) else x)
            df[col] = pd.to_numeric(df[col], errors = "coerce")

    print("Dropping fully empty rows...")
    df.dropna(how="all", inplace=True)

    print("Removing rows where both reporter & partner are missing...")
    df = df[df["reporterISO"].notna() & df["partnerISO"].notna()]

    print("Ensuring year is numeric...")
    if "refYear" in df.columns:
        df["refYear"] = pd.to_numeric(df["refYear"], errors="coerce")

    print("Saving cleaned file as UTF-8...")
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Cleaning complete! Saved to: {output_path}")
    return df
    


if __name__ == "__main__":
    input_path = r"C:\Users\KIIT0001\ai_engine\data\raw\TradeData_11_30_2025_20_59_49.csv"
    output_path = r"C:\Users\KIIT0001\ai_engine\data\processed\TradeData_11_30_2025_20_59_49.csv"
    clean_trade(input_path, output_path)