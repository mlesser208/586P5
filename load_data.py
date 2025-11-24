import pandas as pd
import os


def load_airbnb_data(file_path="listings.csv.gz"):
    """
    Purpose: Loads and displays basic information about the Airbnb listings dataset.
    
    Parameters:
        file_path (str): Path to the compressed CSV file containing Airbnb listings data.
                        Defaults to "listings.csv.gz" in the current directory.
                        Expected to be a gzip-compressed CSV file.
    
    Return Value:
        pandas.DataFrame: A DataFrame containing the Airbnb listings data.
    
    Exceptions:
        FileNotFoundError: Raised if the specified file path does not exist.
        pd.errors.EmptyDataError: Raised if the CSV file is empty.
        pd.errors.ParserError: Raised if the CSV file cannot be parsed correctly.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    airbnb_df = pd.read_csv(file_path, compression='gzip')
    
    print("Airbnb Listings Dataset:")
    print(f"Columns: {list(airbnb_df.columns)}")
    print(f"Number of rows: {len(airbnb_df)}")
    print()
    
    return airbnb_df


def load_lahd_data(file_path="LAHD_Affordable_Housing_Projects_Catalog_And_Listing_20251124.csv"):
    """
    Purpose: Loads and displays basic information about the LAHD affordable housing dataset.
    
    Parameters:
        file_path (str): Path to the CSV file containing LAHD affordable housing data.
                        Defaults to "LAHD_Affordable_Housing_Projects_Catalog_And_Listing_20251124.csv"
                        in the current directory. Expected to be a standard CSV file.
    
    Return Value:
        pandas.DataFrame: A DataFrame containing the LAHD affordable housing data.
    
    Exceptions:
        FileNotFoundError: Raised if the specified file path does not exist.
        pd.errors.EmptyDataError: Raised if the CSV file is empty.
        pd.errors.ParserError: Raised if the CSV file cannot be parsed correctly.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    lahd_df = pd.read_csv(file_path)
    
    print("LAHD Affordable Housing Dataset:")
    print(f"Columns: {list(lahd_df.columns)}")
    print(f"Number of rows: {len(lahd_df)}")
    print()
    
    return lahd_df


if __name__ == "__main__":
    # Load and display Airbnb data
    airbnb_df = load_airbnb_data()
    
    # Load and display LAHD data
    lahd_df = load_lahd_data()

