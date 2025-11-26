"""
combine_housing_data.py

Option A for SSCI 586 Project 5:

- Load Airbnb listings (InsideAirbnb LA)

- Load LAHD affordable housing projects

- Standardize into one schema

- Export combined CSV for later geocoding / mapping

Run this with your ArcGIS Pro Python or any Python 3 environment with pandas installed.
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# CONFIG – CHANGE THESE FOR YOUR COMPUTER
# ---------------------------------------------------------

# Path to InsideAirbnb listings file (Los Angeles)
AIRBNB_PATH = r"listings.csv.gz"

# Path to LAHD Affordable Housing Projects CSV
LAHD_PATH = r"LAHD_Affordable_Housing_Projects_Catalog_And_Listing_20251124.csv"

# Output folder + file name
OUTPUT_FOLDER = r"project5_outputs"
OUTPUT_FILENAME = "combined_housing_west_la.csv"

# Tag everything with your study area
STUDY_AREA = "West LA"   # or "Redondo Beach" if you switch later

# ---------------------------------------------------------
# Helper: make sure output folder exists
# ---------------------------------------------------------

def ensure_output_folder(folder_path: str) -> Path:
    """
    Purpose: Creates the output folder if it doesn't exist.
    
    Parameters:
        folder_path (str): Path to the folder that should exist.
                          Can be a relative or absolute path.
    
    Return Value:
        Path: A Path object representing the created/existing folder.
    
    Exceptions:
        PermissionError: Raised if the folder cannot be created due to insufficient permissions.
        OSError: Raised if there are other filesystem-related errors.
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder

# ---------------------------------------------------------
# Loader 1: InsideAirbnb listings
# ---------------------------------------------------------

def load_airbnb(listings_path: str) -> pd.DataFrame:
    """
    Purpose: Load InsideAirbnb LA listings and map to common schema.
    
    Parameters:
        listings_path (str): Path to the gzip-compressed CSV file containing Airbnb listings.
                            Expected columns: id, name, latitude, longitude, price (string like "$125.00"),
                            bedrooms (optional), bathrooms (optional).
    
    Return Value:
        pd.DataFrame: A DataFrame with standardized columns including source, listing_id, name,
                     address, city, state, zip, price, beds, baths, sqft, lat, lon, url,
                     study_area, and full_address.
    
    Exceptions:
        FileNotFoundError: Raised if the listings_path file does not exist.
        pd.errors.EmptyDataError: Raised if the CSV file is empty.
        pd.errors.ParserError: Raised if the CSV file cannot be parsed correctly.
        KeyError: Raised if required columns (id, latitude, longitude) are missing from the dataset.
    """
    print(f"Loading Airbnb listings from: {listings_path}")
    df_raw = pd.read_csv(listings_path, compression="gzip")

    # Quick sanity check – you can comment these out later
    print(f"Airbnb rows: {len(df_raw)}")
    # print(df_raw.columns)  # uncomment once to inspect

    df = pd.DataFrame()

    df["source"] = "airbnb"
    df["listing_id"] = df_raw["id"].astype(str)
    df["name"] = df_raw.get("name")

    # Airbnb often doesn't give full street address in the public dataset,
    # so we'll leave address-related fields blank for now.
    df["address"] = None
    df["city"] = "Los Angeles"
    df["state"] = "CA"
    df["zip"] = None

    # Price like "$125.00" → 125.0
    if "price" in df_raw.columns:
        price_clean = (
            df_raw["price"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
        )
        df["price"] = pd.to_numeric(price_clean, errors="coerce")
    else:
        df["price"] = None

    # Bedrooms / bathrooms if present
    df["beds"] = df_raw.get("bedrooms")
    df["baths"] = df_raw.get("bathrooms")
    df["sqft"] = None  # Airbnb dataset usually doesn't include this

    # Coordinates
    df["lat"] = df_raw.get("latitude")
    df["lon"] = df_raw.get("longitude")

    # Construct a simple Airbnb URL if id exists
    df["url"] = "https://www.airbnb.com/rooms/" + df["listing_id"]

    df["study_area"] = STUDY_AREA
    df["full_address"] = None  # we don't really have it for Airbnb

    return df

# ---------------------------------------------------------
# Loader 2: LAHD Affordable Housing Projects
# ---------------------------------------------------------

def load_lahd(lahd_path: str) -> pd.DataFrame:
    """
    Purpose: Load LAHD affordable housing projects CSV and map to common schema.
    
    Parameters:
        lahd_path (str): Path to the CSV file containing LAHD affordable housing data.
                        Expected columns may include: NAME (project name), ADDRESS or ahtf_pa
                        (project address), COMMUNITY (neighborhood), PROJECT NUMBER, ZIP, etc.
                        The function will attempt to find columns with various name variations.
    
    Return Value:
        pd.DataFrame: A DataFrame with standardized columns including source, listing_id, name,
                     address, city, state, zip, price, beds, baths, sqft, lat, lon, url,
                     study_area, and full_address. Note that lat/lon will be None as they
                     need to be geocoded later from full_address.
    
    Exceptions:
        FileNotFoundError: Raised if the lahd_path file does not exist.
        pd.errors.EmptyDataError: Raised if the CSV file is empty.
        pd.errors.ParserError: Raised if the CSV file cannot be parsed correctly.
    """
    print(f"Loading LAHD projects from: {lahd_path}")
    df_raw = pd.read_csv(lahd_path)

    print(f"LAHD rows: {len(df_raw)}")
    # print(df_raw.columns)  # uncomment once to see exact names

    df = pd.DataFrame()
    df["source"] = "lahd_affordable"

    # listing_id: use PROJECT NUMBER if it exists, otherwise fallback to index
    if "PROJECT NUMBER" in df_raw.columns:
        df["listing_id"] = df_raw["PROJECT NUMBER"].astype(str)
    elif "project_number" in df_raw.columns:
        df["listing_id"] = df_raw["project_number"].astype(str)
    else:
        df["listing_id"] = df_raw.index.astype(str)

    # Name field
    name_col = None
    for cand in ["NAME", "name", "Project Name"]:
        if cand in df_raw.columns:
            name_col = cand
            break
    df["name"] = df_raw.get(name_col)

    # Address field – LA Open Data uses API name ahtf_pa for ADDRESS
    addr_col = None
    for cand in ["ADDRESS", "ahtf_pa", "address", "Official Address", "officialaddress"]:
        if cand in df_raw.columns:
            addr_col = cand
            break
    df["address"] = df_raw.get(addr_col)

    # City/state – these are all in LA city
    df["city"] = "Los Angeles"
    df["state"] = "CA"

    # Zip – might not be present; if you see one like "ZIP" or "zip", add it here
    zip_col = None
    for cand in ["ZIP", "Zip", "zip", "zipcode"]:
        if cand in df_raw.columns:
            zip_col = cand
            break
    df["zip"] = df_raw.get(zip_col)

    # No explicit rent price here; it's more like project cost, so leave price blank
    df["price"] = None
    df["beds"] = None
    df["baths"] = None
    df["sqft"] = None

    # No lat/lon yet – we will geocode later
    df["lat"] = None
    df["lon"] = None

    # No specific URL per project
    df["url"] = None

    df["study_area"] = STUDY_AREA

    # Build full address string for geocoding in a later step
    df["full_address"] = (
        df["address"].fillna("").astype(str)
        + ", "
        + df["city"]
        + ", "
        + df["state"]
        + " "
        + df["zip"].fillna("").astype(str)
    ).str.strip(", ")

    return df

# ---------------------------------------------------------
# Combine + export
# ---------------------------------------------------------

def build_combined_dataset() -> Path:
    """
    Purpose: Combines Airbnb and LAHD datasets into a single standardized CSV file.
    
    Parameters:
        None (uses module-level constants: AIRBNB_PATH, LAHD_PATH, OUTPUT_FOLDER, OUTPUT_FILENAME)
    
    Return Value:
        Path: A Path object representing the output CSV file that was created.
    
    Exceptions:
        FileNotFoundError: Raised if AIRBNB_PATH or LAHD_PATH files do not exist.
        pd.errors.EmptyDataError: Raised if either input CSV file is empty.
        pd.errors.ParserError: Raised if either CSV file cannot be parsed correctly.
        PermissionError: Raised if the output file cannot be written due to insufficient permissions.
        OSError: Raised if there are other filesystem-related errors when writing the output file.
    """
    airbnb_df = load_airbnb(AIRBNB_PATH)
    lahd_df = load_lahd(LAHD_PATH)

    combined_df = pd.concat([airbnb_df, lahd_df], ignore_index=True)
    print(f"Combined rows: {len(combined_df)}")

    out_folder = ensure_output_folder(OUTPUT_FOLDER)
    out_path = out_folder / OUTPUT_FILENAME

    combined_df.to_csv(out_path, index=False)
    print(f"\nWrote combined dataset to:\n{out_path}")

    return out_path

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    build_combined_dataset()

