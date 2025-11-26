"""
visualize_housing_map.py

Creates an interactive geospatial map visualization of housing units in Los Angeles.
Displays all units with valid coordinates and shows metadata on hover.

Run this with Python 3 and required packages: pandas, folium
"""

import pandas as pd
from pathlib import Path
import folium
from folium.plugins import MarkerCluster

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

# Input CSV file
INPUT_CSV = r"project5_outputs\combined_housing_west_la.csv"

# Output HTML map file
OUTPUT_HTML = r"project5_outputs\housing_map.html"

# Los Angeles center coordinates (approximate center of LA County)
LA_CENTER_LAT = 34.0522
LA_CENTER_LON = -118.2437

# Initial zoom level
INITIAL_ZOOM = 11

# ---------------------------------------------------------
# Helper: Format metadata for hover tooltip
# ---------------------------------------------------------

def format_popup_html(row: pd.Series) -> str:
    """
    Purpose: Formats a row of housing data into an HTML popup string for map click display.
             Uses simplified format to reduce HTML file size.
    
    Parameters:
        row (pd.Series): A pandas Series representing a single housing unit row with columns:
                        name, source, price, beds, baths, address, city, zip, url, listing_id.
    
    Return Value:
        str: An HTML-formatted string containing the unit's metadata, with missing values
             handled gracefully (shown as "N/A" or omitted).
    
    Exceptions:
        KeyError: Raised if required columns are missing from the row (though the function
                 uses .get() to handle missing keys gracefully).
    """
    tooltip_parts = []
    
    # Name
    if pd.notna(row.get("name")) and str(row.get("name")).strip():
        tooltip_parts.append(f"<b>{row['name']}</b>")
    
    # Source
    source = row.get("source", "Unknown")
    tooltip_parts.append(f"<i>Source: {source}</i>")
    
    # Price
    if pd.notna(row.get("price")) and row.get("price") != "":
        tooltip_parts.append(f"Price: ${row['price']:.2f}")
    
    # Beds/Baths
    bed_bath = []
    if pd.notna(row.get("beds")):
        bed_bath.append(f"{int(row['beds'])} bed(s)")
    if pd.notna(row.get("baths")):
        bed_bath.append(f"{row['baths']} bath(s)")
    if bed_bath:
        tooltip_parts.append(", ".join(bed_bath))
    
    # Address
    address_parts = []
    if pd.notna(row.get("address")) and str(row.get("address")).strip():
        address_parts.append(str(row["address"]))
    if pd.notna(row.get("city")) and str(row.get("city")).strip():
        address_parts.append(str(row["city"]))
    if pd.notna(row.get("zip")) and str(row.get("zip")).strip():
        address_parts.append(str(row["zip"]))
    if address_parts:
        tooltip_parts.append("Address: " + ", ".join(address_parts))
    
    # URL (if available)
    if pd.notna(row.get("url")) and str(row.get("url")).strip():
        tooltip_parts.append(f"<a href='{row['url']}' target='_blank'>View Listing</a>")
    
    return "<br>".join(tooltip_parts)

# ---------------------------------------------------------
# Main visualization function
# ---------------------------------------------------------

def create_housing_map(csv_path: str, output_path: str) -> None:
    """
    Purpose: Creates an interactive HTML map visualization of housing units from a CSV file.
             Only displays units with valid latitude/longitude coordinates. Uses marker clustering
             for performance with large datasets.
    
    Parameters:
        csv_path (str): Path to the input CSV file containing housing data with columns:
                       lat, lon, name, source, price, beds, baths, address, city, zip, url, listing_id.
        output_path (str): Path where the output HTML map file will be saved.
    
    Return Value:
        None: The function writes an HTML file to disk but does not return a value.
    
    Exceptions:
        FileNotFoundError: Raised if the csv_path file does not exist.
        pd.errors.EmptyDataError: Raised if the CSV file is empty.
        pd.errors.ParserError: Raised if the CSV file cannot be parsed correctly.
        KeyError: Raised if required columns (lat, lon) are missing from the dataset.
        PermissionError: Raised if the output file cannot be written due to insufficient permissions.
        OSError: Raised if there are other filesystem-related errors when writing the output file.
    """
    print(f"Loading housing data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total rows loaded: {len(df)}")
    
    # Filter to rows with valid coordinates
    df_valid = df[
        df["lat"].notna() & 
        df["lon"].notna() & 
        (df["lat"] != 0) & 
        (df["lon"] != 0)
    ].copy()
    
    print(f"Rows with valid coordinates: {len(df_valid)}")
    
    if len(df_valid) == 0:
        print("ERROR: No rows with valid coordinates found!")
        return
    
    # Create base map centered on Los Angeles
    print("Creating map...")
    m = folium.Map(
        location=[LA_CENTER_LAT, LA_CENTER_LON],
        zoom_start=INITIAL_ZOOM,
        tiles="OpenStreetMap"
    )
    
    # Prepare data for efficient marker creation
    print("Preparing marker data...")
    
    # Create separate clusters for different sources for better organization
    airbnb_cluster = MarkerCluster(name="Airbnb Listings", overlay=True, control=True).add_to(m)
    lahd_cluster = MarkerCluster(name="LAHD Affordable Housing", overlay=True, control=True).add_to(m)
    other_cluster = MarkerCluster(name="Other", overlay=True, control=True).add_to(m)
    
    # Add markers in batches for better performance
    print("Adding markers (this may take a moment)...")
    marker_count = 0
    
    for idx, row in df_valid.iterrows():
        # Get simple tooltip text (just name for hover)
        tooltip_text = str(row.get("name", f"Unit {row.get('listing_id', idx)}"))[:50]  # Limit length
        
        # Create popup HTML only when clicked (more efficient)
        popup_html = format_popup_html(row)
        
        # Determine marker color and cluster based on source
        source = str(row.get("source", "")).lower() if pd.notna(row.get("source")) else ""
        if "airbnb" in source:
            color = "blue"
            cluster = airbnb_cluster
        elif "lahd" in source or "affordable" in source:
            color = "green"
            cluster = lahd_cluster
        else:
            color = "gray"
            cluster = other_cluster
        
        # Use CircleMarker for better performance with large datasets
        # It's lighter weight than full Marker
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip_text,
            color=color,
            fillColor=color,
            fillOpacity=0.6,
            weight=1
        ).add_to(cluster)
        
        marker_count += 1
        if marker_count % 5000 == 0:
            print(f"  Added {marker_count} markers...")
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    print(f"Saving map to: {output_path}")
    m.save(output_path)
    
    print("\nMap created successfully.")
    print(f"Open {output_path} in your web browser to view the interactive map.")
    print("\nMap features:")
    print(f"  - {len(df_valid)} housing units displayed")
    print("  - Hover over markers to see unit details")
    print("  - Click markers for full popup with metadata")
    print("  - Marker clusters group nearby units for better performance")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = Path(OUTPUT_HTML).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_housing_map(INPUT_CSV, OUTPUT_HTML)

