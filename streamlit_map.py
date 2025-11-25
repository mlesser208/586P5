"""
streamlit_map.py

Streamlit web app for visualizing housing units in Los Angeles.
Deploy to Streamlit Cloud for free hosting.

Run locally: streamlit run streamlit_map.py
Deploy: Push to GitHub and connect to streamlit.io
"""

import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path

# Page config
st.set_page_config(
    page_title="LA Housing Map",
    page_icon="🏠",
    layout="wide"
)

# Input CSV file (use Path for cross-platform compatibility)
INPUT_CSV = Path("project5_outputs") / "combined_housing_west_la.csv"

# Los Angeles center coordinates
LA_CENTER_LAT = 34.0522
LA_CENTER_LON = -118.2437

@st.cache_data
def load_data():
    """
    Purpose: Loads and caches housing data from CSV file.
    
    Parameters:
        None (uses module-level INPUT_CSV constant)
    
    Return Value:
        pd.DataFrame: DataFrame with valid coordinates filtered.
    
    Exceptions:
        FileNotFoundError: Raised if INPUT_CSV file does not exist.
        pd.errors.EmptyDataError: Raised if CSV is empty.
        KeyError: Raised if required columns (lat, lon) are missing.
    """
    csv_path = str(INPUT_CSV)
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Check for required columns
    if "lat" not in df.columns or "lon" not in df.columns:
        raise KeyError(f"Required columns 'lat' and/or 'lon' not found. Available columns: {list(df.columns)}")
    
    # Filter to rows with valid coordinates
    df_valid = df[
        df["lat"].notna() & 
        df["lon"].notna() & 
        (df["lat"] != 0) & 
        (df["lon"] != 0)
    ].copy()
    
    # Ensure 'source' column is string type for string operations
    # Fill NaN values with empty string before converting to string
    if "source" in df_valid.columns:
        df_valid["source"] = df_valid["source"].fillna("").astype(str)
    
    return df_valid

def format_popup_html(row: pd.Series) -> str:
    """
    Purpose: Formats a row of housing data into an HTML popup string for map click display.
    
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

def create_map(df_valid: pd.DataFrame):
    """
    Purpose: Creates a Folium map with markers for all housing units.
    
    Parameters:
        df_valid (pd.DataFrame): DataFrame containing housing units with valid coordinates.
                                Expected columns: lat, lon, name, source, price, beds, baths,
                                address, city, zip, url, listing_id.
    
    Return Value:
        folium.Map: A Folium map object with markers and clusters added.
    
    Exceptions:
        KeyError: Raised if required columns (lat, lon) are missing from the DataFrame.
    """
    m = folium.Map(
        location=[LA_CENTER_LAT, LA_CENTER_LON],
        zoom_start=11,
        tiles="OpenStreetMap"
    )
    
    # Create clusters for different sources
    airbnb_cluster = MarkerCluster(name="Airbnb Listings", overlay=True, control=True).add_to(m)
    lahd_cluster = MarkerCluster(name="LAHD Affordable Housing", overlay=True, control=True).add_to(m)
    other_cluster = MarkerCluster(name="Other", overlay=True, control=True).add_to(m)
    
    # Add markers
    for idx, row in df_valid.iterrows():
        tooltip_text = str(row.get("name", f"Unit {row.get('listing_id', idx)}"))[:50]
        popup_html = format_popup_html(row)
        
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
    
    folium.LayerControl().add_to(m)
    return m

# Main app
st.title("🏠 Los Angeles Housing Units Map")
st.markdown("Interactive map of Airbnb listings and affordable housing projects in Los Angeles")

# Load data
with st.spinner("Loading housing data..."):
    try:
        df_valid = load_data()
        if len(df_valid) == 0:
            st.error("⚠️ No data loaded! Please check that the CSV file exists and contains valid data.")
            st.stop()
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {INPUT_CSV}")
        st.error("Please ensure the CSV file exists in the project5_outputs folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Sidebar stats
st.sidebar.header("📊 Statistics")
st.sidebar.metric("Total Units", f"{len(df_valid):,}")

# Check if source column exists and calculate counts safely
if "source" in df_valid.columns:
    airbnb_count = len(df_valid[df_valid["source"].str.contains("airbnb", case=False, na=False)])
    lahd_count = len(df_valid[df_valid["source"].str.contains("lahd", case=False, na=False)])
else:
    airbnb_count = 0
    lahd_count = 0
    st.warning("⚠️ 'source' column not found in data")

st.sidebar.metric("Airbnb Listings", f"{airbnb_count:,}")
st.sidebar.metric("Affordable Housing", f"{lahd_count:,}")

# Debug info (expandable)
with st.sidebar.expander("🔧 Debug Info"):
    st.write(f"**Columns:** {', '.join(df_valid.columns.tolist())}")
    st.write(f"**Data shape:** {df_valid.shape}")
    if "lat" in df_valid.columns and "lon" in df_valid.columns:
        st.write(f"**Valid coordinates:** {df_valid[['lat', 'lon']].notna().all(axis=1).sum()}")
    if "source" in df_valid.columns:
        st.write(f"**Source values:** {df_valid['source'].value_counts().to_dict()}")

# Filter options
st.sidebar.header("🔍 Filters")
show_airbnb = st.sidebar.checkbox("Show Airbnb Listings", value=True)
show_lahd = st.sidebar.checkbox("Show Affordable Housing", value=True)

# Filter data based on selections
if "source" in df_valid.columns:
    if not show_airbnb or not show_lahd:
        if not show_airbnb:
            df_valid = df_valid[~df_valid["source"].str.contains("airbnb", case=False, na=False)]
        if not show_lahd:
            df_valid = df_valid[~df_valid["source"].str.contains("lahd", case=False, na=False)]

# Price filter
st.sidebar.header("💰 Price Range")
price_min = st.sidebar.number_input("Min Price ($)", min_value=0, value=0, step=10)
price_max = st.sidebar.number_input("Max Price ($)", min_value=0, value=10000, step=10)

if "price" in df_valid.columns and (price_min > 0 or price_max < 10000):
    df_valid = df_valid[
        (df_valid["price"].isna()) | 
        ((df_valid["price"] >= price_min) & (df_valid["price"] <= price_max))
    ]

# Create and display map
st.subheader("Map View")

# Check if we have data to display
if len(df_valid) == 0:
    st.warning("⚠️ No data to display after filtering. Please adjust your filters.")
else:
    # Verify required columns exist
    if "lat" not in df_valid.columns or "lon" not in df_valid.columns:
        st.error("❌ Missing required columns: 'lat' and/or 'lon'. Cannot create map.")
    else:
        with st.spinner("Generating map (this may take a moment for large datasets)..."):
            try:
                map_obj = create_map(df_valid)
                map_data = st_folium(map_obj, width=1000, height=600, returned_objects=["last_object_clicked"], key="housing_map")
                
                # Display clicked marker info
                if map_data and map_data.get("last_object_clicked"):
                    st.info("Click on markers to see detailed information in the popup!")
            except Exception as e:
                st.error(f"❌ Error creating map: {str(e)}")
                st.exception(e)

st.caption("💡 Tip: Hover over markers to see names. Click for full details. Use the layer control to toggle different housing types.")

