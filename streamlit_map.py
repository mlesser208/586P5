"""
streamlit_map.py

Streamlit web app for visualizing housing units in Los Angeles.
Deploy to Streamlit Cloud for free hosting.

Run locally: streamlit run streamlit_map.py
Deploy: Push to GitHub and connect to streamlit.io
"""

from collections.abc import Mapping, MutableMapping
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union, cast
import random

import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import streamlit as st
import folium
from folium.plugins import MarkerCluster
try:
    from folium.plugins import FastMarkerCluster
    HAS_FAST_MARKER_CLUSTER = True
except ImportError:
    # FastMarkerCluster may not be available in all folium versions
    HAS_FAST_MARKER_CLUSTER = False
    FastMarkerCluster = None  # type: ignore
from streamlit_folium import st_folium
from pathlib import Path


class ClusterMeta(TypedDict):
    """
    Purpose: Typed dictionary describing cluster metadata.

    Parameters:
        None (used for typing only).

    Return Value:
        Not applicable.

    Exceptions:
        None.
    """

    label: str
    filter: Callable[[str], bool]


ClusterPoint = Tuple[float, float, str]

def spatial_sample_points(points: List[ClusterPoint], max_points: int = 10000) -> List[ClusterPoint]:
    """
    Purpose: Samples points using spatial grid to maintain geographic distribution.
             Divides the map into a grid and samples proportionally from each cell
             to ensure all regions are represented.
    
    Parameters:
        points (List[ClusterPoint]): List of (lat, lon, popup_html) tuples to sample from.
        max_points (int): Maximum number of points to return. Defaults to 10000.
    
    Return Value:
        List[ClusterPoint]: Sampled list of points maintaining geographic distribution.
    
    Exceptions:
        None (returns empty list if points is empty).
    """
    if len(points) <= max_points:
        return points
    
    if not points:
        return []
    
    # Get bounding box of all points
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    # Avoid division by zero if all points are at same location
    if lat_max == lat_min:
        lat_max = lat_min + 0.001
    if lon_max == lon_min:
        lon_max = lon_min + 0.001
    
    # Create grid - use square root to get roughly square grid cells
    # Aim for ~100-400 grid cells for good distribution
    grid_cells = min(400, max(100, int((max_points / 25) ** 0.5) ** 2))
    grid_size = int(grid_cells ** 0.5)
    
    cell_lat = (lat_max - lat_min) / grid_size
    cell_lon = (lon_max - lon_min) / grid_size
    
    # Group points by grid cell
    grid: Dict[Tuple[int, int], List[ClusterPoint]] = {}
    for point in points:
        lat, lon, _ = point
        # Calculate grid cell indices
        i = min(int((lat - lat_min) / cell_lat), grid_size - 1)
        j = min(int((lon - lon_min) / cell_lon), grid_size - 1)
        key = (i, j)
        if key not in grid:
            grid[key] = []
        grid[key].append(point)
    
    # Sample from each grid cell proportionally
    sampled: List[ClusterPoint] = []
    cells_with_data = len(grid)
    if cells_with_data == 0:
        return points[:max_points]
    
    # Calculate points per cell, with some randomness to avoid patterns
    per_cell = max(1, max_points // cells_with_data)
    remaining = max_points - (per_cell * cells_with_data)
    
    for cell_points in grid.values():
        # Take up to per_cell points from this cell
        take = per_cell
        if remaining > 0:
            take += 1
            remaining -= 1
        
        # Randomly sample from cell if it has more points than we need
        if len(cell_points) > take:
            sampled.extend(random.sample(cell_points, take))
        else:
            sampled.extend(cell_points)
    
    # Trim to exact max_points if we went over
    return sampled[:max_points]

# Page config
st.set_page_config(
    page_title="LA Housing Map",
    layout="wide"
)

# Input CSV files (use Path for cross-platform compatibility)
LOCATIONS_CSV = Path("project5_outputs") / "housing_locations.csv"
DETAILS_CSV = Path("project5_outputs") / "housing_details.csv"

# Los Angeles center coordinates
LA_CENTER_LAT = 34.0522
LA_CENTER_LON = -118.2437

@st.cache_data
def load_location_data() -> pd.DataFrame:
    """
    Purpose: Loads lightweight location data for map rendering.
    
    Parameters:
        None (uses module-level LOCATIONS_CSV constant).
    
    Return Value:
        pd.DataFrame: DataFrame containing listing_id, lat, lon, and optional metadata columns
                      required for map filtering.
    
    Exceptions:
        FileNotFoundError: Raised if LOCATIONS_CSV does not exist.
        pd.errors.EmptyDataError: Raised if the CSV is empty.
        KeyError: Raised if required columns (listing_id, lat, lon) are missing.
    """
    csv_path = LOCATIONS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Location CSV file not found: {csv_path}")
    
    df = pd.read_csv(str(csv_path), low_memory=False)
    
    for required in ["listing_id", "lat", "lon"]:
        if required not in df.columns:
            raise KeyError(f"Required column '{required}' not found in {csv_path}")
    
    df["listing_id"] = df["listing_id"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"] != 0) & (df["lon"] != 0)]
    
    if "source" in df.columns:
        df["source"] = df["source"].fillna("").astype(str)
    if "study_area" in df.columns:
        df["study_area"] = df["study_area"].fillna("").astype(str)
    
    return df


@st.cache_data
def load_detail_data() -> pd.DataFrame:
    """
    Purpose: Loads the full housing detail dataset used for filtering and sidebar display.
    
    Parameters:
        None (uses module-level DETAILS_CSV constant).
    
    Return Value:
        pd.DataFrame: Full dataset including metadata columns such as price, beds, baths,
                      address, etc.
    
    Exceptions:
        FileNotFoundError: Raised if DETAILS_CSV does not exist.
        pd.errors.EmptyDataError: Raised if the CSV is empty.
        KeyError: Raised if required column 'listing_id' is missing.
    """
    csv_path = DETAILS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Detail CSV file not found: {csv_path}")
    
    df = pd.read_csv(str(csv_path), low_memory=False)
    if "listing_id" not in df.columns:
        raise KeyError(f"Required column 'listing_id' not found in {csv_path}")
    
    df["listing_id"] = df["listing_id"].astype(str)
    
    for col in ["lat", "lon", "price", "beds", "baths"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "source" in df.columns:
        df["source"] = df["source"].fillna("").astype(str)
    
    return df


def find_listing_id_by_coordinates(
    lat: float,
    lon: float,
    locations_table: pd.DataFrame,
    tolerance: float = 1e-5
) -> Optional[str]:
    """
    Purpose: Finds the listing_id corresponding to a clicked latitude/longitude pair.
    
    Parameters:
        lat (float): Latitude from the map click event.
        lon (float): Longitude from the map click event.
        locations_df (pd.DataFrame): Location dataset containing listing_id, lat, and lon.
        tolerance (float): Allowed difference between stored coordinates and click
                           coordinates. Defaults to 1e-5 (~1 meter).
    
    Return Value:
        Optional[str]: Matching listing_id if found, otherwise None.
    
    Exceptions:
        None (returns None if no match is found).
    """
    if locations_table.empty:
        return None
    
    lat_match = (locations_table["lat"] - lat).abs() <= tolerance
    lon_match = (locations_table["lon"] - lon).abs() <= tolerance
    matches = locations_table[lat_match & lon_match]
    if matches.empty:
        return None
    
    return matches.iloc[0]["listing_id"]

def format_popup_html(row: Union[pd.Series, Mapping[str, Any]]) -> str:
    """
    Purpose: Formats housing data into an HTML popup string for map click display.
    
    Parameters:
        row (Union[pd.Series, Mapping[str, Any]]): Housing data for a single unit.
                        Accepts either a pandas Series or a mapping with keys such as name,
                        source, price, beds, baths, address, city, zip, url, listing_id.
    
    Return Value:
        str: An HTML-formatted string containing the unit's metadata, with missing values
             handled gracefully (shown as "N/A" or omitted).
    
    Exceptions:
        KeyError: Raised if required columns are missing from the row (though the function
                 uses .get() to handle missing keys gracefully).
    """
    if isinstance(row, pd.Series):
        series_row = cast(pd.Series, row)
        data = series_row.to_dict()
    else:
        data = dict(row)

    tooltip_parts = []
    
    # Name
    name_value = data.get("name")
    if pd.notna(name_value) and str(name_value).strip():
        tooltip_parts.append(f"<b>{name_value}</b>")
    
    # Source
    source_value = data.get("source", "Unknown")
    tooltip_parts.append(f"<i>Source: {source_value}</i>")
    
    # Price
    price = data.get("price")
    try:
        if pd.notna(price) and str(price).strip() != "":
            # convert to float safely
            price_val = float(str(price).replace("$", "").replace(",", ""))
            tooltip_parts.append(f"Price: ${price_val:,.0f}")
    except (TypeError, ValueError):
        # If we can't parse it, just skip showing price
        pass
    
    # Beds/Baths
    bed_bath: List[str] = []
    beds_value = data.get("beds")
    baths_value = data.get("baths")
    if pd.notna(beds_value) and str(beds_value).strip():
        bed_bath.append(f"{beds_value} bed(s)")
    if pd.notna(baths_value) and str(baths_value).strip():
        bed_bath.append(f"{baths_value} bath(s)")
    if bed_bath:
        tooltip_parts.append(", ".join(bed_bath))
    
    # Address
    address_parts = []
    if pd.notna(data.get("address")) and str(data.get("address")).strip():
        address_parts.append(str(data.get("address")))
    if pd.notna(data.get("city")) and str(data.get("city")).strip():
        address_parts.append(str(data.get("city")))
    if pd.notna(data.get("zip")) and str(data.get("zip")).strip():
        address_parts.append(str(data.get("zip")))
    if address_parts:
        tooltip_parts.append("Address: " + ", ".join(address_parts))
    
    # URL (if available)
    if pd.notna(data.get("url")) and str(data.get("url")).strip():
        tooltip_parts.append(f"<a href='{data.get('url')}' target='_blank'>View Listing</a>")
    
    return "<br>".join(tooltip_parts)

def create_map_lightweight(
    locations_table: pd.DataFrame,
    initial_zoom: Optional[float] = None,
    map_center: Optional[List[float]] = None
) -> folium.Map:
    """
    Purpose: Creates a Folium map with lightweight markers for efficient rendering when zoomed out.
             Uses FastMarkerCluster for better performance with large datasets.
    
    Parameters:
        locations_table (pd.DataFrame): DataFrame containing housing units with valid coordinates.
                                Expected columns: listing_id, lat, lon, and optional metadata.
        initial_zoom (float): Current zoom level of the map. If None, uses default zoom_start.
        map_center (List[float]): Latitude/longitude pair representing the desired map center.
                                   Defaults to downtown Los Angeles if not provided.
    
    Return Value:
        folium.Map: A Folium map object with lightweight markers and fast clustering.
    
    Exceptions:
        KeyError: Raised if required columns (lat, lon) are missing from the DataFrame.
    """
    zoom_start = int(initial_zoom) if initial_zoom is not None else 11
    center_coords = map_center if map_center is not None else [LA_CENTER_LAT, LA_CENTER_LON]
    m = folium.Map(
        location=center_coords,
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )
    
    # Prepare data for FastMarkerCluster - just coordinates and minimal info
    # FastMarkerCluster is much faster but doesn't support individual popups
    locations = []
    for _, row in locations_table.iterrows():
        locations.append([row["lat"], row["lon"]])
    
    # Use FastMarkerCluster for optimal performance when zoomed out
    # This is much faster than regular MarkerCluster for large datasets
    # Fallback to regular MarkerCluster if FastMarkerCluster is not available
    if HAS_FAST_MARKER_CLUSTER and FastMarkerCluster is not None:
        FastMarkerCluster(
            data=locations,
            name="Housing Units (Fast View)",
            overlay=True,
            control=True
        ).add_to(m)
    else:
        # Fallback: Use regular MarkerCluster with lightweight markers
        fast_cluster = MarkerCluster(
            name="Housing Units (Fast View)",
            overlay=True,
            control=True,
            options={
                "chunkedLoading": True,
                "maxClusterRadius": 80,  # Larger radius for faster clustering
            }
        ).add_to(m)
        # Add simple markers without popups for performance
        for loc in locations:
            folium.Marker(
                location=loc,
                popup=None,  # No popup for lightweight view
                tooltip="Housing Unit"
            ).add_to(fast_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Main app
st.title("Los Angeles Housing Units Map")
st.markdown("Interactive map of Airbnb listings and affordable housing projects in Los Angeles")

# Load data
with st.spinner("Loading housing data..."):
    try:
        locations_df = load_location_data()
        details_df = load_detail_data()
        if len(details_df) == 0:
            st.error("No detail data loaded. Please regenerate the housing dataset.")
            st.stop()
        if len(locations_df) == 0:
            st.error("Location dataset is empty. Ensure geocoded rows exist before rendering the map.")
            st.stop()
    except FileNotFoundError as e:
        st.error(str(e))
        st.error("Please ensure the new CSV files exist in the project5_outputs folder.")
        st.stop()
    except (ParserError, EmptyDataError, KeyError, ValueError, OSError) as load_error:
        st.error(f"Error loading data: {load_error}")
        st.exception(load_error)
        st.stop()

# ============================================================
# SIDEBAR: Overview & Statistics
# ============================================================
st.sidebar.title("📊 Dashboard Overview")

# Calculate statistics
# Handle source column - check for both "airbnb" and "lahd_affordable" or "lahd"
if "source" in details_df.columns:
    # Fill NaN values with empty string for string operations
    source_series = details_df["source"].fillna("").astype(str)
    airbnb_count = len(details_df[source_series.str.contains("airbnb", case=False, na=False)])
    # Match both "lahd_affordable" and any source containing "lahd"
    lahd_count = len(details_df[source_series.str.contains("lahd", case=False, na=False)])
else:
    airbnb_count = 0
    lahd_count = 0
    st.sidebar.warning("⚠️ 'source' column not found in data. Please regenerate data files.")

# Display key metrics in a visually appealing way
st.sidebar.markdown("### 📈 Dataset Statistics")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Units", f"{len(details_df):,}", help="Total housing units in the dataset")
with col2:
    # Calculate percentage breakdown
    if len(details_df) > 0:
        airbnb_pct = (airbnb_count / len(details_df)) * 100
        st.metric("Airbnb %", f"{airbnb_pct:.1f}%", help="Percentage of Airbnb listings")

st.sidebar.markdown("---")

# Source breakdown
st.sidebar.markdown("### 🏠 Housing Sources")

# Check if source column has valid data
if "source" in details_df.columns:
    source_non_null = details_df["source"].notna().sum()
    if source_non_null == 0:
        st.sidebar.error("⚠️ **Data Issue**: Source column is empty. Please regenerate data files by running: `python combine_housing_data.py`")
    else:
        st.sidebar.metric(
            "Airbnb Listings", 
            f"{airbnb_count:,}",
            help="Short-term rental listings from InsideAirbnb dataset"
        )
        st.sidebar.metric(
            "Affordable Housing", 
            f"{lahd_count:,}",
            help="Affordable housing projects from LAHD (Los Angeles Housing Department)"
        )
        
        # Show other sources if they exist
        source_series = details_df["source"].fillna("").astype(str)
        other_sources = details_df[~source_series.str.contains("airbnb|lahd", case=False, na=False)]
        if len(other_sources) > 0:
            st.sidebar.metric("Other Sources", f"{len(other_sources):,}")
else:
    st.sidebar.error("⚠️ **Data Issue**: Source column missing. Please regenerate data files.")

st.sidebar.markdown("---")

# ============================================================
# SIDEBAR: Map Filters
# ============================================================
st.sidebar.markdown("### 🔍 Map Filters")

# Source type filters
st.sidebar.markdown("**Display Options:**")
show_airbnb = st.sidebar.checkbox(
    "Show Airbnb Listings", 
    value=True,
    help="Toggle visibility of Airbnb listings on the map"
)
show_lahd = st.sidebar.checkbox(
    "Show Affordable Housing", 
    value=True,
    help="Toggle visibility of affordable housing projects on the map"
)

# Apply source filters
filtered_details = details_df.copy()
if "source" in filtered_details.columns:
    # Fill NaN values for string operations
    source_series = filtered_details["source"].fillna("").astype(str)
    if not show_airbnb:
        filtered_details = filtered_details[~source_series.str.contains("airbnb", case=False, na=False)]
    if not show_lahd:
        filtered_details = filtered_details[~source_series.str.contains("lahd", case=False, na=False)]

st.sidebar.markdown("---")

# Price filter
st.sidebar.markdown("### 💰 Price Range Filter")
# Calculate price statistics for better defaults from filtered data (after source filter)
if "price" in filtered_details.columns:
    price_data = filtered_details["price"].dropna()
    if len(price_data) > 0:
        price_min_default = max(0, int(price_data.min()))
        price_max_default = max(price_min_default + 100, int(price_data.max()))
        price_median = int(price_data.median())
        price_mean = int(price_data.mean())
    else:
        price_min_default = 0
        price_max_default = 10000
        price_median = 0
        price_mean = 0
else:
    price_min_default = 0
    price_max_default = 10000
    price_median = 0
    price_mean = 0

price_min = st.sidebar.number_input(
    "Minimum Price ($)", 
    min_value=0, 
    value=price_min_default, 
    step=50,
    help="Filter out listings below this price"
)
price_max = st.sidebar.number_input(
    "Maximum Price ($)", 
    min_value=price_min, 
    value=price_max_default, 
    step=50,
    help="Filter out listings above this price"
)

# Show price statistics for currently filtered data
if "price" in filtered_details.columns:
    price_data = filtered_details["price"].dropna()
    if len(price_data) > 0:
        st.sidebar.caption(f"💰 Median: ${price_median:,} | Mean: ${price_mean:,}")

# Apply price filter
if "price" in filtered_details.columns and (price_min > 0 or price_max < 10000):
    filtered_details = filtered_details[
        (filtered_details["price"].isna()) | 
        ((filtered_details["price"] >= price_min) & (filtered_details["price"] <= price_max))
    ]

# Show filtered count
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Results:** {len(filtered_details):,} units")

# Debug info (collapsible, less prominent)
with st.sidebar.expander("🔧 Debug Information"):
    st.write(f"**Detail columns:** {', '.join(details_df.columns.tolist())}")
    st.write(f"**Detail shape:** {details_df.shape}")
    st.write(f"**Location shape:** {locations_df.shape}")
    st.write(f"**Filtered shape:** {filtered_details.shape}")
    if "source" in details_df.columns:
        st.write("**Source breakdown:**")
        source_counts = details_df['source'].value_counts().to_dict()
        for source_name, count in source_counts.items():
            st.write(f"  - {source_name}: {count:,}")

filtered_details_unique = filtered_details.drop_duplicates(subset="listing_id", keep="last")
filtered_ids = set(filtered_details_unique["listing_id"])
filtered_locations = locations_df[locations_df["listing_id"].isin(filtered_ids)]
filtered_detail_lookup = filtered_details_unique.set_index("listing_id", drop=False).to_dict(orient="index")

# Initialize session state
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 11.0
if "selected_listing_id" not in st.session_state:
    st.session_state.selected_listing_id = None
if "map_center" not in st.session_state:
    st.session_state.map_center = [LA_CENTER_LAT, LA_CENTER_LON]

# Create and display map with side detail panel
st.subheader("Map View")
detail_col, map_col = st.columns([1, 3])
map_data: Optional[MutableMapping[str, Any]] = None

with map_col:
    st.write(f"**Data points to display:** {len(filtered_locations):,}")
    if len(filtered_locations) == 0:
        st.warning("No data to display after filtering. Please adjust your filters.")
        map_data = None
    else:
        with st.spinner("Rendering fast map view (location-only data)..."):
            try:
                current_zoom = st.session_state.map_zoom
                current_center = st.session_state.map_center
                map_obj = create_map_lightweight(
                    filtered_locations,
                    initial_zoom=current_zoom,
                    map_center=current_center
                )
                map_data = st_folium(
                    map_obj,
                    height=600,
                    returned_objects=["last_object_clicked", "zoom", "center"],
                    key="housing_map",
                    use_container_width=True
                )
                if map_data:
                    new_zoom = map_data.get("zoom")
                    if isinstance(new_zoom, (float, int)):
                        st.session_state.map_zoom = float(new_zoom)
                    new_center = map_data.get("center")
                    if isinstance(new_center, Mapping):
                        center_lat = new_center.get("lat")
                        center_lon = new_center.get("lng")
                        if isinstance(center_lat, (float, int)) and isinstance(center_lon, (float, int)):
                            st.session_state.map_center = [float(center_lat), float(center_lon)]
            except (ValueError, KeyError, RuntimeError) as map_error:
                map_data = None
                st.error(f"Error creating map: {map_error}")
                st.exception(map_error)

selected_listing: Optional[Dict[str, Any]] = None
if filtered_locations.empty:
    st.session_state.selected_listing_id = None
else:
    click_data = map_data.get("last_object_clicked") if map_data else None
    if isinstance(click_data, Mapping):
        lat_value = click_data.get("lat")
        lon_value = click_data.get("lng")
        if isinstance(lat_value, (float, int)) and isinstance(lon_value, (float, int)):
            listing_id = find_listing_id_by_coordinates(
                float(lat_value),
                float(lon_value),
                filtered_locations
            )
            if listing_id:
                st.session_state.selected_listing_id = listing_id
    selected_id = st.session_state.selected_listing_id
    if selected_id:
        raw_listing = filtered_detail_lookup.get(selected_id)
        if raw_listing is not None:
            selected_listing = cast(Dict[str, Any], raw_listing)
        else:
            st.session_state.selected_listing_id = None

with detail_col:
    st.markdown("### Listing Details")
    if not selected_listing:
        st.info("Click a marker to load its full details here.")
    else:
        st.markdown(format_popup_html(selected_listing), unsafe_allow_html=True)
        listing_id_value = selected_listing.get("listing_id")
        if listing_id_value:
            st.caption(f"Listing ID: {listing_id_value}")

st.caption("Tip: The map now uses a lightweight dataset for rendering. Click any marker to load full details in the left panel without reloading the map.")

