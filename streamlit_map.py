"""
streamlit_map.py

Streamlit web app for visualizing housing units in Los Angeles.
Deploy to Streamlit Cloud for free hosting.

Run locally: streamlit run streamlit_map.py
Deploy: Push to GitHub and connect to streamlit.io
"""

from typing import Callable, Dict, List, Optional, Tuple, TypedDict
import random

import pandas as pd
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

# Input CSV file (use Path for cross-platform compatibility)
INPUT_CSV = Path("project5_outputs") / "combined_housing_west_la.csv"

# Los Angeles center coordinates
LA_CENTER_LAT = 34.0522
LA_CENTER_LON = -118.2437
DETAIL_ZOOM_LEVEL = 18  # Zoom level at which to show detailed markers with full attributes

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
    
    # Make sure key numeric columns are numeric
    for col in ["lat", "lon", "price", "beds", "baths"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
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
    price = row.get("price")
    try:
        if pd.notna(price) and str(price).strip() != "":
            # convert to float safely
            price_val = float(str(price).replace("$", "").replace(",", ""))
            tooltip_parts.append(f"Price: ${price_val:,.0f}")
    except (TypeError, ValueError):
        # If we can't parse it, just skip showing price
        pass
    
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

def create_map_lightweight(df_valid: pd.DataFrame, current_zoom: Optional[int] = None):
    """
    Purpose: Creates a Folium map with lightweight markers for efficient rendering when zoomed out.
             Uses FastMarkerCluster for better performance with large datasets.
    
    Parameters:
        df_valid (pd.DataFrame): DataFrame containing housing units with valid coordinates.
                                Expected columns: lat, lon, name, source.
        current_zoom (int): Current zoom level of the map. If None, uses default zoom_start.
    
    Return Value:
        folium.Map: A Folium map object with lightweight markers and fast clustering.
    
    Exceptions:
        KeyError: Raised if required columns (lat, lon) are missing from the DataFrame.
    """
    zoom_start = current_zoom if current_zoom is not None else 11
    m = folium.Map(
        location=[LA_CENTER_LAT, LA_CENTER_LON],
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )
    
    # Prepare data for FastMarkerCluster - just coordinates and minimal info
    # FastMarkerCluster is much faster but doesn't support individual popups
    locations = []
    for _, row in df_valid.iterrows():
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


def create_map_detailed(df_valid: pd.DataFrame, current_zoom: Optional[int] = None):
    """
    Purpose: Creates a Folium map with detailed markers including full attribute data.
             Used when zoomed in to show individual house icons with complete information.
    
    Parameters:
        df_valid (pd.DataFrame): DataFrame containing housing units with valid coordinates.
                                Expected columns: lat, lon, name, source, price, beds, baths,
                                address, city, zip, url, listing_id.
        current_zoom (int): Current zoom level of the map. If None, uses default zoom_start.
    
    Return Value:
        folium.Map: A Folium map object with detailed markers and full attribute clusters.
    
    Exceptions:
        KeyError: Raised if required columns (lat, lon) are missing from the DataFrame.
    """
    zoom_start = current_zoom if current_zoom is not None else DETAIL_ZOOM_LEVEL
    m = folium.Map(
        location=[LA_CENTER_LAT, LA_CENTER_LON],
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )
    
    # Use regular MarkerCluster with full attribute data for zoomed-in view
    # Create separate clusters for different sources
    airbnb_cluster = MarkerCluster(
        name="Airbnb Listings",
        overlay=True,
        control=True,
        options={
            "chunkedLoading": True,
            "maxClusterRadius": 50,
        }
    ).add_to(m)
    
    lahd_cluster = MarkerCluster(
        name="LAHD Affordable Housing",
        overlay=True,
        control=True,
        options={
            "chunkedLoading": True,
            "maxClusterRadius": 50,
        }
    ).add_to(m)
    
    other_cluster = MarkerCluster(
        name="Other",
        overlay=True,
        control=True,
        options={
            "chunkedLoading": True,
            "maxClusterRadius": 50,
        }
    ).add_to(m)

    # Add all markers with full popup data to appropriate clusters
    marker_count = 0
    for _, row in df_valid.iterrows():
        source = str(row.get("source", "")).lower() if pd.notna(row.get("source")) else ""
        popup_html = format_popup_html(row)
        
        # Determine which cluster to use
        if "airbnb" in source:
            cluster = airbnb_cluster
        elif "lahd" in source or "affordable" in source:
            cluster = lahd_cluster
        else:
            cluster = other_cluster
        
        # Create marker with full popup data
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=str(row.get("name", "Housing Unit"))[:50],
            icon=folium.Icon(icon="home", prefix="fa")  # House icon for detailed view
        ).add_to(cluster)
        
        marker_count += 1

    # Add layer control
    folium.LayerControl().add_to(m)

    return m


def create_map(df_valid: pd.DataFrame, current_zoom: Optional[int] = None, use_detailed: bool = False):
    """
    Purpose: Creates a Folium map with markers, choosing between lightweight or detailed
             rendering based on zoom level for optimal performance.
    
    Parameters:
        df_valid (pd.DataFrame): DataFrame containing housing units with valid coordinates.
                                Expected columns: lat, lon, name, source, price, beds, baths,
                                address, city, zip, url, listing_id.
        current_zoom (int): Current zoom level of the map. If None, uses default zoom_start.
        use_detailed (bool): If True, uses detailed markers with full attributes. If False,
                            uses lightweight fast clustering. Defaults to False.
    
    Return Value:
        folium.Map: A Folium map object with markers and clusters added.
    
    Exceptions:
        KeyError: Raised if required columns (lat, lon) are missing from the DataFrame.
    """
    if use_detailed:
        return create_map_detailed(df_valid, current_zoom)
    else:
        return create_map_lightweight(df_valid, current_zoom)

# Main app
st.title("Los Angeles Housing Units Map")
st.markdown("Interactive map of Airbnb listings and affordable housing projects in Los Angeles")

# Load data
with st.spinner("Loading housing data..."):
    try:
        df_valid = load_data()
        if len(df_valid) == 0:
            st.error("No data loaded. Please check that the CSV file exists and contains valid data.")
            st.stop()
    except FileNotFoundError as e:
        st.error(f"File not found: {INPUT_CSV}")
        st.error("Please ensure the CSV file exists in the project5_outputs folder.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Sidebar stats
st.sidebar.header("Statistics")
st.sidebar.metric("Total Units", f"{len(df_valid):,}")

# Check if source column exists and calculate counts safely
if "source" in df_valid.columns:
    airbnb_count = len(df_valid[df_valid["source"].str.contains("airbnb", case=False, na=False)])
    lahd_count = len(df_valid[df_valid["source"].str.contains("lahd", case=False, na=False)])
else:
    airbnb_count = 0
    lahd_count = 0
    st.warning("'source' column not found in data")

st.sidebar.metric("Airbnb Listings", f"{airbnb_count:,}")
st.sidebar.metric("Affordable Housing", f"{lahd_count:,}")

# Debug info (expandable)
with st.sidebar.expander("Debug Info"):
    st.write(f"**Columns:** {', '.join(df_valid.columns.tolist())}")
    st.write(f"**Data shape:** {df_valid.shape}")
    if "lat" in df_valid.columns and "lon" in df_valid.columns:
        st.write(f"**Valid coordinates:** {df_valid[['lat', 'lon']].notna().all(axis=1).sum()}")
    if "source" in df_valid.columns:
        st.write(f"**Source values:** {df_valid['source'].value_counts().to_dict()}")

# Filter options
st.sidebar.header("Filters")
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
st.sidebar.header("Price Range")
price_min = st.sidebar.number_input("Min Price ($)", min_value=0, value=0, step=10)
price_max = st.sidebar.number_input("Max Price ($)", min_value=0, value=10000, step=10)

if "price" in df_valid.columns and (price_min > 0 or price_max < 10000):
    df_valid = df_valid[
        (df_valid["price"].isna()) | 
        ((df_valid["price"] >= price_min) & (df_valid["price"] <= price_max))
    ]

# Create and display map
st.subheader("Map View")

# Debug: Show filtered data count
st.write(f"**Data points to display:** {len(df_valid):,}")

# Check if we have data to display
if len(df_valid) == 0:
    st.warning("No data to display after filtering. Please adjust your filters.")
else:
    # Verify required columns exist
    if "lat" not in df_valid.columns or "lon" not in df_valid.columns:
        st.error("Missing required columns: 'lat' and/or 'lon'. Cannot create map.")
    else:
        # Verify coordinates are valid
        valid_coords = df_valid[df_valid["lat"].notna() & df_valid["lon"].notna() &
                               (df_valid["lat"] != 0) & (df_valid["lon"] != 0)]
        if len(valid_coords) == 0:
            st.error("No valid coordinates found in filtered data.")
        else:
            # Initialize session state for zoom tracking
            if "map_zoom" not in st.session_state:
                st.session_state.map_zoom = 11
            if "use_detailed_view" not in st.session_state:
                st.session_state.use_detailed_view = False
            
            with st.spinner("Generating map (this may take a moment for large datasets)..."):
                try:
                    # Determine if we should use detailed view based on zoom level
                    # Get zoom from session state (will be updated after first render)
                    current_zoom = st.session_state.map_zoom
                    use_detailed = current_zoom >= DETAIL_ZOOM_LEVEL
                    
                    # Update session state
                    st.session_state.use_detailed_view = use_detailed
                    
                    # Create map with filtered data
                    if use_detailed:
                        st.write(f"Creating detailed map with {len(valid_coords):,} markers (zoom level: {current_zoom})...")
                        st.info("🔍 Zoomed in: Showing detailed markers with full attribute data and house icons.")
                    else:
                        st.write(f"Creating fast map view with {len(valid_coords):,} markers (zoom level: {current_zoom})...")
                        st.info("⚡ Zoomed out: Using fast clustering for better performance. Zoom in to see detailed information.")
                    
                    map_obj = create_map(valid_coords, current_zoom=current_zoom, use_detailed=use_detailed)
                    
                    # Verify map object was created
                    if map_obj is None:
                        st.error("Map object is None after creation.")
                    else:
                        # Display the map using st_folium
                        # Request zoom level in returned objects to track changes
                        try:
                            map_data = st_folium(
                                map_obj, 
                                height=600, 
                                returned_objects=["last_object_clicked", "zoom"], 
                                key="housing_map",
                                use_container_width=True
                            )
                        except TypeError:
                            # Fallback for older streamlit-folium versions
                            map_data = st_folium(
                                map_obj,
                                height=600,
                                key="housing_map"
                            )
                        
                        # Update zoom level from map if available
                        if map_data and "zoom" in map_data and map_data["zoom"] is not None:
                            new_zoom = map_data["zoom"]
                            if new_zoom != st.session_state.map_zoom:
                                st.session_state.map_zoom = new_zoom
                                # Trigger rerun to update map view
                                st.rerun()
                        
                        # Display clicked marker info
                        if map_data and map_data.get("last_object_clicked"):
                            st.info("Click on markers to see detailed information in the popup!")

                        st.success("Map rendered successfully.")
                except Exception as e:
                    st.error(f"Error creating map: {str(e)}")
                    st.exception(e)
                    st.write("**Debug info:**")
                    st.write(f"- DataFrame shape: {valid_coords.shape}")
                    st.write(f"- Columns: {list(valid_coords.columns)}")
                    if len(valid_coords) > 0:
                        st.write(f"- Sample lat/lon: {valid_coords[['lat', 'lon']].iloc[0].to_dict()}")
                        st.write(f"- Lat range: {valid_coords['lat'].min():.4f} to {valid_coords['lat'].max():.4f}")
                        st.write(f"- Lon range: {valid_coords['lon'].min():.4f} to {valid_coords['lon'].max():.4f}")

st.caption("Tip: When zoomed out, the map uses fast clustering for better performance. Zoom in to see detailed house icons with full attribute data. Use the layer control to toggle different housing types.")

