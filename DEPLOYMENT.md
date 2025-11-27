# Streamlit Deployment Guide

## Quick Start

### 1. Test Locally First

Run the app locally to make sure everything works:

```bash
streamlit run streamlit_map.py
```

This will open your browser automatically at `http://localhost:8501`

### 2. Deploy to Streamlit Cloud (so it stays online)

#### Step 1: Push to GitHub
Make sure your code is in a GitHub repository:
- `streamlit_app.py` (default entrypoint for Streamlit Cloud)
- `streamlit_map.py` (main app logic)
- `requirements.txt` (dependencies)
- `project5_outputs/combined_housing_west_la.csv` (data file)

> ✅ With `streamlit_app.py` present, Streamlit Cloud automatically picks the correct entrypoint—you don't need to adjust any settings when creating the app.

#### Step 2: Connect to Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Confirm the main file path is prefilled as `streamlit_app.py`
6. Click "Deploy"

#### Step 3: Wait for Deployment
Streamlit Cloud will:
- Install dependencies from `requirements.txt`
- Run your app (and keep it online at the provided URL)
- Give you a public URL (e.g., `https://your-app-name.streamlit.app`)

## 📊 Data Collection Options

### Option A: Use Existing Data (Recommended)
Run the existing data combination script:
```bash
python combine_housing_data.py
```
This combines Airbnb listings and LAHD affordable housing data.

### Option B: Add Scraped Real Estate Data (Advanced)
For educational purposes only, you can scrape additional rental data:
```bash
python scrape_real_estate.py
```
⚠️ **Important**: This scrapes Zillow and Redfin. Use only for educational purposes and check with your instructor.

## 📁 Required Files for Deployment

Make sure these files are in your GitHub repo:
- ✅ `streamlit_map.py` - Main app
- ✅ `requirements.txt` - Dependencies
- ✅ `project5_outputs/combined_housing_west_la.csv` - Data file (or with scraped data)

## Troubleshooting

### If the app doesn't load:
- Check that `combined_housing_west_la.csv` is in the `project5_outputs/` folder
- Verify all dependencies are in `requirements.txt`
- Check the Streamlit Cloud logs for errors

### If the map is slow:
- The app uses caching (`@st.cache_data`) to speed up data loading
- Marker clustering helps with performance
- Consider filtering data if needed

## Notes

- The app is free on Streamlit Cloud
- Your data stays private (only you can see it unless you make the repo public)
- Updates automatically when you push to GitHub

