"""
scrape_real_estate.py

Educational web scraping script for SSCI 586 Project 5
Demonstrates rental listing scraping from Zillow and Redfin for Los Angeles area.

⚠️ IMPORTANT LEGAL & ETHICAL NOTICE ⚠️

This script is for EDUCATIONAL PURPOSES ONLY as part of a school project.
- NOT for commercial use
- May violate website Terms of Service
- Use responsibly and at your own risk
- Consult your instructor about scraping policies
- Sites may block your IP address
- Results may be incomplete due to anti-scraping measures

The authors and contributors are not responsible for any misuse.

🔍 SCRAPING LIMITATIONS:
- Zillow actively blocks scrapers (you'll likely get 403 errors)
- Redfin uses JavaScript rendering (basic HTML parsing gets limited data)
- For production use, consider official APIs instead

💡 BETTER ALTERNATIVES:
- Realtor.com API (affordable commercial access)
- Apartments.com API
- Government housing data (HUD, local MLS)
- Licensed data providers

Run: python scrape_real_estate.py
Output: Creates scraped_rentals_la.csv in project5_outputs/
"""

import requests
import pandas as pd
import time
import random
import json
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlencode
import re

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Search parameters
SEARCH_CITY = "Los Angeles"
SEARCH_STATE = "CA"
MAX_PAGES = 5  # Limit pages to be respectful
DELAY_BETWEEN_REQUESTS = 2  # seconds
DELAY_BETWEEN_PAGES = 5     # seconds

# Output
OUTPUT_FOLDER = "project5_outputs"
OUTPUT_FILENAME = "scraped_rentals_la.csv"

# User agents to rotate (helps avoid detection)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# --------------------------------------------------
# DATA SCHEMA (matches your existing combine_housing_data.py)
# --------------------------------------------------

REQUIRED_COLUMNS = [
    'source', 'listing_id', 'name', 'address', 'city', 'state', 'zip',
    'price', 'beds', 'baths', 'sqft', 'lat', 'lon', 'url',
    'study_area', 'full_address'
]

# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------

def get_random_user_agent() -> str:
    """Rotate user agents to avoid detection."""
    return random.choice(USER_AGENTS)

def respectful_delay():
    """Add random delay between requests to be respectful."""
    time.sleep(DELAY_BETWEEN_REQUESTS + random.uniform(0, 1))

def make_request(url: str, headers: Optional[Dict] = None) -> Optional[requests.Response]:
    """Make HTTP request with error handling."""
    default_headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    if headers:
        default_headers.update(headers)

    try:
        response = requests.get(url, headers=default_headers, timeout=10)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

def extract_price(text: str) -> Optional[float]:
    """Extract numeric price from text like '$1,500/mo' or '$2,500'."""
    if not text:
        return None

    # Remove $ and commas, extract numbers
    price_match = re.search(r'\$?([\d,]+)', str(text))
    if price_match:
        try:
            return float(price_match.group(1).replace(',', ''))
        except ValueError:
            pass
    return None

def extract_bed_bath(text: str) -> tuple[Optional[int], Optional[float]]:
    """Extract beds and baths from text like '2 bed, 1 bath'."""
    beds = None
    baths = None

    if not text:
        return beds, baths

    # Look for bed count
    bed_match = re.search(r'(\d+)\s*bed', text, re.IGNORECASE)
    if bed_match:
        beds = int(bed_match.group(1))

    # Look for bath count
    bath_match = re.search(r'(\d+(?:\.\d+)?)\s*bath', text, re.IGNORECASE)
    if bath_match:
        baths = float(bath_match.group(1))

    return beds, baths

def clean_address(address: str) -> str:
    """Clean and standardize address format."""
    if not address:
        return ""

    # Remove extra whitespace
    address = re.sub(r'\s+', ' ', str(address).strip())

    # Remove common unwanted text
    address = re.sub(r'\s*\([^)]*\)\s*', '', address)  # Remove parentheses
    address = re.sub(r'\s*-\s*.*$', '', address)       # Remove trailing dash text

    return address

# --------------------------------------------------
# ZILLOW SCRAPING
# --------------------------------------------------

def scrape_zillow_listings() -> List[Dict]:
    """
    Scrape rental listings from Zillow.
    Note: Zillow's search uses complex APIs, this is a simplified approach.
    """
    listings = []
    base_url = "https://www.zillow.com"

    print("🔍 Scraping Zillow rentals...")

    # Zillow search URL for Los Angeles rentals
    search_params = {
        'searchQueryState': json.dumps({
            'pagination': {},
            'isMapVisible': False,
            'mapBounds': {},
            'usersSearchTerm': f'{SEARCH_CITY}, {SEARCH_STATE}',
            'regionSelection': [{'regionId': 12447, 'regionType': 6}],  # LA region
            'filterState': {
                'fr': {'value': True},  # For rent
                'fsba': {'value': False},  # No FSBO
                'fsbo': {'value': False},  # No For sale by owner
                'nc': {'value': False},   # No new construction
                'cmsn': {'value': False}, # No coming soon
                'auc': {'value': False},  # No auction
                'fore': {'value': False}  # No foreclosure
            },
            'isListVisible': True
        })
    }

    search_url = f"{base_url}/search/GetSearchPageState.htm?{urlencode(search_params)}"

    try:
        response = make_request(search_url)
        if not response:
            print("❌ Failed to get Zillow search results")
            return listings

        data = response.json()

        # Parse the response (this is simplified - Zillow's API structure changes)
        if 'cat1' in data and 'searchResults' in data['cat1']:
            results = data['cat1']['searchResults'].get('listResults', [])

            for result in results[:20]:  # Limit to 20 per page
                try:
                    listing = {
                        'source': 'zillow',
                        'listing_id': str(result.get('zpid', '')),
                        'name': result.get('address', ''),
                        'address': clean_address(result.get('addressStreet', '')),
                        'city': SEARCH_CITY,
                        'state': SEARCH_STATE,
                        'zip': result.get('addressZipcode', ''),
                        'price': extract_price(result.get('price', '')),
                        'beds': result.get('beds'),
                        'baths': result.get('baths'),
                        'sqft': result.get('area'),
                        'lat': result.get('lat'),
                        'lon': result.get('lng'),
                        'url': result.get('detailUrl', ''),
                        'study_area': 'West LA',
                        'full_address': result.get('address', '')
                    }

                    if listing['url'] and not listing['url'].startswith('http'):
                        listing['url'] = base_url + listing['url']

                    listings.append(listing)

                except Exception as e:
                    print(f"Error parsing Zillow listing: {e}")
                    continue

    except Exception as e:
        print(f"Error scraping Zillow: {e}")

    respectful_delay()
    return listings

# --------------------------------------------------
# REDFIN SCRAPING
# --------------------------------------------------

def scrape_redfin_listings() -> List[Dict]:
    """
    Scrape rental listings from Redfin.
    Redfin has a more straightforward search interface.
    """
    listings = []
    base_url = "https://www.redfin.com"

    print("🔍 Scraping Redfin rentals...")

    # Redfin search URL
    search_url = f"{base_url}/city/11203/{SEARCH_CITY.replace(' ', '-')}-{SEARCH_STATE}/filter/include=sold-3yr"

    try:
        response = make_request(search_url)
        if not response:
            print("❌ Failed to get Redfin search page")
            return listings

        # This is a simplified approach - Redfin uses JavaScript rendering
        # In a real implementation, you'd need Selenium or similar for full scraping
        # For now, we'll use a basic HTML parsing approach

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for listing cards (this selector may change)
        listing_cards = soup.find_all('div', class_=re.compile(r'HomeCard|property-card'))

        for card in listing_cards[:20]:  # Limit results
            try:
                # Extract data from card (simplified - real implementation needs more robust parsing)
                address_elem = card.find(['span', 'div'], string=re.compile(r'\d+.*'))
                price_elem = card.find(['span', 'div'], string=re.compile(r'\$[\d,]+'))
                beds_elem = card.find(['span', 'div'], string=re.compile(r'\d+\s*bed'))
                baths_elem = card.find(['span', 'div'], string=re.compile(r'\d+(?:\.\d+)?\s*bath'))

                listing = {
                    'source': 'redfin',
                    'listing_id': f"redfin_{len(listings)}",  # Generate ID since Redfin doesn't expose them easily
                    'name': address_elem.get_text().strip() if address_elem else '',
                    'address': clean_address(address_elem.get_text().strip() if address_elem else ''),
                    'city': SEARCH_CITY,
                    'state': SEARCH_STATE,
                    'zip': '',
                    'price': extract_price(price_elem.get_text() if price_elem else ''),
                    'beds': None,
                    'baths': None,
                    'sqft': None,
                    'lat': None,
                    'lon': None,
                    'url': '',
                    'study_area': 'West LA',
                    'full_address': address_elem.get_text().strip() if address_elem else ''
                }

                # Extract beds/baths
                beds, baths = extract_bed_bath(card.get_text())
                listing['beds'] = beds
                listing['baths'] = baths

                listings.append(listing)

            except Exception as e:
                print(f"Error parsing Redfin listing: {e}")
                continue

    except Exception as e:
        print(f"Error scraping Redfin: {e}")

    respectful_delay()
    return listings

# --------------------------------------------------
# MAIN SCRAPING FUNCTION
# --------------------------------------------------

def scrape_all_sources() -> pd.DataFrame:
    """
    Scrape listings from all sources and combine into DataFrame.
    """
    all_listings = []

    print("🏠 Starting real estate scraping for Los Angeles...")
    print("⚠️  Remember: This is for educational purposes only!")
    print()

    # Scrape Zillow (likely to be blocked)
    print("🔍 Attempting Zillow scraping (note: Zillow heavily blocks scrapers)...")
    zillow_listings = scrape_zillow_listings()
    all_listings.extend(zillow_listings)
    if len(zillow_listings) == 0:
        print("📊 Zillow: 0 listings (blocked or no data)")
        print("💡 Zillow requires sophisticated anti-detection measures")
    else:
        print(f"📊 Zillow: {len(zillow_listings)} listings found")

    # Respectful delay between sites
    time.sleep(DELAY_BETWEEN_PAGES)

    # Scrape Redfin (limited by HTML parsing)
    print("🔍 Attempting Redfin scraping (basic HTML parsing)...")
    redfin_listings = scrape_redfin_listings()
    all_listings.extend(redfin_listings)
    if len(redfin_listings) == 0:
        print("📊 Redfin: 0 listings (parsing failed)")
    else:
        print(f"📊 Redfin: {len(redfin_listings)} listings found (limited data due to JS rendering)")
        print("💡 Redfin data is incomplete - would need Selenium for full scraping")

    # Convert to DataFrame
    df = pd.DataFrame(all_listings)

    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Reorder columns to match schema
    df = df[REQUIRED_COLUMNS]

    print(f"📊 Total scraped listings: {len(df)}")
    return df

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(df: pd.DataFrame) -> Path:
    """
    Save scraped data to CSV file.
    """
    # Create output folder
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / OUTPUT_FILENAME

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"💾 Saved {len(df)} listings to: {output_path}")
    print(f"📊 Sample of data:")
    print(df.head())

    return output_path

# --------------------------------------------------
# INTEGRATION WITH EXISTING WORKFLOW
# --------------------------------------------------

def integrate_with_existing_data(scraped_path: Path) -> Path:
    """
    Combine scraped data with existing Airbnb/LAHD data.
    """
    print("\n🔗 Integrating with existing data...")

    # Load existing combined data
    existing_path = Path("project5_outputs") / "combined_housing_west_la.csv"

    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        scraped_df = pd.read_csv(scraped_path)

        # Combine datasets
        combined_df = pd.concat([existing_df, scraped_df], ignore_index=True)

        # Save updated combined file
        output_path = Path("project5_outputs") / "combined_housing_west_la_with_scraped.csv"
        combined_df.to_csv(output_path, index=False)

        print(f"✅ Combined dataset saved to: {output_path}")
        print(f"📊 Total listings: {len(combined_df)}")
        print(f"   - Original: {len(existing_df)}")
        print(f"   - Scraped: {len(scraped_df)}")

        return output_path
    else:
        print("⚠️  No existing combined data found. Scraped data saved separately.")
        return scraped_path

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():
    """Main scraping workflow."""
    print("=" * 60)
    print("🏠 REAL ESTATE SCRAPING SCRIPT")
    print("   Educational use only for SSCI 586 Project 5")
    print("=" * 60)

    # Scrape data
    scraped_df = scrape_all_sources()

    if len(scraped_df) > 0:
        # Save scraped data
        scraped_path = save_results(scraped_df)

        # Integrate with existing data
        final_path = integrate_with_existing_data(scraped_path)

        print("\n✅ Scraping complete!")
        print(f"📁 Files created:")
        print(f"   - Scraped data: {scraped_path}")
        if final_path != scraped_path:
            print(f"   - Combined data: {final_path}")

        print("\n💡 Next steps:")
        print("   1. Review the scraped data for accuracy")
        print("   2. Geocode addresses if needed (lat/lon)")
        print("   3. Update your Streamlit app to include new data source")

        print("\n🔄 For Better Data Collection:")
        print("   📊 Official APIs (recommended for real projects):")
        print("      - Realtor.com API (~$50/month)")
        print("      - Apartments.com API")
        print("      - Zillow API (Mortgages only, no rentals)")
        print("   🏛️ Government Data:")
        print("      - HUD User datasets")
        print("      - Local housing authority APIs")
        print("   🤝 Licensed Providers:")
        print("      - CoStar, LoopNet, Crexi")
        print("      - Local MLS data through brokers")
    else:
        print("❌ No data scraped. Check error messages above.")

if __name__ == "__main__":
    main()
