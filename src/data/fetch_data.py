import os
import requests
import zipfile
import io
from datetime import datetime

DATA_DIR = "data/raw"
ZIP_URL_PREFIX = "https://s3.amazonaws.com/tripdata/"

def construct_file_names(start_year=2023, start_month=1):
    """Construct expected file names from start date to current month."""
    now = datetime.now()
    current_year, current_month = now.year, now.month

    year, month = start_year, start_month
    filenames = []

    while (year < current_year) or (year == current_year and month <= current_month):
        name_variants = [
            # f"{year}{month:02d}-citibike-tripdata.csv.zip",
            f"JC-{year}{month:02d}-citibike-tripdata.csv.zip"
        ]
        filenames.extend(name_variants)

        # Increment month
        month += 1
        if month > 12:
            month = 1
            year += 1

    return filenames

def download_and_extract(file_name, output_dir=DATA_DIR):
    """Download and extract a ZIP file into output_dir."""
    url = ZIP_URL_PREFIX + file_name
    print(f"Attempting to download: {file_name}")

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"File not found: {file_name}")
            return False

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(output_dir)
            print(f"Extracted: {file_name}")
        return True
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        return False

def fetch_citibike_data(start_year=2023, start_month=1):
    """Fetch and extract Citi Bike files starting from a given date."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_names = construct_file_names(start_year, start_month)

    print(f"Checking {len(file_names)} file variants...")

    downloaded = 0
    for file_name in file_names:
        csv_name = file_name.replace(".zip", ".csv")
        output_path = os.path.join(DATA_DIR, csv_name)
        if os.path.exists(output_path):
            print(f"Already downloaded: {csv_name}")
            continue
        if download_and_extract(file_name):
            downloaded += 1

    print(f"Finished. Downloaded {downloaded} files.")

if __name__ == "__main__":
    fetch_citibike_data(start_year=2023, start_month=1)
