import os
import requests
import zipfile
import io
from datetime import datetime, timedelta

DATA_DIR = "data/raw"
ZIP_URL_PREFIX = "https://s3.amazonaws.com/tripdata/"

def get_recent_file_names(months_back=2):
    """Return JC filenames for the last `months_back` months."""
    now = datetime.now()
    file_names = []

    for i in range(months_back):
        target = now.replace(day=1) - timedelta(days=30 * i)
        year, month = target.year, target.month
        file_names.append(f"JC-{year}{month:02d}-citibike-tripdata.csv.zip")

    return file_names

def download_and_extract(file_name, output_dir=DATA_DIR):
    """Download and extract a ZIP file into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    csv_name = file_name.replace(".zip", ".csv")
    output_path = os.path.join(output_dir, csv_name)

    if os.path.exists(output_path):
        print(f"✅ Already exists: {csv_name}")
        return

    url = ZIP_URL_PREFIX + file_name
    print(f"⬇️  Downloading: {file_name}")

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"❌ File not found: {file_name}")
            return

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(output_dir)
            print(f"📦 Extracted to: {output_dir}")
    except Exception as e:
        print(f"❌ Error downloading {file_name}: {e}")

if __name__ == "__main__":
    files = get_recent_file_names(months_back=2)
    print(f"🗂️  Target files: {files}")
    for file_name in files:
        download_and_extract(file_name)
