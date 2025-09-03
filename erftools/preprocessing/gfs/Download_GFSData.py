# --- Read user values from file ---
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import sys
import os
from tqdm import tqdm
import threading

def read_user_input(filename):
    inputs = {}
    with open(filename, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            key = key.strip()
            value = value.strip()
            if key == "area":
                inputs[key] = [float(x) for x in value.split(',')]
            else:
                inputs[key] = [value]
    return inputs

from datetime import datetime, timedelta

def generate_urls_for_24_hours(data):
    year = data['year'][0]
    month = data['month'][0].zfill(2)
    day = data['day'][0].zfill(2)
    hour = data['time'][0].split(':')[0].zfill(2)

    yyyymmddhh = f"{year}{month}{day}{hour}"
    yyyymmdd = f"{year}{month}{day}"

    urls = []
    filenames = []
    # 24 hours / 3 = 8 forecast times: 0,3,6,...,21
    for fhour in range(0, 123, 3):
        fhour_str = f"f{fhour:03d}"
        filename = f"gfs.0p25.{yyyymmddhh}.{fhour_str}.grib2"
        url = f"https://data-osdf.rda.ucar.edu/ncar/rda/d084001/{year}/{yyyymmdd}/{filename}"
        urls.append(url)
        filenames.append(filename)
    return urls, filenames 

def construct_url_filename(data):

    year = data['year'][0]        # your read_user_input returns a list for year
    month = data['month'][0].zfill(2)
    day = data['day'][0].zfill(2)
    hour = data['time'][0].split(':')[0].zfill(2)

    yyyymmddhh = f"{year}{month}{day}{hour}"
    yyyymm = f"{year}{month}"
    yyyymmdd = f"{year}{month}{day}"

    # Historical forecast data
    filename = f"gfs.0p25.{yyyymmddhh}.f000.grib2"
    url = f"https://data-osdf.rda.ucar.edu/ncar/rda/d084001/{year}/{yyyymmdd}/{filename}"

    # Final reanalaysis data
    #filename = f"gdas1.fnl0p25.{yyyymmddhh}.f00.grib2"
    #url = f"https://data-osdf.rda.ucar.edu/ncar/rda/d083003/{year}/{yyyymm}/{filename}"
    
    print("URL is ", url)

    return url, filename

def Download_GFS_Data(inputs):
    data = read_user_input(inputs)
    lat_max, lon_min, lat_min, lon_max = data.get('area')

    url, filename = construct_url_filename(data)

    print("Download URL:", url)
    print("Filename:", filename)

    import urllib.request
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")
    except Exception as e:
        print("Download failed:", e)

    area = [lat_max, lon_min, lat_min, lon_max]
    return filename, area

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)
    sys.stdout.write(f"\r  Downloaded: {percent:.1f}%")
    sys.stdout.flush()
    if downloaded >= total_size:
        print()  # Newline at end

def download_one_with_progress(url, filename, position):
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pbar.total = total_size
            pbar.update(downloaded - pbar.n)

    pbar = tqdm(total=0, unit='B', unit_scale=True, position=position, desc=os.path.basename(filename))
    try:
        urllib.request.urlretrieve(url, filename, hook)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    pbar.close()


def Download_GFS_ForecastData(inputs):
    data = read_user_input(inputs)
    lat_max, lon_min, lat_min, lon_max = data.get('area')

    urls, filenames = generate_urls_for_24_hours(data)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, (url, fname) in enumerate(zip(urls, filenames)):
            executor.submit(download_one_with_progress, url, fname, i)

    area = [lat_max, lon_min, lat_min, lon_max]
    return filenames, area


