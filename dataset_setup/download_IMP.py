#!/usr/bin/env python3
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

BASE_URL = "https://open-datasets.inesctec.pt/NQ3sxFMZ/IMP-CRS2024-Dataset/CRS_Test/slides/"
TARGET_DIR = "/data/fftmil/IMP/images/"
MAX_WORKERS = 32
RETRY_LIMIT = 3

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_files():
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    return [link for link in links if not link.startswith('?') and not link.startswith('/')]

def is_valid_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def download_file_with_retries(filename):
    url = urljoin(BASE_URL, filename)
    target_path = os.path.join(TARGET_DIR, filename)

    if is_valid_file(target_path):
        print(f"[SKIP] {filename}")
        return filename

    for attempt in range(RETRY_LIMIT):
        try:
            print(f"[{attempt+1}/{RETRY_LIMIT}] Downloading {filename}")
            resp = requests.get(url, stream=True, timeout=10)
            resp.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if is_valid_file(target_path):
                print(f"[DONE] {filename}")
                return filename
            else:
                print(f"[RETRY] {filename} was empty after download.")
        except Exception as e:
            print(f"[ERROR] {filename} – {e}")
        time.sleep(1)
    print(f"[FAILED] {filename} after {RETRY_LIMIT} attempts")
    return filename

def main():
    ensure_dir(TARGET_DIR)
    files = list_files()
    print(f"Found {len(files)} files. Downloading to {TARGET_DIR} using {MAX_WORKERS} threads...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_file_with_retries, f) for f in files]
        for future in as_completed(futures):
            future.result()

    print("All done.")

if __name__ == "__main__":
    main()
