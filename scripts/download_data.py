#!/usr/bin/env python3
"""
Script to download images from the Library of Congress IIIF service.
Downloads images numbered from 003 to 049.
"""

import requests
import os
from pathlib import Path

# Base URL template
URL_TEMPLATE = "https://tile.loc.gov/image-services/iiif/service:gdc:gdcwdl:wd:l_:13:51:6:wdl_13516:{number}/full/pct:100/0/default.jpg"

# List of image numbers to exclude (add numbers here that you don't want to download)
# Example: EXCLUDE_NUMBERS = [5, 12, 23, 37]
EXCLUDE_NUMBERS = [16, 17, 18, 24, 25, 27, 28, 30, 38]

# Create output directory
output_dir = Path("data/datasets/book1/")
output_dir.mkdir(exist_ok=True)

print(f"Starting download of images 003-049 to '{output_dir}' directory...\n")

# Download images from 003 to 049
successful = 0
failed = 0
skipped = 0

for num in range(3, 50):  # 3 to 49 inclusive
    # Format number with leading zeros (003, 004, etc.)
    number_str = f"{num:03d}"
    
    # Check if this number should be excluded
    if num in EXCLUDE_NUMBERS:
        print(f"Skipping image {number_str} (in exclude list)")
        skipped += 1
        continue
    
    # Build URL
    url = URL_TEMPLATE.format(number=number_str)
    
    # Output filename
    filename = output_dir / f"wdl_13516_{number_str}.jpg"
    
    try:
        print(f"Downloading image {number_str}... ", end="", flush=True)
        
        # Download the image
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Save the image
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Saved to {filename}")
        successful += 1
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed: {e}")
        failed += 1

# Summary
print(f"\n{'='*50}")
print(f"Download complete!")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Skipped: {skipped}")
print(f"Total processed: {successful + failed + skipped}")
print(f"Images saved in: {output_dir.absolute()}")