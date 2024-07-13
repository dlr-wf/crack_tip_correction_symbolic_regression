"""
Script to download the dataset from Zenodo

Melching, D., Strohmann, T., Paysan, F., & Breitbarth, E. (2024).
Physical deep symbolic regression to learn crack tip correction formulas [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.10730749
"""

import requests
import zipfile
from tqdm import tqdm
import os

# URL of the file to be downloaded
url = 'https://zenodo.org/records/10730749/files/crack_tip_correction_symbolic_regression.zip?download=1'

# Send a GET request to the URL
response = requests.get(url, stream=True)

# Get the total file size
file_size = int(response.headers.get('Content-Length', 0))

# Save the content of the response to a file
with open('dataset.zip', 'wb') as file:
    # Initialize the progress bar
    print('Downloading the dataset...')
    with tqdm(response.iter_content(1024), f'Downloading',
              total=file_size,
              unit='B',
              unit_scale=True,
              unit_divisor=1024) as progress:
        for data in progress.iterable:
            # Write data read to the file
            file.write(data)
            # Update the progress bar manually
            progress.update(len(data))

# Open the zip file
print('Extracting the dataset...')
with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    # Extract all the files
    zip_ref.extractall()

# Delete the zip file
print('Deleting the zip file...')
os.remove('dataset.zip')

print('Dataset downloaded successfully!')
